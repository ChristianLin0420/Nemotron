#!/usr/bin/env python3
"""Convert CR3's LLaVA-style JSON datasets to Energon ChatMLWebdataset shards.

This is the data port between the CR3 (cosmos-rl) trainer and the Nemotron
omni3 SFT recipe. The on-disk schema we emit is the one consumed by
megatron.bridge.data.energon.task_encoder_utils.ChatMLWebdataset, which is
what `nemotron_omni_valor32k_sft_config` (our chosen recipe) loads.

Per-sample shard contents:
    __key__               unique id within the shard
    conversation.json     OpenAI-style ChatML messages (role/content list)
    video.mp4             raw mp4 bytes
    audio.wav             16 kHz mono PCM (real audio if present, else silence)

Usage (single TOML):
    python cr3_to_energon.py \\
        --cr3-toml /workspace/cosmos-reason2/examples/cosmos_rl/configs/cr3-8b/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds1.toml \\
        --output   /lustre/fsw/.../chrislin/datasets/cr3-nemotron/energon/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds1 \\
        --val-fraction 0.1

Usage (batch all 28 CR3 TOMLs):
    python cr3_to_energon.py \\
        --cr3-tomls-dir /workspace/cosmos-reason2/examples/cosmos_rl/configs/cr3-8b \\
        --output-root   /lustre/fsw/.../chrislin/datasets/cr3-nemotron/energon \\
        --val-fraction 0.1

Run inside the omni3-sft container — needs `webdataset` (pip install webdataset)
and `ffmpeg` on PATH (already present in the container). For interactive
testing, allocate a node with cr3-nemotron/interactive.sh first.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import uuid
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("cr3_to_energon")


# Energon dataset.yaml schema — copied from
# Nemotron/src/nemotron/data_prep/recipes/sft_omni.py:_DATASET_YAML so the
# resulting datasets are loaded by the same ChatMLWebdataset class the recipe
# uses. If megatron-bridge changes this contract, the dataset will build but
# silently misread — keep this in sync.
DATASET_YAML = {
    "__module__": "megatron.bridge.data.energon.task_encoder_utils",
    "__class__": "ChatMLWebdataset",
    "field_map": {
        "conversation": "conversation.json",
        "audio": "audio.wav",
        "videos": "video.mp4",
    },
    "subflavors": {},
}


def _resolve_ffmpeg() -> str | None:
    """Return a usable ffmpeg binary path, or None if none is available.

    Tries (in order):
      1. ``CR3_FFMPEG`` env var override
      2. system ``ffmpeg`` on PATH (``shutil.which``)
      3. ``imageio_ffmpeg.get_ffmpeg_exe()`` (pip-installable static binary)

    Audio is **frozen** in cr3_base.yaml, so the converter only really needs
    ANY 16 kHz mono PCM payload — real audio is a nice-to-have, not a must.
    See ``_make_silent_wav`` for the no-ffmpeg fallback.
    """
    import shutil
    ev = os.environ.get("CR3_FFMPEG")
    if ev and Path(ev).exists():
        return ev
    binp = shutil.which("ffmpeg")
    if binp:
        return binp
    try:
        import imageio_ffmpeg  # optional
        binp = imageio_ffmpeg.get_ffmpeg_exe()
        if binp and Path(binp).exists():
            return binp
    except ImportError:
        pass
    return None


def _make_silent_wav(audio_path: Path, duration_sec: float = 1.0) -> None:
    """Write a 16 kHz mono PCM ``.wav`` of silence to ``audio_path``.

    No ffmpeg required — uses the stdlib ``wave`` module. The frozen sound
    encoder in cr3_base.yaml means the encoder runs in the forward pass but
    contributes no gradients, so silence is loss-equivalent to real audio
    for our purposes. Generous default duration (1s) so Parakeet's framing
    has at least one frame.
    """
    import wave
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    nsamples = max(1, int(duration_sec * sample_rate))
    silence = b"\x00\x00" * nsamples  # 16-bit signed PCM zeros
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)         # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(silence)


def _ffmpeg_extract_audio(video_path: Path, audio_path: Path,
                          ffmpeg_bin: str | None) -> None:
    """Best-effort 16 kHz mono PCM extraction from ``video_path``.

    If ffmpeg is available, try to extract the real audio track from the
    video. If extraction fails (no track, decode error) or ffmpeg is not
    available at all, fall back to a 1s silent WAV via ``_make_silent_wav``.
    The frozen audio encoder in cr3_base.yaml means either path is loss-
    equivalent for training; real audio is preferred only if you ever
    unfreeze the audio encoder.
    """
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    if ffmpeg_bin is not None:
        proc = subprocess.run(
            [ffmpeg_bin, "-y", "-loglevel", "error",
             "-i", str(video_path),
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
             str(audio_path)],
            capture_output=True,
        )
        if proc.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 44:
            # 44 bytes is just the WAV header; reject empty/header-only outputs
            return
        # Fall through — extraction didn't yield audio (silent video, weird
        # codec, container missing audio stream); silence placeholder is fine.

    _make_silent_wav(audio_path, duration_sec=1.0)


def _build_chatml(human_text: str, assistant_text: str, system_prompt: str | None) -> str:
    """Serialize a single CR3 turn into Energon ChatML conversation JSON.

    ChatMLWebdataset expects a list of {"role", "content"} where content is
    a list of typed parts. We attach the video to the user turn.
    """
    # Strip CR3's literal "<video>\n" sentinel — the typed video part below
    # tells the processor where to inject vision tokens; the literal string
    # would either be tokenized as text or be a duplicate sentinel.
    if human_text.startswith("<video>\n"):
        human_text = human_text[len("<video>\n"):]
    elif human_text.startswith("<video>"):
        human_text = human_text[len("<video>"):]

    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": human_text},
    ]})
    messages.append({"role": "assistant", "content": [
        {"type": "text", "text": assistant_text},
    ]})
    return json.dumps(messages, ensure_ascii=False)


def _coerce_str_list(value) -> list[str]:
    """Normalise a TOML field that *should* be a list of strings.

    The CR3-8B template TOMLs (cosmos-reason2/examples/cosmos_rl/configs/cr3-8b/...)
    store ``name`` and ``split`` as triple-quoted strings whose contents are a
    Python-list literal::

        name = '''[
            "/lustre/.../bcq.json",
            "/lustre/.../mcq.json",
            ...
        ]'''

    Other CR3 TOMLs (the in-trainer-saved ``outputs/.../*_config.toml``) use a
    proper TOML array. Handle both: if it's already a list, return it; if it's
    a string, ``ast.literal_eval`` it.
    """
    import ast as _ast
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = _ast.literal_eval(s)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"could not parse string-encoded list (got {type(value).__name__}): "
                f"first 200 chars: {s[:200]!r}"
            ) from e
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(
                f"string did not evaluate to a list/tuple, got {type(parsed).__name__}"
            )
        return [str(v) for v in parsed]
    raise TypeError(f"expected list or string, got {type(value).__name__}")


def _load_cr3_toml(toml_path: Path) -> dict:
    """Load a CR3 TOML and surface the bits the converter needs."""
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)
    ds = cfg.get("train", {}).get("train_policy", {}).get("dataset", {})
    custom = cfg.get("custom", {}) or {}
    return {
        "json_paths": _coerce_str_list(ds.get("name")),
        "splits":     _coerce_str_list(ds.get("split")),
        "test_size":  float(ds.get("test_size", 0.1) or 0.1),
        "system_prompt": custom.get("system_prompt", None),
        # Vision config from CR3's TOML — informational; Nemotron processor
        # has its own video sampling. We log these so the operator can verify.
        "fps":         (cfg.get("custom", {}).get("vision", {}) or {}).get("fps"),
        "max_frames":  (cfg.get("custom", {}).get("vision", {}) or {}).get("max_frames"),
    }


def _resolve_cr3_records(json_paths: list[str], splits: list[str],
                         dataset_root_override: Path | None) -> list[tuple[str, dict, Path]]:
    """Read each CR3 JSON and emit (split_label, record, json_dir) triples.

    json_dir is the directory containing the JSON, used to resolve relative
    video paths in the record. dataset_root_override lets the user remap the
    /lustre/fsw/.../lliou/... paths in the TOML to a locally-accessible
    mirror (e.g. cosmos-reason2/datasets/) without editing the TOML.
    """
    if splits and len(splits) != len(json_paths):
        logger.warning("split list length (%d) != json_paths length (%d); "
                       "labels will be auto-generated for the mismatch",
                       len(splits), len(json_paths))

    out: list[tuple[str, dict, Path]] = []
    for i, jp in enumerate(json_paths):
        json_path = Path(jp)
        if dataset_root_override is not None and not json_path.exists():
            # Try mapping the well-known CR3 data root to the override
            for prefix in (
                "/lustre/fsw/portfolios/edgeai/users/lliou/multi-modality-research/VILA/sample_data/",
                "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/cr/cosmos-reason2/datasets/",
            ):
                if jp.startswith(prefix):
                    rel = jp[len(prefix):]
                    candidate = dataset_root_override / rel
                    if candidate.exists():
                        json_path = candidate
                        break

        if not json_path.exists():
            logger.warning("Skipping missing JSON: %s", jp)
            continue

        label = splits[i] if i < len(splits) else f"split_{i:03d}"
        with open(json_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON %s: %s", json_path, e)
                continue

        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            logger.warning("Skipping %s: top level is %s, expected list",
                           json_path, type(data).__name__)
            continue

        for rec in data:
            out.append((label, rec, json_path.parent))

    return out


def _stratified_split(records: list[tuple[str, dict, Path]], val_fraction: float,
                      seed: int) -> tuple[list[tuple[str, dict, Path]], list[tuple[str, dict, Path]]]:
    """Per-split-label train/val partition so each label keeps its proportion."""
    rng = random.Random(seed)
    by_label: dict[str, list] = defaultdict(list)
    for label, rec, jdir in records:
        by_label[label].append((label, rec, jdir))
    train, val = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction))) if items else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _write_shard(shard_path: Path, samples: list[dict]) -> int:
    """Write one Energon-compatible tar shard. Mirrors WebDatasetShardStage._write_one_shard."""
    import webdataset as wds  # local — only available inside omni3-sft container
    written = 0
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with wds.TarWriter(str(shard_path)) as sink:
        for sample in samples:
            sink.write(sample)
            written += 1
    return written


def _build_nv_meta(dataset_path: Path, split_shards: dict[str, list[str]]) -> int:
    """Write .nv-meta/ — index.sqlite, .info.yaml, split.yaml, dataset.yaml.

    Logic ported from Nemotron/src/nemotron/data_prep/recipes/sft_omni.py:_build_energon_index
    so we don't depend on `energon prepare` (which has a known deadlock on the
    pinned megatron-energon).
    """
    import yaml

    meta_dir = dataset_path / ".nv-meta"
    meta_dir.mkdir(exist_ok=True)

    ordered_shards = [
        name for split in ("train", "val", "test") for name in split_shards.get(split, [])
    ]

    db_path = meta_dir / "index.sqlite"
    db_path.unlink(missing_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE samples ("
        "  id           INTEGER PRIMARY KEY,"
        "  tar_file_id  INTEGER,"
        "  sample_key   TEXT,"
        "  sample_index INTEGER,"
        "  byte_offset  INTEGER,"
        "  byte_size    INTEGER)"
    )
    conn.execute("CREATE INDEX idx_samples_sample_key ON samples(sample_key)")

    # The fields each sample writes — their names come from the writer's
    # ``DATASET_YAML["field_map"]`` and become the file extensions on the
    # WebDataset side. We strip the longest matching extension to recover
    # ``__key__``; splitting at the first ``.`` would mangle keys that
    # themselves contain ``.`` (e.g. version-tagged split labels like
    # ``ASSY17_v1.2_mcq``).
    KNOWN_FIELDS = ("conversation.json", "video.mp4", "audio.wav")

    def _key_of(member_name: str) -> str:
        for f in KNOWN_FIELDS:
            suffix = "." + f
            if member_name.endswith(suffix):
                return member_name[: -len(suffix)]
        # Fallback for unexpected members (e.g. PaxHeader entries) — group
        # them by the first dot-segment so they don't crash the indexer.
        return member_name.split(".", 1)[0]

    shard_counts: dict[str, int] = {}
    for tar_file_id, shard_name in enumerate(ordered_shards):
        with tarfile.open(dataset_path / shard_name) as tf:
            members = tf.getmembers()
        groups: dict[str, list] = defaultdict(list)
        for m in members:
            groups[_key_of(m.name)].append(m)
        ordered_groups = sorted(groups.items(), key=lambda kv: min(m.offset for m in kv[1]))
        rows = []
        for sample_index, (sample_key, mems) in enumerate(ordered_groups):
            mems_sorted = sorted(mems, key=lambda m: m.offset)
            byte_offset = mems_sorted[0].offset
            if sample_index + 1 < len(ordered_groups):
                byte_size = min(m.offset for m in ordered_groups[sample_index + 1][1]) - byte_offset
            else:
                last = mems_sorted[-1]
                byte_size = last.offset_data + ((last.size + 511) // 512) * 512 - byte_offset
            rows.append((tar_file_id, sample_key, sample_index, byte_offset, byte_size))
        conn.executemany(
            "INSERT INTO samples (tar_file_id, sample_key, sample_index, byte_offset, byte_size)"
            " VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
        shard_counts[shard_name] = len(rows)
    conn.close()
    total = sum(shard_counts.values())

    (meta_dir / ".info.yaml").write_text(yaml.dump({"shard_counts": shard_counts}))
    (meta_dir / "index.uuid").write_text(str(uuid.uuid4()))
    (meta_dir / "split.yaml").write_text(yaml.dump({"split_parts": dict(split_shards), "exclude": []}))
    (meta_dir / "dataset.yaml").write_text(yaml.dump(DATASET_YAML, sort_keys=False))
    return total


def convert_one(toml_path: Path, output: Path, val_fraction: float,
                samples_per_shard: int, dataset_root_override: Path | None,
                tmp_audio_dir: Path, seed: int = 42) -> dict:
    """End-to-end: load CR3 TOML, write Energon shards + .nv-meta to ``output``."""
    cfg = _load_cr3_toml(toml_path)
    logger.info("[%s] loaded TOML: %d JSON paths, %d split labels, fps=%s, max_frames=%s",
                toml_path.name, len(cfg["json_paths"]), len(cfg["splits"]),
                cfg["fps"], cfg["max_frames"])

    records = _resolve_cr3_records(cfg["json_paths"], cfg["splits"], dataset_root_override)
    if not records:
        raise RuntimeError(f"No records resolved from {toml_path}")
    logger.info("[%s] resolved %d records across %d split labels",
                toml_path.name, len(records),
                len({lbl for lbl, _, _ in records}))

    train_recs, val_recs = _stratified_split(records, val_fraction, seed)
    logger.info("[%s] split: train=%d val=%d (val_fraction=%.2f)",
                toml_path.name, len(train_recs), len(val_recs), val_fraction)

    if output.exists():
        logger.warning("Removing existing output dir: %s", output)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = _resolve_ffmpeg()
    if ffmpeg_bin:
        logger.info("[%s] using ffmpeg at %s for real audio extraction",
                    toml_path.name, ffmpeg_bin)
    else:
        logger.info("[%s] no ffmpeg on PATH and imageio_ffmpeg not installed — "
                    "writing 1s silent placeholders to audio.wav (frozen audio "
                    "encoder in cr3_base.yaml makes this loss-equivalent)",
                    toml_path.name)

    split_shards: dict[str, list[str]] = {}
    audio_cache: dict[str, Path] = {}

    def _emit(split_name: str, recs: list[tuple[str, dict, Path]]) -> None:
        if not recs:
            return
        shard_idx = 0
        sample_idx_in_shard = 0
        current_samples: list[dict] = []

        def _flush() -> None:
            nonlocal shard_idx, sample_idx_in_shard, current_samples
            if not current_samples:
                return
            shard_name = f"{split_name}-shard-{shard_idx:06d}.tar"
            n = _write_shard(output / shard_name, current_samples)
            split_shards.setdefault(split_name, []).append(shard_name)
            logger.info("[%s] wrote %s (%d samples)", toml_path.name, shard_name, n)
            shard_idx += 1
            sample_idx_in_shard = 0
            current_samples = []

        for split_label, rec, json_dir in recs:
            video_rel = rec.get("video") or rec.get("media") or rec.get("video_path")
            if not video_rel:
                logger.debug("Skipping record with no video field: %s",
                             rec.get("id"))
                continue
            video_path = (json_dir / video_rel).resolve()
            if not video_path.exists():
                logger.warning("Missing video %s (rec id=%s)", video_path, rec.get("id"))
                continue

            convs = rec.get("conversations") or []
            if len(convs) < 2:
                logger.warning("Record %s has < 2 conversation turns; skipping", rec.get("id"))
                continue
            human = next((c.get("value", "") for c in convs if c.get("from") == "human"), None)
            asst  = next((c.get("value", "") for c in convs if c.get("from") == "gpt"), None)
            if human is None or asst is None:
                logger.warning("Record %s missing human/gpt turn; skipping", rec.get("id"))
                continue

            # Cache audio extraction per video — multiple QA pairs share one mp4
            audio_path = audio_cache.get(str(video_path))
            if audio_path is None:
                audio_path = tmp_audio_dir / f"{video_path.stem}_{abs(hash(str(video_path))) & 0xFFFFFFFF:08x}.wav"
                if not audio_path.exists():
                    _ffmpeg_extract_audio(video_path, audio_path, ffmpeg_bin)
                audio_cache[str(video_path)] = audio_path

            chatml = _build_chatml(human, asst, cfg.get("system_prompt"))

            current_samples.append({
                "__key__": f"{split_label}_{shard_idx:06d}_{sample_idx_in_shard:08d}",
                "conversation.json": chatml.encode("utf-8"),
                "video.mp4":         video_path.read_bytes(),
                "audio.wav":         audio_path.read_bytes(),
            })
            sample_idx_in_shard += 1
            if sample_idx_in_shard >= samples_per_shard:
                _flush()
        _flush()

    _emit("train", train_recs)
    _emit("val",   val_recs)

    total = _build_nv_meta(output, split_shards)
    logger.info("[%s] DONE: %d samples across %d shards under %s",
                toml_path.name, total,
                sum(len(v) for v in split_shards.values()), output)
    return {"toml": str(toml_path), "output": str(output),
            "n_train": len(train_recs), "n_val": len(val_recs),
            "n_total": total, "shards": split_shards}


def _toml_to_run_name(toml_path: Path) -> str:
    """Strip ``.toml`` to get the per-config run name (matches CR3 sbatch naming)."""
    return toml_path.stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cr3-toml", type=Path,
                   help="single CR3 TOML; --output specifies the dataset path")
    g.add_argument("--cr3-tomls-dir", type=Path,
                   help="walk all *.toml under this dir (recurse one level)")

    ap.add_argument("--output", type=Path,
                    help="single-TOML mode: Energon dataset output dir")
    ap.add_argument("--output-root", type=Path,
                    help="batch mode: <output-root>/<dataset>/<run-name>/")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--samples-per-shard", type=int, default=100)
    ap.add_argument("--dataset-root-override", type=Path, default=None,
                    help="If set, remap CR3 dataset paths starting with the "
                         "well-known lliou prefix to this directory (e.g. "
                         "/workspace/cosmos-reason2/datasets when running "
                         "interactively without /lustre/fsw mounted).")
    ap.add_argument("--tmp-audio-dir", type=Path, default=None,
                    help="Where to cache extracted WAVs (default: <output>/.audio_cache)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.cr3_toml is not None:
        if not args.output:
            ap.error("--output required with --cr3-toml")
        tmp_audio_dir = args.tmp_audio_dir or args.output / ".audio_cache"
        tmp_audio_dir.mkdir(parents=True, exist_ok=True)
        result = convert_one(
            args.cr3_toml.resolve(), args.output.resolve(),
            args.val_fraction, args.samples_per_shard,
            args.dataset_root_override.resolve() if args.dataset_root_override else None,
            tmp_audio_dir,
        )
        print(json.dumps(result, indent=2))
        return 0

    # Batch mode
    if not args.output_root:
        ap.error("--output-root required with --cr3-tomls-dir")
    out_root: Path = args.output_root.resolve()
    src_root: Path = args.cr3_tomls_dir.resolve()

    tomls = sorted(src_root.rglob("*.toml"))
    if not tomls:
        logger.error("No *.toml files found under %s", src_root)
        return 1

    summary: list[dict] = []
    for toml_path in tomls:
        # cr3-8b/<dataset>/<config>.toml -> dataset = parent dirname
        dataset = toml_path.parent.name
        run_name = _toml_to_run_name(toml_path)
        out = out_root / dataset / run_name
        if (out / ".nv-meta" / "dataset.yaml").exists():
            logger.info("[%s] skip — .nv-meta/dataset.yaml already present at %s",
                        toml_path.name, out)
            summary.append({"toml": str(toml_path), "output": str(out), "skipped": True})
            continue
        tmp_audio_dir = args.tmp_audio_dir or (out / ".audio_cache")
        tmp_audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = convert_one(
                toml_path, out, args.val_fraction, args.samples_per_shard,
                args.dataset_root_override.resolve() if args.dataset_root_override else None,
                tmp_audio_dir,
            )
            summary.append(result)
        except Exception:  # noqa: BLE001
            logger.exception("[%s] FAILED", toml_path.name)
            summary.append({"toml": str(toml_path), "output": str(out), "failed": True})

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
