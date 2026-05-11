#!/bin/bash
# End-to-end CR3 SFT interactive runner. Run this *inside* the omni3-sft
# container (after `bash Nemotron/cr3/interactive.sh` drops you into a shell)
# and it will, for one CR3 run:
#
#   1. Resolve all paths from $CR3_RUN (e.g. assy17_lr15e5_lr1e6_dmcq2p2n_ds1).
#   2. Sanity-check that the source TOML, per-run YAML, and pretrain ckpt
#      directory all exist.
#   3. Convert CR3 TOML -> Energon WebDataset shards via cr3_to_energon.py.
#      Idempotent: skips fresh datasets, regenerates pre-fc4ec4c dotted-key
#      shards in place (see _first_tar_member_has_dotted_stem).
#   4. Patch Megatron-Bridge's fusions.py so the apex extension check no
#      longer falsely reports True via TE imports (the image is launched
#      with --rm so the patch must reapply each session). Same patch as
#      cr3/test/scripts/run_train_smoke.sh.
#   5. Install pydantic-settings into the Megatron-Bridge .venv if absent.
#   6. torchrun the per-run YAML on 8 GPUs (TP=2, EP=4, PP=1; matches the
#      assy17 sbatch wrappers). The recipe is whatever cr3_base.yaml's
#      `recipe.name` resolves to — currently LoRA (peft_config). Full-LM SFT
#      (sft_config) doesn't fit on a single 8 x A100-80 node (DDP buffer
#      ~56 GiB + Adam state ~84 GiB on top of the ~60 GiB model); flip
#      cr3_base.yaml back to sft_config only on a >= 2-node allocation.
#      Hydra struct mode means recipe.name CANNOT be overridden via CLI
#      (the key is consumed at dispatch); edit cr3_base.yaml directly.
#
# Usage:
#     bash /workspace/Nemotron/cr3/run_interactive_sft.sh                          # default run
#     bash /workspace/Nemotron/cr3/run_interactive_sft.sh assy17_lr15e5_lr1e6_dmcq1p2n_ds1
#     CR3_RUN=tp00302_dmcq2p2n_ds1 bash /workspace/Nemotron/cr3/run_interactive_sft.sh
#
# Env knobs (defaults in []):
#     CR3_RUN                    [assy17_lr15e5_lr1e6_dmcq2p2n_ds1]
#     CR3_DATASET                [<first _-segment of $CR3_RUN>, e.g. assy17]
#     CR3_LUSTRE_USER_ROOT       [/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin]
#                                  Matches the path the assy17 sbatch wrappers
#                                  hardcode; override for a different user.
#     CR3_TRAIN_ITERS            [4000]
#     CR3_LM_LR                  [1.5e-5]
#     CR3_VAL_FRACTION           [0.1]
#     CR3_SAMPLES_PER_SHARD      [100]
#     OMNI3_MEGATRON_CHECKPOINT  [/workspace/Nemotron/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16]
#                                  Megatron-format pretrain checkpoint dir
#                                  (override if your import landed elsewhere,
#                                  e.g. $CR3_NEMOTRON_CACHE/checkpoints/nemotron_omni).
#
# Outputs:
#     $CR3_LUSTRE_USER_ROOT/datasets/cr3-nemotron/energon/$CR3_DATASET/$CR3_RUN
#     $CR3_LUSTRE_USER_ROOT/cr3-nemotron/ckpt/$CR3_DATASET/$CR3_RUN

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Resolve run name + paths.
# ---------------------------------------------------------------------------
CR3_RUN="${1:-${CR3_RUN:-assy17_lr15e5_lr1e6_dmcq2p2n_ds1}}"
CR3_DATASET="${CR3_DATASET:-${CR3_RUN%%_*}}"
CR3_LUSTRE_USER_ROOT="${CR3_LUSTRE_USER_ROOT:-/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin}"

CR3_TOML="/workspace/cosmos-reason2/examples/cosmos_rl/configs/cr3-8b/$CR3_DATASET/$CR3_RUN.toml"
CR3_YAML="/workspace/Nemotron/src/nemotron/recipes/omni3/stage0_sft/config/cr3/$CR3_DATASET/$CR3_RUN.yaml"
export CR3_ENERGON_PATH="$CR3_LUSTRE_USER_ROOT/datasets/cr3-nemotron/energon/$CR3_DATASET/$CR3_RUN"
export CR3_CKPT_SAVE="$CR3_LUSTRE_USER_ROOT/cr3-nemotron/ckpt/$CR3_DATASET/$CR3_RUN"

# Training knobs (mirror sbatch_<run>.sh defaults).
# The active recipe is whatever cr3_base.yaml:recipe.name resolves to
# (Hydra consumes that key at dispatch, so it cannot be overridden via CLI).
# We default to LoRA (peft_config) because full-LM SFT (sft_config) doesn't
# fit on a single 8 x A100-80 node — see cr3_base.yaml's comment.
export CR3_TRAIN_ITERS="${CR3_TRAIN_ITERS:-4000}"
export CR3_LM_LR="${CR3_LM_LR:-1.5e-5}"
CR3_VAL_FRACTION="${CR3_VAL_FRACTION:-0.1}"
CR3_SAMPLES_PER_SHARD="${CR3_SAMPLES_PER_SHARD:-100}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# OMNI3_MEGATRON_CHECKPOINT defaults to the in-container HF-imported Megatron
# checkpoint dir (run_import_ckpt.sh's standard destination). Override if your
# import landed on lustre, e.g.
#     export OMNI3_MEGATRON_CHECKPOINT=$CR3_NEMOTRON_CACHE/checkpoints/nemotron_omni
export OMNI3_MEGATRON_CHECKPOINT="${OMNI3_MEGATRON_CHECKPOINT:-/workspace/Nemotron/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"

echo "=== resolved ==="
echo "  run        : $CR3_RUN"
echo "  dataset    : $CR3_DATASET"
echo "  recipe     : $(grep -E '^\s*name:' /workspace/Nemotron/src/nemotron/recipes/omni3/stage0_sft/config/cr3_base.yaml | head -1 | awk '{print $2}')"
echo "  TOML       : $CR3_TOML"
echo "  YAML       : $CR3_YAML"
echo "  energon out: $CR3_ENERGON_PATH"
echo "  ckpt out   : $CR3_CKPT_SAVE"
echo "  pretrain   : $OMNI3_MEGATRON_CHECKPOINT"
echo "  iters      : $CR3_TRAIN_ITERS    lm_lr: $CR3_LM_LR"
echo

# ---------------------------------------------------------------------------
# 1. Sanity checks (fail fast with readable errors).
# ---------------------------------------------------------------------------
[[ -f "$CR3_TOML" ]] || { echo "ERROR: CR3 TOML missing: $CR3_TOML" >&2; exit 2; }
[[ -f "$CR3_YAML" ]] || { echo "ERROR: per-run YAML missing: $CR3_YAML (regenerate via cr3/scripts/gen_cr3_configs.py?)" >&2; exit 2; }
[[ -d "$OMNI3_MEGATRON_CHECKPOINT" ]] || { echo "ERROR: pretrain ckpt dir missing: $OMNI3_MEGATRON_CHECKPOINT" >&2; exit 2; }
command -v python >/dev/null   || { echo "ERROR: python not on PATH (run inside omni3-sft container)" >&2; exit 2; }
command -v torchrun >/dev/null || { echo "ERROR: torchrun not on PATH (run inside omni3-sft container)" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 2. Convert TOML -> Energon. Three idempotency layers in the converter:
#    * fresh dataset dir -> normal write
#    * dataset.yaml exists AND first shard's stem is dot-free -> skip
#    * dataset.yaml exists AND first shard's stem has dots -> auto-regenerate
#      (a9e9f1e adds the detection; 642433b refuses to wipe non-dataset dirs)
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$CR3_ENERGON_PATH")"
echo "=== convert ==="
python /workspace/Nemotron/cr3/scripts/cr3_to_energon.py \
    --cr3-toml          "$CR3_TOML" \
    --output            "$CR3_ENERGON_PATH" \
    --val-fraction      "$CR3_VAL_FRACTION" \
    --samples-per-shard "$CR3_SAMPLES_PER_SHARD"

# Quick post-condition: first member should have a dot-free stem and the
# .tar.idx sidecars must exist (training opens them via TarIndexReader).
FIRST_TAR="$(find "$CR3_ENERGON_PATH" -maxdepth 1 -name 'train-shard-*.tar' -print -quit 2>/dev/null || true)"
if [[ -n "$FIRST_TAR" ]]; then
    FIRST_MEMBER="$(tar -tf "$FIRST_TAR" 2>/dev/null | head -1 || true)"
    echo "  first member : $FIRST_MEMBER"
    [[ -f "${FIRST_TAR}.idx" ]] || { echo "ERROR: missing $FIRST_TAR.idx — energon prepare didn't run; re-run after git pull (commit 0afefb0)" >&2; exit 3; }
fi
echo

# ---------------------------------------------------------------------------
# 3. Patch Megatron-Bridge fusions.py. The container is launched with --rm
#    so the writable overlay is fresh each session; re-apply on every run.
#    Same patch logic as cr3/test/scripts/run_train_smoke.sh.
# ---------------------------------------------------------------------------
FUSIONS_PY=/workspace/Megatron-Bridge/src/megatron/bridge/utils/fusions.py
FUSIONS_CACHE=/workspace/Megatron-Bridge/src/megatron/bridge/utils/__pycache__/fusions.cpython-312.pyc
echo "=== fusions.py patch ==="
python3 - <<PY
import re
src = open("$FUSIONS_PY").read()
ALREADY_PATCHED = "def can_enable_gradient_accumulation_fusion() -> bool:\n    return False\n"
if ALREADY_PATCHED in src:
    print("fusions.py already patched, skipping")
else:
    patched = re.sub(
        r"(def can_enable_gradient_accumulation_fusion\(\) -> bool:\n)([\s\S]*?)(?=\n\ndef |\Z)",
        r"\1    return False\n",
        src, count=1,
    )
    if patched == src:
        raise SystemExit("fusions.py patch failed: regex did not match")
    open("$FUSIONS_PY", "w").write(patched)
    print("Patched", "$FUSIONS_PY")
PY
rm -f "$FUSIONS_CACHE"
echo

# ---------------------------------------------------------------------------
# 4. Install pydantic-settings into the Megatron-Bridge .venv if missing
#    (no-op on repeat). The venv has uv but no pip baked in.
# ---------------------------------------------------------------------------
VENV_PYTHON=/workspace/Megatron-Bridge/.venv/bin/python
echo "=== deps ==="
"$VENV_PYTHON" -c "import pydantic_settings" 2>/dev/null || \
    uv pip install --python "$VENV_PYTHON" --quiet "pydantic-settings>=2.12.0"
"$VENV_PYTHON" -c "import pydantic_settings as p; print('pydantic_settings', p.__version__)"
echo

# ---------------------------------------------------------------------------
# 5. torchrun. Matches the assy17 sbatch wrapper invocation: per-run YAML,
#    8 GPUs on one node, no Hydra-style overrides (those live in the YAML
#    or are picked up via ${oc.env:...} from the env vars exported above).
# ---------------------------------------------------------------------------
export PYTHONPATH="/workspace/Nemotron/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CR3_CKPT_SAVE"

cd /workspace/Nemotron/src/nemotron/recipes/omni3/stage0_sft
echo "=== torchrun ==="
exec torchrun --nproc-per-node=8 train.py --config "$CR3_YAML"
