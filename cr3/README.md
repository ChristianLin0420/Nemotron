# Nemotron/cr3 — apply CR3 finetuning to Nemotron-3-Nano-Omni-30B-A3B-Reasoning

CR3-style SFT (the SOP finetuning recipe originally built on Cosmos-Reason1-8B
via cosmos-rl) ported onto **`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`**
using the official Nemotron Megatron-Bridge stack and the `omni3-sft.sqsh`
container produced by `nemotron omni3 build sft`.

> **First-time setup on a fresh user account?** Read **[`SETUP.md`](./SETUP.md)** —
> reproducible, parameterized by `$USER`, no editing required.

## Layout

```
Nemotron/                                      ← upstream Nemotron repo
├── env.toml                                   ← cluster profiles (edgeai-cluster, etc.)
├── cr3/                                       ← THIS DIRECTORY (CR3 work)
│   ├── README.md                              ← (this file)
│   ├── SETUP.md                               ← per-user one-time install steps
│   ├── env-setup.sh                           ← `source` to set $USER-keyed paths
│   ├── interactive.sh                         ← srun an interactive node inside omni3-sft
│   ├── submit_all.sh                          ← submit the 28-job sweep, optionally per-wave
│   ├── scripts/
│   │   ├── cr3_to_energon.py                  ← CR3 LLaVA JSON → Energon ChatMLWebdataset
│   │   └── gen_cr3_configs.py                 ← templates 28 YAMLs + 28 sbatch from CR3 TOMLs
│   ├── sbatch/                                ← 28 generated training sbatch files
│   │   ├── assy17/sbatch_*.sh                 (7)
│   │   ├── c310/sbatch_*.sh                   (7)
│   │   ├── tp00302/sbatch_*.sh                (7)
│   │   └── tp00303/sbatch_*.sh                (7)
│   └── eval/
│       ├── sbatch_serve_vllm.sh               ← 1-node × 8-GPU vLLM server for one HF ckpt
│       └── run_all_evals.sh                   ← serve → eval → teardown loop, all 28 ckpts
└── src/nemotron/recipes/omni3/stage0_sft/config/
    ├── cr3_base.yaml                          ← shared SFT config (TP=2 EP=4, frozen ViT/audio)
    └── cr3/<dataset>/<run>.yaml               ← 28 generated overrides (epoch, lm_lr, train_iters)
```

The 28 overrides come from translating the CR3-8B sweep
(`cosmos-reason2/examples/cosmos_rl/configs/cr3-8b/{assy17,c310,tp00302,tp00303}/*.toml`,
4 datasets × 7 configs each) into Megatron-Bridge YAML form.

## End-to-end run sequence

These steps assume:

- Setup from `SETUP.md` is done (uv on lustre, `uv sync --extra data-sdg`,
  patches applied).
- You've sourced the env helper:
  ```bash
  source <Nemotron>/cr3/env-setup.sh
  ```
- Repos `Nemotron/`, `cosmos-reason2/`, `sop-inference-bp/` are siblings
  under your `cr/` checkout (or under whatever parent `interactive.sh`
  detects).

### 0. One-time setup (already done if `SETUP.md` was followed)

```bash
# Build the SFT container — produces $CR3_NEMOTRON_CACHE/containers/omni3-sft.sqsh
cd <Nemotron>
uv run nemotron omni3 build sft --run edgeai-cluster

# Convert HF GA → Megatron format (one-time, ~30 min)
uv run nemotron omni3 model import pretrain --run edgeai-cluster \
    --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
    --megatron-path $OMNI3_MEGATRON_CHECKPOINT
```

### 1. Convert CR3 datasets to Energon (interactive node)

```bash
# Get an interactive node; lands in /workspace/Nemotron/cr3 inside omni3-sft
bash <Nemotron>/cr3/interactive.sh
```

Inside the container:

```bash
cd /workspace/Nemotron/cr3

# (1a) Smoke-test on one CR3 split — ~170 records, ~1 min
python scripts/cr3_to_energon.py \
    --cr3-toml /workspace/cosmos-reason2/examples/cosmos_rl/configs/cr3-8b/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds1.toml \
    --output   $CR3_ENERGON_ROOT/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds1 \
    --val-fraction 0.1 \
    --samples-per-shard 100

# (1b) Confirm Energon can load it back
python - <<'EOF'
from megatron.energon import get_train_dataset, WorkerConfig
import os
ds = get_train_dataset(
    f"{os.environ['CR3_ENERGON_ROOT']}/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds1",
    batch_size=1, worker_config=WorkerConfig.default_worker_config(),
)
sample = next(iter(ds))
print("type:", type(sample).__name__)
print("conversation[0:200]:", str(sample.conversation)[:200])
print("video bytes:", len(sample.videos[0]))
EOF

# (1c) Batch-convert all 28 CR3 TOMLs (~30-60 min total)
python scripts/cr3_to_energon.py \
    --cr3-tomls-dir /workspace/cosmos-reason2/examples/cosmos_rl/configs/cr3-8b \
    --output-root   $CR3_ENERGON_ROOT \
    --val-fraction 0.1
```

### 2. Generate per-config YAMLs and sbatch files

Run from anywhere (uses `$Nemotron/cr3/env-setup.sh`-exported paths):

```bash
python3 <Nemotron>/cr3/scripts/gen_cr3_configs.py \
    --cr3-tomls-dir <cr-root>/cosmos-reason2/examples/cosmos_rl/configs/cr3-8b \
    --energon-root  $CR3_ENERGON_ROOT \
    --ckpt-root     $CR3_CKPT_ROOT
```

This regenerates **28 YAMLs** under
`<Nemotron>/src/nemotron/recipes/omni3/stage0_sft/config/cr3/<ds>/<run>.yaml`
and **28 sbatch files** under `<Nemotron>/cr3/sbatch/<ds>/sbatch_<run>.sh`.

Validate one without running:

```bash
sbatch --test-only <Nemotron>/cr3/sbatch/assy17/sbatch_assy17_lr15e5_lr1e6_dmcq2p2n_ds1.sh
# → "Job <id> to start at <date> using 248 processors on <node> in partition polar4"
```

### 3. Smoke-test one training run interactively (1-node × 8 GPU, 10 iters)

```bash
bash <Nemotron>/cr3/interactive.sh
```

Inside the container:

```bash
cd /workspace/Nemotron/src/nemotron/recipes/omni3/stage0_sft

# env-setup.sh already exported these — keep for clarity:
export OMNI3_MEGATRON_CHECKPOINT=...   # already set
export CR3_ENERGON_PATH=$CR3_ENERGON_ROOT/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds1
export CR3_CKPT_SAVE=/tmp/smoke_ckpt
export CR3_LM_LR=1.5e-5
export CR3_TRAIN_ITERS=10
export PYTORCH_ALLOC_CONF=expandable_segments:True

# TP=2 EP=4 = 8 GPUs (matches single-node A100 constraint)
torchrun --nproc-per-node=8 train.py \
    --config config/cr3/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds1.yaml \
    train.train_iters=10 \
    train.global_batch_size=8 \
    checkpoint.save_interval=10
```

OOM triage ladder if 30B doesn't fit:
1. `dataset.seq_length=4096 model.seq_length=4096`
2. `train.global_batch_size=4`
3. `model.optimizer.offload_to_cpu=true`
4. Switch to LoRA via `recipe.name=nemotron_omni_valor32k_peft_config` in `cr3_base.yaml`

### 3b. tp00302 local smoke (docker on local 8× A40)

End-to-end smoke that exercises the public docker image, the converter,
and `run_train_smoke.sh` against one small split of the tp00302 dataset
(`ds_1run.json`, 182 records). Produces a 10-iter loss curve and a
training checkpoint at `cr3/test/ckpt/tp00302_smoke/iter_0000010/`.

**Host prerequisites (one time):**

```bash
# 1a. Either pull the prebuilt image …
docker pull christianlin0420/omni3-sft:public

# 1b. … OR build locally (~30-60 min, produces nemotron/omni3-sft:public)
bash cr3/build_omni3_sft_public.sh
export OMNI3_SFT_IMAGE=nemotron/omni3-sft:public

# 2. Point docker-interactive.sh at the in-repo datasets dir.
#    The default ($CR3_DATASETS_ROOT=/localhome/$USER/datasets) does NOT
#    contain tp00302 — your tp00302 copy is under Nemotron/datasets/.
export CR3_DATASETS_ROOT=/localhome/$USER/Nemotron/datasets

# 3. Enter the container
bash cr3/docker-interactive.sh
```

**Container side (one time, ~30 min):** convert HF → Megatron format.

```bash
bash /workspace/Nemotron/cr3/test/scripts/run_import_ckpt.sh \
    /workspace/Nemotron/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
    /workspace/Nemotron/checkpoints/megatron/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
```

**Container side (smoke, ~10 min, idempotent):**

```bash
bash /workspace/Nemotron/cr3/test/scripts/run_tp00302_smoke.sh
```

The runner converts the TOML → Energon, then delegates to
`run_train_smoke.sh` for 10 iters with A40-safe overrides
(`seq_length=4096`, `global_batch_size=8`, `recompute_num_layers=1`).

**Verifying success:**

```bash
# inside the container
ls /workspace/Nemotron/cr3/test/energon/tp00302_smoke/.nv-meta/dataset.yaml   # convert OK
ls /workspace/Nemotron/cr3/test/ckpt/tp00302_smoke/iter_0000010/               # train OK
```

**SLURM A100 reuse:** the same `run_tp00302_smoke.sh` runs unchanged on
the cluster. Inside the sbatch's container, set
`CR3_DATASET_ROOT_OVERRIDE=""` (lustre is mounted, paths resolve as-is)
and point `CR3_ENERGON_PATH` and `CR3_CKPT_SAVE` at lustre dirs.

### 4. Submit the full sweep

```bash
# Wave 1 — ASSY17 only (7 jobs ≈ ~340 GPU-h). Gates the budget.
bash <Nemotron>/cr3/submit_all.sh --wave assy17

# Subsequent waves once accuracy looks reasonable:
bash <Nemotron>/cr3/submit_all.sh --wave c310
bash <Nemotron>/cr3/submit_all.sh --wave tp00302
bash <Nemotron>/cr3/submit_all.sh --wave tp00303     # 30 epochs, longest

# Or all 28 at once after smoke tests pass:
bash <Nemotron>/cr3/submit_all.sh --all
```

### 5. Export each finished checkpoint back to HF format

```bash
for CKPT in $CR3_CKPT_ROOT/*/*/iter_*; do
    OUT=$(dirname "$CKPT")/hf
    [[ -d "$OUT" ]] && continue
    uv run nemotron omni3 model export pretrain --run edgeai-cluster \
        --megatron-path "$CKPT" --hf-path "$OUT"
done
```

### 6. Eval — serve via vLLM, run sop-inference-bp

```bash
HF_CKPT=$CR3_CKPT_ROOT/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds1/hf
sbatch <Nemotron>/cr3/eval/sbatch_serve_vllm.sh "$HF_CKPT" 8000

# Wait for "Application startup complete." in stdout, then:
HEAD=$(squeue -u $USER -h -o '%N' --name=cr3-nemotron-serve | head -1)
bash <cr-root>/sop-inference-bp/sop-inference-bp/scripts/run_evaluate_pipeline.sh \
    --backend openai \
    --vlm-server http://$HEAD:8000/v1 \
    --output-dir output/cr3-nemotron-eval/assy17_lr15e5_lr1e6_dmcq2p2n_ds1
scancel --name=cr3-nemotron-serve

# Or run the orchestrator over all 28 ckpts sequentially:
bash <Nemotron>/cr3/eval/run_all_evals.sh
```

## Energon dataset shape (what `cr3_to_energon.py` emits)

For each CR3 TOML, the converter writes one Energon dataset directory:

```
$CR3_ENERGON_ROOT/<dataset>/<run-name>/
├── train-shard-000000.tar
├── train-shard-000001.tar
├── ...
├── val-shard-000000.tar
└── .nv-meta/
    ├── dataset.yaml      # ChatMLWebdataset schema
    ├── index.sqlite      # byte-offset index for random access
    ├── .info.yaml
    ├── split.yaml
    └── index.uuid
```

Each tar entry is a single training sample with three fields:

| File | Contents |
|---|---|
| `__key__` | unique sample id (string, prefixed with split label) |
| `conversation.json` | OpenAI-style ChatML messages (see below) |
| `video.mp4` | raw mp4 bytes from the source CR3 video file |
| `audio.wav` | 16 kHz mono PCM (real audio if present, silence otherwise) |

The conversation JSON shape (consumed by
`megatron.bridge.data.energon.task_encoder_utils.ChatMLWebdataset`):

```json
[
  {"role": "user", "content": [
    {"type": "video"},
    {"type": "text", "text": "There are 8 possible steps for the SOP of the given video. What step is the operator doing?\n(1) picking up the cable\n..."}
  ]},
  {"role": "assistant", "content": [
    {"type": "text", "text": "(8) doing none of the above (1) picking up the cable"}
  ]}
]
```

CR3's `<video>\n` placeholder in the user text is stripped by the converter
because the typed `{"type": "video"}` content part is what the Nemotron
processor uses to inject vision tokens — leaving the literal string would
either be re-tokenized as text or duplicate the sentinel.

## Eval-side patch (sop-inference-bp)

`sop-inference-bp/sop-inference-bp/sop_monitoring/vlm.py` has been extended
with a `backend="openai"` path so `CosmosReason1` can talk to the vLLM
server via `chat.completions` instead of loading transformers in-process.
The companion `scripts/action_recognition_multi_gpu.py` accepts
`--backend openai --vlm_server http://...:8000/v1 [--served_model_name <name>]`,
and bypasses the multi-GPU pool when in OpenAI mode (vLLM handles
parallelism server-side).

## Known issues

- **Container build failing with `403` on `nvcr.io/nvidian/nemo`**: the GA
  build pulls `nvcr.io/nvidian/nemo:26.04.rc7`, which is an NVIDIA-internal
  repo. The default NGC API key in `~/.config/enroot/.credentials` does
  not have read access. Workarounds: (a) use a teammate's pre-built
  `omni3-sft.sqsh` and override `OMNI3_SFT_SQSH` in your shell, (b) ask
  the team for credentials with `nvcr.io/nvidian/` read access. The 28
  training sbatch files all pass `sbatch --test-only` independently of
  this issue — submission infrastructure is healthy.
- **`tunnel = None` AttributeError** in `omni3 build sft --run`: fixed
  by the `LocalTunnel(job_dir="")` patches in `nemo_runspec/execution.py`
  and `cli/commands/omni3/build.py`. See `SETUP.md` step 5.
