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
(`ds_1run.json`, 182 records). On A40-46 the smoke validates the
pipeline up through 8-rank model load + DDP init; the actual training
step **does not fit on A40** for the 30B-A3B full-SFT recipe (model +
gradients + Adam optimizer states need ~65 GiB per rank, A40 has 44
GiB). Use this section to verify converter / image / mount plumbing on
A40, then use §3c to actually train on the A100-80 cluster.

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

### 3c. tp00302 SLURM interactive smoke (A100-80, 1 node × 8 GPU)

Runs the **same** `run_tp00302_smoke.sh` inside an interactive SLURM
allocation, this time against a real A100-80 node where 30B-A3B full
SFT actually fits. Use this to validate the full training step (10
iters with loss curve + iter_0000010 checkpoint) before submitting the
sweep in §4.

**1. Host (login node, one time per shell):**

```bash
# Source the per-user lustre env (uv paths, CR3_NEMOTRON_CACHE,
# CR3_ENERGON_ROOT, CR3_CKPT_ROOT, OMNI3_MEGATRON_CHECKPOINT)
source <Nemotron>/cr3/env-setup.sh

# Prerequisite (one-time): import the HF GA checkpoint to Megatron format.
# interactive.sh pulls the docker image directly via pyxis docker:// (no
# .sqsh build step required).
#   uv run nemotron omni3 model import pretrain --run edgeai-cluster \
#       --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
#       --megatron-path "$OMNI3_MEGATRON_CHECKPOINT"
```

**2. Host: allocate the interactive container (~30s to land in shell;
~5-10 min on first pyxis pull of the docker image):**

```bash
bash <Nemotron>/cr3/interactive.sh
# Lands you at /workspace/Nemotron/cr3 inside christianlin0420/omni3-sft:public, with
#   $HOME              -> /root
#   /lustre            -> /lustre (datasets, checkpoints)
#   <Nemotron>/        -> /workspace/Nemotron
#   <cosmos-reason2>/  -> /workspace/cosmos-reason2  (if present)
#
# To override the image (e.g., a locally-built tag):
#   export OMNI3_SFT_IMAGE=nemotron/omni3-sft:public
#   bash <Nemotron>/cr3/interactive.sh
```

**3. Container: configure the smoke env vars:**

```bash
# Tell the runner that lustre is mounted (no /datasets remap needed).
export CR3_DATASET_ROOT_OVERRIDE=""

# Output dirs on lustre (so checkpoints survive past the allocation):
export CR3_ENERGON_PATH="$CR3_ENERGON_ROOT/tp00302/tp00302_smoke"
export CR3_CKPT_SAVE="$CR3_CKPT_ROOT/tp00302/tp00302_smoke"

# Megatron checkpoint imported in step 1 — env-setup.sh already set this:
echo "ckpt_in : $OMNI3_MEGATRON_CHECKPOINT"

# A100-80 can afford the larger smoke seq length (the A40-default of 1024
# leaves throughput on the table; 4096 matches the eventual sweep).
export CR3_SEQ_LENGTH=4096
export CR3_GLOBAL_BATCH_SIZE=8

# Optional: keep the rest of the defaults (10 iters, lr 1.5e-5).
export CR3_TRAIN_ITERS=10
```

**4. Container: run the smoke (~5-10 min on A100-80):**

```bash
bash /workspace/Nemotron/cr3/test/scripts/run_tp00302_smoke.sh
```

The runner converts the TOML → Energon at `$CR3_ENERGON_PATH` (skipped if
`.nv-meta/dataset.yaml` already exists), then delegates to
`run_train_smoke.sh` for 10 torchrun iterations.

**5. Container: verify success:**

```bash
ls "$CR3_ENERGON_PATH/.nv-meta/dataset.yaml"        # convert OK
ls "$CR3_CKPT_SAVE/iter_0000010/"                    # train OK
```

**Iteration ideas (still in the interactive container):**

```bash
# Different split — point CR3_TOML at any TOML you've authored.
export CR3_TOML=/workspace/Nemotron/cr3/test/tp00302_smoke.toml
export CR3_ENERGON_PATH="$CR3_ENERGON_ROOT/tp00302/tp00302_smoke_v2"

# Longer run — bump iters + match cr3_base.yaml's seq_length to test
# the production setting.
export CR3_TRAIN_ITERS=200
export CR3_SEQ_LENGTH=8192     # cr3_base.yaml's default

bash /workspace/Nemotron/cr3/test/scripts/run_tp00302_smoke.sh
```

**Why A40 doesn't suffice (for reference):** with TP=2 / EP=4 / PP=1
the dense LM is held on each TP rank (~3-5 GiB BF16) and the MoE
experts are split across EP=4 ranks (~7 GiB each). On top of that,
Adam optimizer states (FP32 master + m + v = 6 bytes/param) for the
local-DP-rank's parameter shard land at ~18 GiB per rank even with
`use_distributed_optimizer=True`. Plus BF16 grads, FP32 main-param
mirror created during DDP init, etc. — the resident-then-needs-grow
buffer overflows the A40's 44 GiB ceiling. A100-80 has the headroom.
Switching parallelism to TP=8 / EP=1 might fit on A40 but breaks the
checkpoint's TP=2/EP=4 shard layout, so the cluster path is the
straightforward fix.

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
