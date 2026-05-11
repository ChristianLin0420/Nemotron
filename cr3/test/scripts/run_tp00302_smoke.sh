#!/bin/bash
# End-to-end CR3 tp00302 smoke runner. Run *inside* the omni3-sft container.
#
# Performs two steps:
#   1. Convert tp00302_smoke.toml -> Energon WebDataset shards
#      (idempotent: skipped if .nv-meta/dataset.yaml already exists).
#   2. Delegate to cr3/test/scripts/run_train_smoke.sh for a 10-iter
#      torchrun training pass (A40 OOM-safe overrides live there).
#
# Cross-environment behaviour is driven by CR3_DATASET_ROOT_OVERRIDE:
#   * Local docker: '/datasets'  (default; matches docker-interactive.sh mount)
#   * SLURM A100  : ''           (empty; lustre is mounted, paths resolve as-is)
#
# Env vars (defaults shown):
#   OMNI3_MEGATRON_CHECKPOINT  /workspace/Nemotron/checkpoints/megatron/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
#   CR3_DATASET_ROOT_OVERRIDE  /datasets
#   CR3_TOML                   /workspace/Nemotron/cr3/test/tp00302_smoke.toml
#   CR3_ENERGON_PATH           /workspace/Nemotron/cr3/test/energon/tp00302_smoke
#   CR3_CKPT_SAVE              /workspace/Nemotron/cr3/test/ckpt/tp00302_smoke
#   CR3_TRAIN_ITERS            10
#   CR3_LM_LR                  1.5e-5
#   CR3_VAL_FRACTION           0.1
#   CR3_SAMPLES_PER_SHARD      50

set -euo pipefail

OMNI3_MEGATRON_CHECKPOINT="${OMNI3_MEGATRON_CHECKPOINT:-/workspace/Nemotron/checkpoints/megatron/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
CR3_DATASET_ROOT_OVERRIDE="${CR3_DATASET_ROOT_OVERRIDE-/datasets}"   # unset/empty disables remap
CR3_TOML="${CR3_TOML:-/workspace/Nemotron/cr3/test/tp00302_smoke.toml}"
CR3_ENERGON_PATH="${CR3_ENERGON_PATH:-/workspace/Nemotron/cr3/test/energon/tp00302_smoke}"
CR3_CKPT_SAVE="${CR3_CKPT_SAVE:-/workspace/Nemotron/cr3/test/ckpt/tp00302_smoke}"
CR3_TRAIN_ITERS="${CR3_TRAIN_ITERS:-10}"
CR3_LM_LR="${CR3_LM_LR:-1.5e-5}"
CR3_VAL_FRACTION="${CR3_VAL_FRACTION:-0.1}"
CR3_SAMPLES_PER_SHARD="${CR3_SAMPLES_PER_SHARD:-50}"

# ---------------------------------------------------------------------------
# 0. Sanity checks (fail fast with a readable error)
# ---------------------------------------------------------------------------
[[ -f "$CR3_TOML" ]] || { echo "ERROR: CR3_TOML not found: $CR3_TOML" >&2; exit 2; }
[[ -d "$OMNI3_MEGATRON_CHECKPOINT" ]] || { echo "ERROR: OMNI3_MEGATRON_CHECKPOINT dir not found: $OMNI3_MEGATRON_CHECKPOINT" >&2; exit 2; }
command -v python >/dev/null 2>&1 || { echo "ERROR: python not on PATH (are you inside the omni3-sft container?)" >&2; exit 2; }
command -v torchrun >/dev/null 2>&1 || { echo "ERROR: torchrun not on PATH (are you inside the omni3-sft container?)" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 1. Convert TOML -> Energon (idempotent)
# ---------------------------------------------------------------------------
DATASET_YAML="$CR3_ENERGON_PATH/.nv-meta/dataset.yaml"
if [[ -f "$DATASET_YAML" ]]; then
    echo "[convert] skip: $DATASET_YAML already exists"
else
    echo "[convert] $CR3_TOML -> $CR3_ENERGON_PATH"
    mkdir -p "$(dirname "$CR3_ENERGON_PATH")"
    OVERRIDE_ARGS=()
    if [[ -n "$CR3_DATASET_ROOT_OVERRIDE" ]]; then
        OVERRIDE_ARGS=(--dataset-root-override "$CR3_DATASET_ROOT_OVERRIDE")
    fi
    python /workspace/Nemotron/cr3/scripts/cr3_to_energon.py \
        --cr3-toml          "$CR3_TOML" \
        --output            "$CR3_ENERGON_PATH" \
        "${OVERRIDE_ARGS[@]}" \
        --val-fraction      "$CR3_VAL_FRACTION" \
        --samples-per-shard "$CR3_SAMPLES_PER_SHARD"
    [[ -f "$DATASET_YAML" ]] || { echo "ERROR: convert step did not produce $DATASET_YAML" >&2; exit 3; }
fi

# ---------------------------------------------------------------------------
# 2. Delegate to run_train_smoke.sh
#    It will: re-apply fusions patch, set PYTHONPATH, ensure pydantic-settings
#    is installed, and torchrun with A40-safe overrides (seq_length=4096,
#    global_batch_size=8, recompute_num_layers=1).
# ---------------------------------------------------------------------------
export OMNI3_MEGATRON_CHECKPOINT CR3_ENERGON_PATH CR3_CKPT_SAVE
export CR3_TRAIN_ITERS CR3_LM_LR
echo "[train] -> run_train_smoke.sh (iters=$CR3_TRAIN_ITERS, lm_lr=$CR3_LM_LR)"
exec bash /workspace/Nemotron/cr3/test/scripts/run_train_smoke.sh
