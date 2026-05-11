#!/bin/bash
# Launch a 10-iter CR3 SFT smoke test on 8x A40, using cr3_base.yaml directly
# with Hydra-style CLI overrides for seq_length / batch_size / iters.
#
# Run inside the omni3-sft container. Caller must have set:
#   $OMNI3_MEGATRON_CHECKPOINT   parent dir of iter_NNN/ (Megatron format)
#   $CR3_ENERGON_PATH            Energon dataset directory (contains .nv-meta/)
#   $CR3_CKPT_SAVE               directory to write iter_*/ training ckpts
#
# A40 (46 GB) is below the cr3_base.yaml's A100-80 target, so we drop:
#   * dataset.seq_length / model.seq_length 8192 -> $CR3_SEQ_LENGTH (default 4096)
#   * train.global_batch_size 32 -> $CR3_GLOBAL_BATCH_SIZE (default 8)
# and keep TP=2 EP=4 (= 8 GPUs).
#
# Override knobs:
#   CR3_SEQ_LENGTH         dataset.seq_length AND model.seq_length        (default 4096)
#   CR3_GLOBAL_BATCH_SIZE  train.global_batch_size                         (default 8)
# A40 needs CR3_SEQ_LENGTH=1024 to keep activations under the 46 GB budget;
# A100-80 can run at CR3_SEQ_LENGTH=4096 or above. Set these in the caller.
#
# Re-applies the gradient-accum-fusion patch on every container boot because
# the container is launched with --rm (image is pristine each time).

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Re-patch fusions.py (apex extension is missing from the image; helper
#    falsely reports True because TE imports). Same patch as run_import_ckpt.sh.
# ---------------------------------------------------------------------------
FUSIONS_PY=/workspace/Megatron-Bridge/src/megatron/bridge/utils/fusions.py
FUSIONS_CACHE=/workspace/Megatron-Bridge/src/megatron/bridge/utils/__pycache__/fusions.cpython-312.pyc

python3 - <<PY
import re
src = open("$FUSIONS_PY").read()
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

# ---------------------------------------------------------------------------
# 2. Make the Nemotron repo's nemotron + nemo_runspec packages importable.
#    The image bakes Megatron-Bridge but NOT the Nemotron repo (we mount it).
# ---------------------------------------------------------------------------
export PYTHONPATH=/workspace/Nemotron/src${PYTHONPATH:+:$PYTHONPATH}

# ---------------------------------------------------------------------------
# 2a. Install Nemotron deps that aren't baked into the public image. Currently
#     just pydantic-settings (used by nemo_runspec.config.pydantic_loader).
#     The Megatron-Bridge .venv ships without pip, so use uv (already on
#     PATH at /root/.local/bin/uv) targeting the venv's interpreter. Quiet +
#     idempotent across container restarts (--rm wipes installs each boot).
# ---------------------------------------------------------------------------
VENV_PYTHON=/workspace/Megatron-Bridge/.venv/bin/python
"$VENV_PYTHON" -c "import pydantic_settings" 2>/dev/null || \
    uv pip install --python "$VENV_PYTHON" --quiet "pydantic-settings>=2.12.0"

# Cluster-y env vars the train script reads via ${oc.env:...}
: "${OMNI3_MEGATRON_CHECKPOINT:?must be set}"
: "${CR3_ENERGON_PATH:?must be set}"
: "${CR3_CKPT_SAVE:?must be set}"
export CR3_TRAIN_ITERS="${CR3_TRAIN_ITERS:-10}"
export CR3_LM_LR="${CR3_LM_LR:-1.5e-5}"
export CR3_SEQ_LENGTH="${CR3_SEQ_LENGTH:-4096}"
export CR3_GLOBAL_BATCH_SIZE="${CR3_GLOBAL_BATCH_SIZE:-8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$CR3_CKPT_SAVE"

cd /workspace/Nemotron/src/nemotron/recipes/omni3/stage0_sft

echo "=== launch ==="
echo "  ckpt_in  : $OMNI3_MEGATRON_CHECKPOINT"
echo "  energon  : $CR3_ENERGON_PATH"
echo "  ckpt_out : $CR3_CKPT_SAVE"
echo "  iters    : $CR3_TRAIN_ITERS"
echo

exec torchrun --nproc-per-node=8 train.py \
    --config config/cr3_base.yaml \
    train.train_iters="$CR3_TRAIN_ITERS" \
    train.global_batch_size="$CR3_GLOBAL_BATCH_SIZE" \
    dataset.seq_length="$CR3_SEQ_LENGTH" \
    model.seq_length="$CR3_SEQ_LENGTH" \
    model.recompute_num_layers=1 \
    checkpoint.save_interval=10
