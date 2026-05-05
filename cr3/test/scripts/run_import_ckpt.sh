#!/bin/bash
# Run megatron-bridge convert_checkpoints.py import with the
# gradient-accumulation-fusion helper hard-disabled.
#
# Why: omni3-sft:public ships transformer_engine but NOT apex's
# fused_weight_gradient_mlp_cuda. megatron-bridge's
# can_enable_gradient_accumulation_fusion() returns True the moment TE
# imports, but the Mamba output_layer goes through the non-TE
# ColumnParallelLinear path, which then errors out asking for the apex CUDA
# extension. For a weights-only HF -> Megatron import we don't want fusion
# at all.
#
# Strategy: rewrite the helper's source so it returns False unconditionally,
# wipe the bytecode cache, then run the converter. Container is --rm, so the
# patch only lasts for this invocation.
#
# Usage (inside the container):
#   bash run_import_ckpt.sh <hf-model-path> <megatron-output-path>

set -euo pipefail

HF_PATH="${1:?missing hf model path}"
MEGATRON_PATH="${2:?missing megatron output path}"

FUSIONS_PY=/workspace/Megatron-Bridge/src/megatron/bridge/utils/fusions.py
FUSIONS_CACHE=/workspace/Megatron-Bridge/src/megatron/bridge/utils/__pycache__/fusions.cpython-312.pyc

# Replace the function body with a single ``return False``. We match on the
# def line so we don't accidentally rewrite anything else.
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

cd /workspace/Megatron-Bridge
exec python examples/conversion/convert_checkpoints.py import \
    --hf-model "$HF_PATH" \
    --megatron-path "$MEGATRON_PATH" \
    --torch-dtype bfloat16 \
    --trust-remote-code
