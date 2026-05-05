#!/usr/bin/env python3
"""Run megatron-bridge convert_checkpoints.py import with apex fusion disabled.

The omni3-sft:public image ships transformer_engine but NOT apex's
fused_weight_gradient_mlp_cuda extension. megatron-bridge's
``can_enable_gradient_accumulation_fusion()`` returns True the moment
``transformer_engine.pytorch`` imports, so the GPT/Mamba providers default
``gradient_accumulation_fusion=True``. The non-TE ColumnParallelLinear path
(used by Mamba's output_layer in this Nemotron-Omni LLaVA stack) then errors
out asking for the missing apex CUDA ext.

Patching the helper to return False BEFORE the providers' dataclass fields are
materialised flips the default to False, which is what we want for a
weights-only import that doesn't need gradient fusion at all.

Usage (inside the omni3-sft container):
    python run_import_ckpt.py \
        /workspace/Nemotron/checkpoints/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
        /workspace/Nemotron/checkpoints/megatron/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print("usage: run_import_ckpt.py <hf-model-path> <megatron-output-path>",
          file=sys.stderr)
    sys.exit(2)

hf_path, megatron_path = sys.argv[1], sys.argv[2]

# Patch BEFORE importing any provider — dataclass fields capture the function
# object at class-definition time via default_factory.
import megatron.bridge.utils.fusions as _fusions
_fusions.can_enable_gradient_accumulation_fusion = lambda: False

CONVERT_SCRIPT = Path("/workspace/Megatron-Bridge/examples/conversion/convert_checkpoints.py")
if not CONVERT_SCRIPT.exists():
    raise SystemExit(f"convert_checkpoints.py not found at {CONVERT_SCRIPT}")

sys.argv = [
    str(CONVERT_SCRIPT),
    "import",
    "--hf-model", hf_path,
    "--megatron-path", megatron_path,
    "--torch-dtype", "bfloat16",
    "--trust-remote-code",
]
runpy.run_path(str(CONVERT_SCRIPT), run_name="__main__")
