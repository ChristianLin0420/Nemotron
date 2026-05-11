#!/bin/bash
# Simple SLURM interactive launcher for the omni3-sft image.
#
# Usage:  bash Nemotron/cr3/interactive.sh
#
# The omni3-sft image stores its python interpreter under
# /root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/.
# Mounting $HOME over /root would shadow it, so we mount only the HF
# cache and pass --no-container-mount-home so pyxis doesn't auto-mount
# $HOME → /root behind our back.
#
# Override the image with: export OMNI3_SFT_IMAGE=<other-tag>

ACCOUNT="edgeai_tao-ptm_image-foundation-model-clip"
PARTITION="interactive_singlenode"
IMAGE="${OMNI3_SFT_IMAGE:-christianlin0420/omni3-sft:public}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMOTRON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CR_ROOT="$(cd "$NEMOTRON_ROOT/.." && pwd)"

srun \
    --account=$ACCOUNT \
    --partition=$PARTITION \
    --job-name "cr3-nemotron-interactive" \
    --gpus 8 \
    --ntasks-per-node 1 \
    --time 04:00:00 \
    --container-image="$IMAGE" \
    --container-mounts=/lustre:/lustre,$NEMOTRON_ROOT:/workspace/Nemotron,$CR_ROOT/cosmos-reason2:/workspace/cosmos-reason2,$HOME/.cache:/root/.cache \
    --no-container-mount-home \
    --container-writable \
    --pty /bin/bash
