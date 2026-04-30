#!/bin/bash
# Interactive SLURM allocation (1 node × 8 GPU) inside the omni3-sft container.
# Mirrors cosmos-reason2/interactive.sh and sop-inference-bp/interactive.sh.
#
# Use to:
#   * Hand-test the CR3 -> Energon converter on real lustre data
#   * Smoke-test torchrun training on one node before the full sweep
#   * Inspect Megatron-Bridge / Energon Python APIs interactively
#
# Source the env helper first so $OMNI3_SFT_SQSH and $CR3_LUSTRE_HOME resolve:
#   source Nemotron/cr3/env-setup.sh
#   bash   Nemotron/cr3/interactive.sh
#
# Build the container once before this works:
#   cd Nemotron
#   uv run nemotron omni3 build sft --run edgeai-cluster
# (the build lands omni3-sft.sqsh under $CR3_NEMOTRON_CACHE/containers/)

ACCOUNT="${CR3_SLURM_ACCOUNT:-edgeai_tao-ptm_image-foundation-model-clip}"
PARTITION="${CR3_INTERACTIVE_PARTITION:-interactive_singlenode}"
JOB_NAME="${CR3_INTERACTIVE_JOBNAME:-cr3-nemotron-interactive}"
TIME="${CR3_INTERACTIVE_TIME:-04:00:00}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMOTRON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"     # .../Nemotron
CR_ROOT="$(cd "$NEMOTRON_ROOT/.." && pwd)"        # .../cr (parent of Nemotron, cosmos-reason2, sop-inference-bp)

# Container image: env-setup.sh sets OMNI3_SFT_SQSH; fall back to the
# canonical per-user lustre cache layout if env-setup wasn't sourced.
SQSH="${OMNI3_SFT_SQSH:-/lustre/fsw/portfolios/edgeai/users/$USER/.cache/nemotron/containers/omni3-sft.sqsh}"

if [[ ! -f "$SQSH" ]]; then
    echo "ERROR: container image not found at $SQSH" >&2
    echo "Build it first: cd $NEMOTRON_ROOT && source cr3/env-setup.sh && uv run nemotron omni3 build sft --run edgeai-cluster" >&2
    echo "(or override OMNI3_SFT_SQSH=/path/to/some.sqsh)" >&2
    exit 1
fi

# Mounts:
#   $HOME:/root          — for ~/.cache/huggingface, ~/.netrc, etc.
#   /lustre:/lustre      — datasets, checkpoints, scratch
#   Nemotron repo        — code under test (cr3/ + recipes/ + src/)
#   cosmos-reason2 repo  — source of truth for CR3 TOMLs and JSON datasets
MOUNTS="$HOME:/root,/lustre:/lustre"
MOUNTS="$MOUNTS,$NEMOTRON_ROOT:/workspace/Nemotron"
[[ -d "$CR_ROOT/cosmos-reason2"            ]] && MOUNTS="$MOUNTS,$CR_ROOT/cosmos-reason2:/workspace/cosmos-reason2"
[[ -d "$CR_ROOT/sop-inference-bp/sop-inference-bp" ]] && MOUNTS="$MOUNTS,$CR_ROOT/sop-inference-bp/sop-inference-bp:/workspace/sop-inference-bp"

echo "Allocating interactive node:"
echo "  account   = $ACCOUNT"
echo "  partition = $PARTITION"
echo "  gpus      = 8"
echo "  time      = $TIME"
echo "  image     = $SQSH"
echo

srun \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$JOB_NAME" \
    --gpus=8 \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --time="$TIME" \
    --container-image="$SQSH" \
    --container-mounts="$MOUNTS" \
    --container-writable \
    --pty bash -c "cd /workspace/Nemotron/cr3 && exec /bin/bash"
