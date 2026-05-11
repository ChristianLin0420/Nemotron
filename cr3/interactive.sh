#!/bin/bash
# Interactive SLURM allocation (1 node × 8 GPU) inside the omni3-sft container.
# Mirrors cosmos-reason2/interactive.sh and sop-inference-bp/interactive.sh.
#
# Use to:
#   * Hand-test the CR3 -> Energon converter on real lustre data
#   * Smoke-test torchrun training on one node before the full sweep
#   * Inspect Megatron-Bridge / Energon Python APIs interactively
#
# Source the env helper first so $CR3_LUSTRE_HOME resolves:
#   source Nemotron/cr3/env-setup.sh
#   bash   Nemotron/cr3/interactive.sh
#
# Container: pyxis pulls the docker image directly (no .sqsh build step).
# First run on a node caches the image under enroot's runtime root; later
# runs reuse the cache. Override the tag with $OMNI3_SFT_IMAGE if needed:
#   export OMNI3_SFT_IMAGE=nemotron/omni3-sft:public          # local build
#   export OMNI3_SFT_IMAGE=christianlin0420/omni3-sft:public  # Docker Hub (default)

ACCOUNT="${CR3_SLURM_ACCOUNT:-edgeai_tao-ptm_image-foundation-model-clip}"
PARTITION="${CR3_INTERACTIVE_PARTITION:-interactive_singlenode}"
JOB_NAME="${CR3_INTERACTIVE_JOBNAME:-cr3-nemotron-interactive}"
TIME="${CR3_INTERACTIVE_TIME:-04:00:00}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMOTRON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"     # .../Nemotron
CR_ROOT="$(cd "$NEMOTRON_ROOT/.." && pwd)"        # .../cr (parent of Nemotron, cosmos-reason2, sop-inference-bp)

# Container image: docker registry reference. pyxis resolves docker:// URIs
# transparently against Docker Hub and caches the unpacked rootfs per-node.
IMAGE="${OMNI3_SFT_IMAGE:-christianlin0420/omni3-sft:public}"
CONTAINER_IMAGE="${IMAGE}"

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
echo "  image     = $CONTAINER_IMAGE"
echo

srun \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --job-name="$JOB_NAME" \
    --gpus=8 \
    --ntasks-per-node=1 \
    --cpus-per-task=32 \
    --time="$TIME" \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$MOUNTS" \
    --container-writable \
    --pty bash -c "cd /workspace/Nemotron/cr3 && exec /bin/bash"
