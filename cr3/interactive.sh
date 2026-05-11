#!/bin/bash
# Interactive SLURM allocation (1 node × 8 GPU) inside the omni3-sft container.
# Mirrors cosmos-reason2/interactive.sh and sop-inference-bp/interactive.sh.
#
# Use to:
#   * Hand-test the CR3 -> Energon converter on real lustre data
#   * Smoke-test torchrun training on one node before the full sweep
#   * Inspect Megatron-Bridge / Energon Python APIs interactively
#
# Usage:
#   bash Nemotron/cr3/interactive.sh
#
# This script auto-sources cr3/env-setup.sh before srun so that
# $CR3_ENERGON_ROOT, $CR3_CKPT_ROOT, $OMNI3_MEGATRON_CHECKPOINT, etc.
# propagate into the container via srun's --export=ALL default. No
# need to source env-setup.sh manually first (re-sourcing is safe
# because env-setup.sh uses ``: "${VAR:=default}"``).
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

# Pull CR3_* / OMNI3_* / NEMORUN_HOME / CR3_LUSTRE_HOME etc. into our
# environment so srun --export=ALL (default) sends them into the
# container. Idempotent if the user already sourced env-setup.sh.
if [[ -f "$SCRIPT_DIR/env-setup.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/env-setup.sh"
fi

# Container image: docker registry reference. pyxis resolves docker:// URIs
# transparently against Docker Hub and caches the unpacked rootfs per-node.
IMAGE="${OMNI3_SFT_IMAGE:-christianlin0420/omni3-sft:public}"
CONTAINER_IMAGE="${IMAGE}"

# Mounts:
#   /lustre:/lustre                                    — datasets, checkpoints, scratch
#   $HOME/.cache/huggingface:/root/.cache/huggingface  — HF model + token cache (mirrors docker-interactive.sh)
#   Nemotron repo                                      — code under test (cr3/ + recipes/ + src/)
#   cosmos-reason2 repo                                — source of truth for CR3 TOMLs and JSON datasets
#
# DO NOT mount $HOME:/root wholesale — the image stores uv's Python install
# under /root/.local/share/uv/python/... and uv itself at /root/.local/bin/uv.
# Mounting the user's $HOME onto /root shadows both, breaking the venv at
# /workspace/Megatron-Bridge/.venv/bin/python (its symlink target ENOENTs).
#
# Pyxis on this cluster ALSO auto-mounts $HOME → /root by default even when
# we don't include it in --container-mounts. We disable that below with
# --no-container-mount-home so the image's /root stays intact.
HF_CACHE_HOST="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE_HOST"

MOUNTS="/lustre:/lustre"
MOUNTS="$MOUNTS,$HF_CACHE_HOST:/root/.cache/huggingface"
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
    --no-container-mount-home \
    --container-writable \
    --pty bash -c "cd /workspace/Nemotron/cr3 && exec /bin/bash"
