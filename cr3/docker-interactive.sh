#!/bin/bash
# Local docker equivalent of cr3/interactive.sh (no SLURM/enroot).
#
# Launches christianlin0420/omni3-sft:public with:
#   - all GPUs visible (--gpus all, nvidia container runtime)
#   - the Nemotron repo mounted at /workspace/Nemotron
#   - the local datasets dir mounted at /datasets
#   - the HF checkpoint mounted at /workspace/Nemotron/checkpoints/...
#     (already lives under the repo, so it's covered by the repo mount)
#   - $HOME mounted at /root for HF cache and credentials
#
# Usage:
#   bash Nemotron/cr3/docker-interactive.sh           # interactive shell
#   bash Nemotron/cr3/docker-interactive.sh -- <cmd>  # one-shot command
#
# Override the image with $OMNI3_SFT_IMAGE.

set -euo pipefail

IMAGE="${OMNI3_SFT_IMAGE:-christianlin0420/omni3-sft:public}"
CONTAINER_NAME="${OMNI3_SFT_CONTAINER:-cr3-nemotron-interactive}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMOTRON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASETS_ROOT="${CR3_DATASETS_ROOT:-/localhome/$USER/datasets}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

mkdir -p "$HF_CACHE"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image $IMAGE not found locally. Pull it first:" >&2
    echo "    docker pull $IMAGE" >&2
    exit 1
fi

# If a container with this name is already running, exec into it instead of
# starting a second one — helpful when the user re-invokes the script.
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Reusing running container '$CONTAINER_NAME' (docker exec)."
    exec docker exec -it "$CONTAINER_NAME" /bin/bash
fi
# Stale stopped container with same name? Remove it.
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# Use -it only when both stdin and stdout are TTYs. With `-- <cmd>` invocations
# from non-interactive contexts (CI, this orchestrator), keep them as plain pipes.
TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
    TTY_FLAGS=(-i -t)
fi

DOCKER_ARGS=(
    --rm
    "${TTY_FLAGS[@]}"
    --name "$CONTAINER_NAME"
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --shm-size=32g
    -v "$NEMOTRON_ROOT:/workspace/Nemotron"
    -v "$DATASETS_ROOT:/datasets"
    -v "$HF_CACHE:/root/.cache/huggingface"
    -e "HF_HOME=/root/.cache/huggingface"
    -e "PYTORCH_ALLOC_CONF=expandable_segments:True"
    -e "OMNI3_MEGATRON_CHECKPOINT=/workspace/Nemotron/checkpoints/megatron/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    -e "CR3_ENERGON_ROOT=/workspace/Nemotron/cr3/test/energon"
    -e "CR3_CKPT_ROOT=/workspace/Nemotron/cr3/test/ckpt"
    -w /workspace/Nemotron/cr3
)

echo "Launching $IMAGE"
echo "  repo     -> /workspace/Nemotron"
echo "  datasets -> /datasets"
echo "  hf_cache -> /root/.cache/huggingface"
echo

if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
    exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" /bin/bash -lc "$*"
fi
exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" /bin/bash
