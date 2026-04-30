#!/bin/bash
# Build omni3-sft.sqsh from the PUBLIC-base Dockerfile.public, completely
# bypassing nvcr.io/nvidian/ (no internal NGC creds required).
#
# This is the workaround for the 403 you hit on
# ``nvcr.io/nvidian/nemo:26.04.rc7``: we build from
# ``nvcr.io/nvidia/cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04`` (anonymous
# pull) and reproduce the layers needed by Megatron-Bridge nemotron_3_omni.
#
# Usage:
#   bash Nemotron/cr3/build_omni3_sft_public.sh                  # build + sqsh, no push
#   bash Nemotron/cr3/build_omni3_sft_public.sh --push <registry/repo:tag>
#
# Output:
#   $CR3_NEMOTRON_CACHE/containers/omni3-sft.sqsh    (~26 GB, enroot-compatible)
#
# Prereqs:
#   * Sourced env-setup.sh (so CR3_NEMOTRON_CACHE is set)
#   * podman (preferred) OR docker on $PATH
#   * enroot on $PATH (for the .sqsh conversion). On this cluster enroot
#     ships with the slurm pyxis plugin, so it's typically available on
#     login nodes — verify with ``which enroot``.
#
# Why podman, not docker? podman runs rootless and is what Nemotron's
# upstream build dispatcher uses; on this cluster docker isn't always
# available to non-admin users, podman usually is. The script auto-detects
# either.

set -euo pipefail

PUSH_TARGET=""
while (( $# > 0 )); do
    case "$1" in
        --push)   PUSH_TARGET="${2:?--push needs <registry/repo:tag>}"; shift 2 ;;
        --help|-h)
            sed -n '1,/^set -euo pipefail/p' "$0" | sed '$d'
            exit 0 ;;
        *)        echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve paths from env-setup.sh — fall back to canonical defaults if
# someone runs this without sourcing it.
# ---------------------------------------------------------------------------
: "${CR3_NEMOTRON_CACHE:=/lustre/fsw/portfolios/edgeai/users/$USER/.cache/nemotron}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMOTRON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGE_DIR="$NEMOTRON_ROOT/src/nemotron/recipes/omni3/stage0_sft"
DOCKERFILE="$STAGE_DIR/Dockerfile.public"

IMAGE_TAG="nemotron/omni3-sft:public"
SQSH_OUT_DIR="$CR3_NEMOTRON_CACHE/containers"
SQSH_OUT="$SQSH_OUT_DIR/omni3-sft.sqsh"

mkdir -p "$SQSH_OUT_DIR"

# ---------------------------------------------------------------------------
# Pick a build backend. Prefer podman; fall back to docker.
# ---------------------------------------------------------------------------
if command -v podman >/dev/null 2>&1; then
    BUILDER=podman
elif command -v docker >/dev/null 2>&1; then
    BUILDER=docker
else
    echo "ERROR: neither podman nor docker found on PATH" >&2
    echo "       on this cluster you typically need an interactive node to use podman:" >&2
    echo "         bash Nemotron/cr3/interactive.sh   (then re-run this script)" >&2
    exit 1
fi

if ! command -v enroot >/dev/null 2>&1; then
    echo "ERROR: enroot not found on PATH (needed to convert OCI image -> .sqsh)" >&2
    exit 1
fi

echo "=========================================="
echo " omni3-sft.sqsh build"
echo "=========================================="
echo "  builder      : $BUILDER"
echo "  Dockerfile   : $DOCKERFILE"
echo "  image tag    : $IMAGE_TAG"
echo "  sqsh output  : $SQSH_OUT"
[[ -n "$PUSH_TARGET" ]] && echo "  push target  : $PUSH_TARGET"
echo

# ---------------------------------------------------------------------------
# 1. Build the OCI image
# ---------------------------------------------------------------------------
echo "[1/3] $BUILDER build  (~30-60 min on first run)"
$BUILDER build \
    -t "$IMAGE_TAG" \
    -f "$DOCKERFILE" \
    "$STAGE_DIR"

# ---------------------------------------------------------------------------
# 2. Convert OCI -> squashfs via enroot (this is what pyxis mounts at
#    --container-image=...sqsh time). Output is anatomically identical to
#    what ``nemotron omni3 build sft`` would have produced from the
#    private-base Dockerfile.
# ---------------------------------------------------------------------------
echo
echo "[2/3] enroot import -> $SQSH_OUT"
rm -f "$SQSH_OUT"
case "$BUILDER" in
    podman) enroot import -o "$SQSH_OUT" "podman://$IMAGE_TAG" ;;
    docker) enroot import -o "$SQSH_OUT" "dockerd://$IMAGE_TAG" ;;
esac
ls -lh "$SQSH_OUT"

# ---------------------------------------------------------------------------
# 3. Optional: push the OCI image to a registry so teammates can pull it
#    (instead of each rebuilding ~26 GB locally).
# ---------------------------------------------------------------------------
if [[ -n "$PUSH_TARGET" ]]; then
    echo
    echo "[3/3] $BUILDER push -> $PUSH_TARGET"
    $BUILDER tag  "$IMAGE_TAG" "$PUSH_TARGET"
    $BUILDER push "$PUSH_TARGET"
    echo "  pushed: $PUSH_TARGET"
fi

echo
echo "=========================================="
echo "DONE"
echo "=========================================="
echo "Use it via:"
echo "  export OMNI3_SFT_SQSH=$SQSH_OUT"
echo "  bash Nemotron/cr3/interactive.sh"
echo
echo "Or update env-setup.sh to make this the default:"
echo "  edit Nemotron/cr3/env-setup.sh -> set OMNI3_SFT_SQSH=<path>"
