#!/bin/bash
# Loop over every HF-exported CR3-Nemotron checkpoint:
#   1. sbatch a vLLM server for it
#   2. wait until the OpenAI endpoint responds
#   3. kick off sop-inference-bp eval pointing at the endpoint
#   4. scancel the server
#
# Run from the cr/ root (sibling of Nemotron, cosmos-reason2, cr3-nemotron,
# sop-inference-bp).
#
# Usage:
#   bash cr3-nemotron/eval/run_all_evals.sh [<ckpt_root>]
#
# ckpt_root defaults to /lustre/fsw/.../chrislin/cr3-nemotron/ckpt and is
# expected to contain <dataset>/<run_name>/hf/ subdirs (output of
# `nemotron omni3 model export pretrain`).

set -euo pipefail

CKPT_ROOT="${1:-/lustre/fsw/portfolios/edgeai/users/chrislin/cr3-nemotron/ckpt}"
EVAL_OUT_ROOT="${EVAL_OUT_ROOT:-output/cr3-nemotron-eval}"
PORT_BASE="${PORT_BASE:-8000}"

if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "ERROR: ckpt root not found: $CKPT_ROOT" >&2
    exit 1
fi

mapfile -t HF_CKPTS < <(find "$CKPT_ROOT" -maxdepth 3 -type d -name hf | sort)
if (( ${#HF_CKPTS[@]} == 0 )); then
    echo "No HF-exported checkpoints under $CKPT_ROOT (look for */*/hf/)" >&2
    exit 1
fi

echo "Will evaluate ${#HF_CKPTS[@]} checkpoints sequentially:"
printf '  %s\n' "${HF_CKPTS[@]}"
echo

wait_for_endpoint() {
    local url="$1" deadline=$(( $(date +%s) + 1200 ))   # 20 min
    while (( $(date +%s) < deadline )); do
        if curl -fsS -m 3 "$url/models" >/dev/null 2>&1; then
            echo "  endpoint ready: $url"
            return 0
        fi
        sleep 10
    done
    echo "  timed out waiting for $url" >&2
    return 1
}

for i in "${!HF_CKPTS[@]}"; do
    HF_CKPT="${HF_CKPTS[$i]}"
    RUN_NAME="$(basename "$(dirname "$HF_CKPT")")"
    DATASET="$(basename "$(dirname "$(dirname "$HF_CKPT")")")"
    PORT=$(( PORT_BASE + i ))
    EVAL_OUT="$EVAL_OUT_ROOT/$DATASET/$RUN_NAME"

    if [[ -f "$EVAL_OUT/accuracy.json" ]]; then
        echo "[skip] $RUN_NAME already evaluated (accuracy.json present)"
        continue
    fi

    echo "[$((i+1))/${#HF_CKPTS[@]}] eval $DATASET/$RUN_NAME"
    JID=$(sbatch --parsable cr3-nemotron/eval/sbatch_serve_vllm.sh "$HF_CKPT" "$PORT")
    echo "  serving job $JID on port $PORT"

    # Wait for the job to start, then poll the endpoint
    while [[ "$(squeue -j "$JID" -h -o %T 2>/dev/null)" != "RUNNING" ]]; do
        sleep 20
    done
    HEAD_NODE="$(squeue -j "$JID" -h -o %N | head -1)"
    if ! wait_for_endpoint "http://$HEAD_NODE:$PORT/v1"; then
        scancel "$JID" || true
        continue
    fi

    mkdir -p "$EVAL_OUT"
    bash sop-inference-bp/sop-inference-bp/scripts/run_evaluate_pipeline.sh \
        --backend openai \
        --vlm-server "http://$HEAD_NODE:$PORT/v1" \
        --output-dir "$EVAL_OUT" \
        --dataset "$DATASET" || true

    scancel "$JID" || true
    echo "  done -> $EVAL_OUT/accuracy.json"
done

echo
echo "All evals complete. Reports under $EVAL_OUT_ROOT/"
