#!/bin/bash
# Serve a CR3-finetuned Nemotron-Omni HF checkpoint via vLLM (1 node × 8 A100).
#
# Used by cr3-nemotron/eval/run_all_evals.sh: submits this sbatch with the
# HF checkpoint path + a port, then waits for the server to come up before
# kicking off sop-inference-bp eval.
#
# Usage:
#   sbatch cr3-nemotron/eval/sbatch_serve_vllm.sh <hf_ckpt_path> [<port>]
#
# Stdout will print "Application startup complete." once the OpenAI-compatible
# endpoint is ready at http://<head_node>:<port>/v1.

#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --job-name=cr3-nemotron-serve
#SBATCH --output=./logs/%x-%j/stdout.log
#SBATCH --error=./logs/%x-%j/stderr.log

set -euo pipefail

HF_CKPT="${1:?usage: sbatch sbatch_serve_vllm.sh <hf_ckpt> [<port>]}"
PORT="${2:-8000}"

if [[ ! -d "$HF_CKPT" ]]; then
    echo "ERROR: HF checkpoint not found: $HF_CKPT" >&2
    exit 1
fi

SQSH="${OMNI3_SFT_SQSH:-/home/$USER/.cache/nemotron/containers/omni3-sft.sqsh}"
SERVED_NAME="nemotron-cr3-$(basename "$(dirname "$HF_CKPT")")"

echo "Serving $HF_CKPT as model name $SERVED_NAME on port $PORT"
echo "Endpoint will be http://$(hostname):$PORT/v1"

srun \
    --container-image="$SQSH" \
    --container-mounts="$HOME:/root,/lustre:/lustre" \
    --container-writable \
    bash -c "
set -e
echo 'Launching vllm serve ...'
vllm serve '$HF_CKPT' \\
    --tensor-parallel-size 8 \\
    --gpu-memory-utilization 0.9 \\
    --trust-remote-code \\
    --port $PORT \\
    --host 0.0.0.0 \\
    --served-model-name '$SERVED_NAME' \\
    --max-model-len 16384
"
