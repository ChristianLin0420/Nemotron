#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=polar4,polar3,polar,batch_block1,grizzly,batch_block2,batch_block3
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --job-name=cr3-nemotron-assy17-lr15e5-lr1e6-dmcq2p2n-ds2
#SBATCH --output=./logs/%x-%j/stdout.log
#SBATCH --error=./logs/%x-%j/stderr.log

# Auto-resume across 4-hour wall-clock windows. Each entry runs one
# 4-hour window of training; scontrol requeue picks up the next one.
# The number of entries here equals ceil(epochs / windows_per_epoch);
# adjust if you change train_iters / global_batch_size in the YAML.
COMMANDS=(
    "torchrun-omni3-sft"
    "torchrun-omni3-sft"
    "torchrun-omni3-sft"
    "torchrun-omni3-sft"
)

INDEX_FILE="${SLURM_LOG_DIR:-${HOME}/slurm-logs}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_command_index.txt"
mkdir -p "$(dirname $INDEX_FILE)"

if [[ ! -f $INDEX_FILE ]]; then
    CURRENT_INDEX=0
    echo "0" > "$INDEX_FILE"
else
    CURRENT_INDEX=$(cat "$INDEX_FILE")
fi

if [[ $CURRENT_INDEX -ge ${#COMMANDS[@]} ]]; then
    echo "All commands processed. Exiting."
    exit 0
fi

CURRENT_COMMAND=${COMMANDS[$CURRENT_INDEX]}
echo "[wave $((CURRENT_INDEX+1))/${#COMMANDS[@]}] $CURRENT_COMMAND"

# All paths come from env-setup.sh — sbatch inherits the submitter's env.
# If you submit without sourcing env-setup.sh first, override these on the
# sbatch command line via --export=ALL,VAR=val,...
SQSH="${OMNI3_SFT_SQSH:-/lustre/fsw/portfolios/edgeai/users/$USER/.cache/nemotron/containers/omni3-sft.sqsh}"
NEMOTRON_ROOT="/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/cr/Nemotron"
CR_ROOT="$(cd "$NEMOTRON_ROOT/.." && pwd)"

MOUNTS="$HOME:/root,/lustre:/lustre,$NEMOTRON_ROOT:/workspace/Nemotron"
[[ -d "$CR_ROOT/cosmos-reason2" ]] && MOUNTS="$MOUNTS,$CR_ROOT/cosmos-reason2:/workspace/cosmos-reason2"

timeout 3.95h srun \
    --container-image="$SQSH" \
    --container-mounts="$MOUNTS" \
    --container-writable \
    bash -c "
set -e
export OMNI3_MEGATRON_CHECKPOINT='${OMNI3_MEGATRON_CHECKPOINT}'
export CR3_ENERGON_PATH='/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/datasets/cr3-nemotron/energon/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds2'
export CR3_CKPT_SAVE='/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/cr3-nemotron/ckpt/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds2'
export CR3_TRAIN_ITERS='4000'
export CR3_LM_LR='1.5e-05'
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE SSL_CERT_DIR XDG_RUNTIME_DIR

cd /workspace/Nemotron/src/nemotron/recipes/omni3/stage0_sft
echo 'Starting omni3 SFT for run: assy17_lr15e5_lr1e6_dmcq2p2n_ds2'
torchrun --nproc-per-node=8 train.py \
    --config /workspace/Nemotron/src/nemotron/recipes/omni3/stage0_sft/config/cr3/assy17/assy17_lr15e5_lr1e6_dmcq2p2n_ds2.yaml
"

CURRENT_INDEX=$((CURRENT_INDEX + 1))
echo $CURRENT_INDEX > "$INDEX_FILE"

if [[ $CURRENT_INDEX -lt ${#COMMANDS[@]} ]]; then
    echo "Requeuing for next 4-hour window..."
    scontrol requeue $SLURM_JOB_ID
else
    echo "All windows finished."
fi
