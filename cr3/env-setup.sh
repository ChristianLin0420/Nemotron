#!/bin/bash
# Source this BEFORE invoking nemotron / uv on this cluster.
#   source Nemotron/cr3/env-setup.sh
#
# Lustre-rooted uv install + cache so nothing fills the home-dir quota.
# All paths use $USER so this works across teammates without edits.
#
# REQUIRED first-time setup (one-shot, see SETUP.md for full instructions):
#   /home/$USER/miniconda3/bin/python -m venv $CR3_LUSTRE_CACHE/uv-venv
#   $CR3_LUSTRE_CACHE/uv-venv/bin/pip install --cache-dir $CR3_LUSTRE_CACHE/pip-cache uv
#   cd Nemotron && uv sync --extra data-sdg

# ---------------------------------------------------------------------------
# Per-user lustre roots — override CR3_LUSTRE_HOME if your account uses a
# different lustre path than /lustre/fsw/portfolios/edgeai/users/$USER/.
# ---------------------------------------------------------------------------
: "${CR3_LUSTRE_HOME:=/lustre/fsw/portfolios/edgeai/users/$USER}"
: "${CR3_LUSTRE_CACHE:=$CR3_LUSTRE_HOME/.cache}"

# ---------------------------------------------------------------------------
# uv binary + caches (avoid ~/.local quota issues)
# ---------------------------------------------------------------------------
export PATH=$CR3_LUSTRE_CACHE/uv-venv/bin:$PATH
export UV_CACHE_DIR=$CR3_LUSTRE_CACHE/uv
export UV_PYTHON_INSTALL_DIR=$CR3_LUSTRE_CACHE/uv/python
export UV_TOOL_DIR=$CR3_LUSTRE_CACHE/uv/tool
export UV_PROJECT_ENVIRONMENT=$CR3_LUSTRE_CACHE/uv/nemotron-venv
export XDG_CACHE_HOME=$CR3_LUSTRE_CACHE

# ---------------------------------------------------------------------------
# NeMo-Run experiment manifests / job state. Defaults to ~/.nemo_run which
# fills the home-dir quota; redirect to lustre.
# ---------------------------------------------------------------------------
export NEMORUN_HOME=$CR3_LUSTRE_CACHE/nemotron/nemo_run
export NEMO_RUN_LOCAL_JOB_DIR=$NEMORUN_HOME
mkdir -p "$NEMORUN_HOME"

# ---------------------------------------------------------------------------
# Nemotron build cache & training checkpoints. The container build dispatcher
# reads CR3_NEMOTRON_CACHE and lands omni3-sft.sqsh under containers/.
# CR3_CKPT_ROOT is where the training sbatch wrappers write iter_*/.
# ---------------------------------------------------------------------------
# Use ``: "${VAR:=default}"`` so values you pre-export (e.g. to point at a
# teammate's pre-built sqsh) survive a re-source of this script.
: "${CR3_NEMOTRON_CACHE:=$CR3_LUSTRE_CACHE/nemotron}"
: "${CR3_ENERGON_ROOT:=$CR3_LUSTRE_HOME/datasets/cr3-nemotron/energon}"
: "${CR3_CKPT_ROOT:=$CR3_LUSTRE_HOME/cr3-nemotron/ckpt}"
: "${OMNI3_SFT_SQSH:=$CR3_NEMOTRON_CACHE/containers/omni3-sft.sqsh}"
: "${OMNI3_MEGATRON_CHECKPOINT:=$CR3_NEMOTRON_CACHE/checkpoints/nemotron_omni}"
export CR3_NEMOTRON_CACHE CR3_ENERGON_ROOT CR3_CKPT_ROOT OMNI3_SFT_SQSH OMNI3_MEGATRON_CHECKPOINT
mkdir -p "$CR3_NEMOTRON_CACHE" "$CR3_ENERGON_ROOT" "$CR3_CKPT_ROOT"
