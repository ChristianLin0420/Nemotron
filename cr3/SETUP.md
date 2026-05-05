# cr3 — first-time environment setup (reproducible per-user)

This is the one-shot setup that must run before anything in `Nemotron/cr3/`
works. After this, sourcing `env-setup.sh` is the only thing you need each
new shell.

Everything is keyed off `$USER`, so the steps below copy-paste verbatim for
any teammate. The only assumption is that you have:

- A Linux account on this cluster with shell access
- Write permission under `/lustre/fsw/portfolios/edgeai/users/$USER/`
- The system Python at `/home/$USER/miniconda3/bin/python` (or any python ≥3.10 — substitute below)
- An NGC API key in `~/.config/enroot/.credentials` (used by the build
  to authenticate to `nvcr.io`; the file is already populated for cluster
  members in this org)

> **Why so much per-user wiring?** The host's `~` partition has a small
> disk quota that fills the moment uv, NeMo-Run, or pip start writing
> caches. Every user-writable path below is parked on lustre instead.

---

## 1. Define your lustre roots

```bash
# Use your own user name; everything else derives from this.
export CR3_LUSTRE_HOME=/lustre/fsw/portfolios/edgeai/users/$USER
export CR3_LUSTRE_CACHE=$CR3_LUSTRE_HOME/.cache

mkdir -p $CR3_LUSTRE_CACHE
```

If your account uses a different lustre tree, override `CR3_LUSTRE_HOME`
once here and the rest of `env-setup.sh` follows.

## 2. Install `uv` to lustre (avoid home-dir quota)

The Nemotron CLI is a `uv` project; we install `uv` into a lustre-rooted
venv so its on-disk caches don't touch `~/.local`.

```bash
/home/$USER/miniconda3/bin/python -m venv $CR3_LUSTRE_CACHE/uv-venv
$CR3_LUSTRE_CACHE/uv-venv/bin/pip install --cache-dir $CR3_LUSTRE_CACHE/pip-cache uv
$CR3_LUSTRE_CACHE/uv-venv/bin/uv --version
```

You should see something like `uv 0.11.x`.

## 3. Source the env helper for every new shell

```bash
source <path-to>/Nemotron/cr3/env-setup.sh
which uv          # /lustre/fsw/.../$USER/.cache/uv-venv/bin/uv
echo $NEMORUN_HOME # /lustre/fsw/.../$USER/.cache/nemotron/nemo_run
```

`env-setup.sh` exports:

| var | purpose |
|---|---|
| `PATH` | prepends the lustre `uv-venv/bin` |
| `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `UV_TOOL_DIR`, `UV_PROJECT_ENVIRONMENT` | uv lustre cache layout |
| `NEMORUN_HOME` | NeMo-Run experiment manifests on lustre (was `~/.nemo_run`) |
| `CR3_NEMOTRON_CACHE` | `omni3-sft.sqsh` build output + checkpoints land here |
| `CR3_ENERGON_ROOT` | converted CR3 datasets (Energon WebDataset shards) |
| `CR3_CKPT_ROOT` | per-run Megatron training checkpoints |
| `OMNI3_SFT_SQSH` | container image path consumed by `interactive.sh` and the 28 training sbatch files |
| `OMNI3_MEGATRON_CHECKPOINT` | pretrain checkpoint path consumed by `cr3_base.yaml` |

## 4. Sync Nemotron's Python deps into the lustre venv

```bash
cd <path-to>/Nemotron
uv sync --extra data-sdg
```

`--extra data-sdg` is required because the CLI introspects every long-document
SDG recipe at startup; without it the `nemotron --help` line fails with
`ModuleNotFoundError: No module named 'data_designer'`.

Sanity check:
```bash
uv run nemotron --help | head
uv run nemotron omni3 build sft --run edgeai-cluster --dry-run | tail
```

The dry-run prints the resolved profile (account, partition, cache mount,
container image) — confirm everything points at your lustre paths.

## 5. Apply the on-cluster patches (one-time)

Two upstream nemo_runspec / nemotron files assume an SSH-tunnel submission
flow; for on-cluster (login-node) submission they fail with
`AttributeError: 'NoneType' object has no attribute '_set_job_dir'` because
no SSH tunnel is configured. The fix is a `LocalTunnel(job_dir="")` fallback
plus stripping a couple of host env vars that the build container can't use
(`SSL_CERT_FILE`, `XDG_RUNTIME_DIR`).

These patches are already applied in this checkout — files modified:

- `Nemotron/src/nemo_runspec/execution.py`
  fall through to `run.LocalTunnel(job_dir="")` when `tunnel != "ssh"`.
- `Nemotron/src/nemotron/cli/commands/omni3/build.py`
  same fallback + `unset SSL_CERT_FILE SSL_CERT_DIR XDG_RUNTIME_DIR` and
  `export REGISTRY_AUTH_FILE=/root/.config/containers/auth.json` in the
  inline build script (so curl finds the Fedora cert bundle and podman
  finds the auth.json mounted by `materialize_podman_auth_from_enroot`).
- `Nemotron/env.toml`
  cluster profile (`edgeai-cluster`) with the right account / partition /
  mounts / build_cache_dir, plus three derived profiles
  (`edgeai-omni3-sft`, `edgeai-data-prep`, `edgeai-model-import`).

Run `git diff` from `Nemotron/` to inspect; nothing else needs editing.

## 6. Build the SFT container (one-time)

```bash
source Nemotron/cr3/env-setup.sh
cd Nemotron
uv run nemotron omni3 build sft --run edgeai-cluster
```

This submits a CPU sbatch job that pulls `nvcr.io/nvidian/nemo:26.04.rc7`,
runs the Dockerfile, and converts the resulting image to a squashfs at
`$CR3_NEMOTRON_CACHE/containers/omni3-sft.sqsh` (≈ 26 GB, 30–60 min).

> **Known issue**: `nvcr.io/nvidian/nemo` is an NVIDIA-internal repo and the
> NGC API key in `~/.config/enroot/.credentials` returns 403 against it.
> If the build job's `log-*.out` ends with
> `creating build container: initializing source docker://nvcr.io/nvidian/nemo:26.04.rc7: Requesting bearer token: invalid status code from registry 403 (Forbidden)`,
> ask the team for credentials that have read access to `nvcr.io/nvidian/`,
> or ask a teammate for a pre-built `omni3-sft.sqsh` and point
> `OMNI3_SFT_SQSH` at that file in `env-setup.sh`.

## 7. Convert HF GA checkpoint → Megatron format (one-time)

```bash
uv run nemotron omni3 model import pretrain --run edgeai-cluster \
    --hf-model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \
    --megatron-path $OMNI3_MEGATRON_CHECKPOINT
```

Lands the model under `$CR3_NEMOTRON_CACHE/checkpoints/nemotron_omni/`.

## 8. Validate the slurm submission path (no GPUs needed)

After steps 1-5 you can already prove the sbatch flow works without paying
for GPU time:

```bash
# Generate the 28 per-config training sbatch files (idempotent).
python3 Nemotron/cr3/scripts/gen_cr3_configs.py \
    --cr3-tomls-dir <path-to>/cosmos-reason2/examples/cosmos_rl/configs/cr3-8b \
    --energon-root  $CR3_ENERGON_ROOT \
    --ckpt-root     $CR3_CKPT_ROOT

# Validate one — slurm parses the file and reports an estimated start time.
sbatch --test-only Nemotron/cr3/sbatch/assy17/sbatch_assy17_lr15e5_lr1e6_dmcq2p2n_ds1.sh
# Expect: "Job <id> to start at <date> using 248 processors on <node> in partition polar4"
```

If `sbatch --test-only` returns a job estimate, the submission path is
healthy: `--nodes=1 --gpus-per-node=8 --cpus-per-task=32` are all accepted
and the partition is reachable. Only after the container build (step 6)
completes will an *actual* `sbatch ...` run produce useful work.

## 9. Daily flow

```bash
# new shell — only step needed each session
source Nemotron/cr3/env-setup.sh

# interactive node for hand-debug:
bash Nemotron/cr3/interactive.sh

# submit one wave of training (7 jobs):
bash Nemotron/cr3/submit_all.sh --wave assy17

# submit all 28:
bash Nemotron/cr3/submit_all.sh --all

# eval after training:
bash Nemotron/cr3/eval/run_all_evals.sh
```

See `Nemotron/cr3/README.md` for the full data + training + eval pipeline.

## 10. Training inside an interactive container

Steps 1–8 leave you with a built `omni3-sft.sqsh`, a Megatron-format pretrain
checkpoint at `$OMNI3_MEGATRON_CHECKPOINT`, and the lustre paths exported by
`env-setup.sh`. This section is the hand-driven equivalent of the
`sbatch_assy17_*.sh` wrappers — useful when you want to inspect a training
run before committing it to a 4-hour batch window.

### 10.1 Allocate the node

```bash
source Nemotron/cr3/env-setup.sh    # if not already in this shell
bash   Nemotron/cr3/interactive.sh
```

`interactive.sh` srun's `--gpus=8` on `interactive_singlenode`, mounts
`$HOME:/root`, `/lustre:/lustre`, and the repo at `/workspace/Nemotron`,
then drops you at `/workspace/Nemotron/cr3` (`cr3/interactive.sh:44-68`).
All env vars from `env-setup.sh` are inherited.

### 10.2 Build the Energon dataset (once per CR3 TOML)

The trainer only ever loads `$CR3_ENERGON_PATH`; the CR3 TOML is converted
ahead of time. Inside the container:

```bash
RUN_NAME=assy17_lr15e5_lr1e6_dmcq2p2n_ds1
TOML=/workspace/cosmos-reason2/examples/cosmos_rl/configs/cr3-8b/assy17/${RUN_NAME}.toml
ENERGON=$CR3_ENERGON_ROOT/assy17/${RUN_NAME}

python /workspace/Nemotron/cr3/scripts/cr3_to_energon.py \
    --cr3-toml "$TOML" \
    --output   "$ENERGON" \
    --val-fraction 0.1
```

`cosmos-reason2` is auto-mounted by `interactive.sh` if its sibling clone
exists (`cr3/interactive.sh:46`). Skip this step if `$ENERGON/.nv-meta/`
already exists from a previous build — the converter wipes the output dir
on rebuild.

### 10.3 Export the per-run env vars

These are the four `${oc.env:...}` keys `cr3_base.yaml` consumes
(lines 32, 33, 38, 58, 62), plus the alloc / TLS hygiene that the production
sbatch sets (`sbatch/assy17/sbatch_assy17_lr15e5_lr1e6_dmcq2p2n_ds1.sh:61-67`):

```bash
export CR3_ENERGON_PATH="$ENERGON"
export CR3_CKPT_SAVE="$CR3_CKPT_ROOT/assy17/${RUN_NAME}"
export CR3_TRAIN_ITERS=4000           # production budget; 10 for a quick check
export CR3_LM_LR=1.5e-05               # match the per-run YAML
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset SSL_CERT_FILE SSL_CERT_DIR XDG_RUNTIME_DIR
```

`OMNI3_MEGATRON_CHECKPOINT` is already set by `env-setup.sh`; override it
inline if you're pointing at a different ckpt.

### 10.4 Launch torchrun

Same command the production sbatch uses (`sbatch_assy17_*.sh:71-72`) — no
Hydra overrides because the per-run YAML extends `cr3_base.yaml`:

```bash
cd /workspace/Nemotron/src/nemotron/recipes/omni3/stage0_sft
torchrun --nproc-per-node=8 train.py \
    --config config/cr3/assy17/${RUN_NAME}.yaml
```

Iterations land under `$CR3_CKPT_SAVE/iter_NNNNNNN/`; default
`checkpoint.save_interval=200` from `cr3_base.yaml`. For an interactive
session you might Ctrl-C, append `checkpoint.save_interval=50` to the
torchrun line so you don't lose progress.

### 10.5 Quick smoke before the real run

To validate the wiring without burning the 4-hour window, use the helper that
drops `seq_length` / `batch_size` / `recompute_num_layers`
(`cr3/test/scripts/run_train_smoke.sh:77-85`):

```bash
CR3_TRAIN_ITERS=10 bash /workspace/Nemotron/cr3/test/scripts/run_train_smoke.sh
```

It re-applies the `fusions.py` apex-disable patch on every container boot
(needed because `--container-writable` doesn't persist between allocations)
and `uv pip install`s `pydantic-settings` into the image's `.venv` (which
ships python but no pip — `uv` at `/root/.local/bin/uv` covers it). Both
are no-ops when already present.

---

## Quick troubleshooting

| symptom | fix |
|---|---|
| `uv: command not found` | step 2/3 not done — install uv on lustre and source `env-setup.sh` |
| `ModuleNotFoundError: No module named 'data_designer'` | step 4 — `uv sync --extra data-sdg` |
| `'NoneType' object has no attribute '_set_job_dir'` | step 5 patches not applied (`git status` should show clean — patches are committed) |
| `OSError: [Errno 122] Disk quota exceeded` writing to `~/.nemo_run/...` | `NEMORUN_HOME` not set — re-source `env-setup.sh` |
| `sbatch: error: Unable to open file ... _sbatch.sh` | LocalTunnel was constructed with a non-empty `job_dir`; verify the patch in `omni3/build.py` uses `LocalTunnel(job_dir="")` |
| `curl: (77) error setting certificate file: /usr/lib/ssl/...` | host's Debian SSL cert path bleeding into the Fedora build container — verify `unset SSL_CERT_FILE SSL_CERT_DIR` is in the build script |
| `403 (Forbidden)` pulling `nvcr.io/nvidian/...` | NGC key in `~/.config/enroot/.credentials` doesn't have access to the internal repo — ask the team or use a teammate's pre-built sqsh |
| `OverridesError: Could not override 'train.optimizer.use_distributed_optimizer'` from torchrun | the key is currently in `model:` of `cr3_base.yaml` and gets silently dropped; either move it to the `optimizer:` section in the YAML, or prefix the CLI override with `+` (Hydra's "add missing key" syntax) |
| `/workspace/Megatron-Bridge/.venv/bin/pip: No such file or directory` inside the container | the public image's venv ships python without pip; install via `uv pip install --python /workspace/Megatron-Bridge/.venv/bin/python <pkg>` instead — `uv` is at `/root/.local/bin/uv` |
