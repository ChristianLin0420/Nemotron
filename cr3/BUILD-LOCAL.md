# Build `omni3-sft` on a local machine and push to Docker Hub

> This doc is written so a Claude Code instance running on the user's local
> machine (laptop/workstation) can execute it end-to-end without prior
> context. The build cluster (NVIDIA OCI cluster) cannot run podman/docker,
> so we build the OCI image off-cluster, push to Docker Hub, then pull on
> the cluster as a `.sqsh` via enroot.
>
> **Three phases:**
>
> 1. (cluster) push repo state to a GitHub fork
> 2. (local machine) clone, build, push image
> 3. (cluster) pull the image and convert to `.sqsh`
>
> **What the local Claude Code instance needs:** docker (or podman),
> ~40 GB free disk, ~1 h wall-clock, network access to GitHub +
> docker.io + nvcr.io, and a Docker Hub login.

---

## Phase 1 — push the repo from the cluster (one-time, do this BEFORE running local Claude)

> **You** (the human) run this on the cluster login node. Skip if you've
> already pushed all the recent local changes to your GitHub fork.

```bash
cd /lustre/fsw/portfolios/edgeai/users/$USER/projects/cr/Nemotron

# Confirm your fork is configured as a git remote (e.g. "origin" or "fork").
git remote -v
# Expected: origin -> https://github.com/<your-gh-user>/Nemotron.git

# Stage + commit the new and modified files
# env.toml is intentionally gitignored — it carries per-user cluster paths
# (account, partition, build_cache_dir keyed off the submitter's USER) and
# every teammate writes their own. SETUP.md walks each user through it.
git add \
    cr3/ \
    src/nemo_runspec/execution.py \
    src/nemotron/cli/commands/omni3/build.py \
    src/nemotron/recipes/omni3/stage0_sft/Dockerfile.public \
    src/nemotron/recipes/omni3/stage0_sft/config/cr3_base.yaml \
    src/nemotron/recipes/omni3/stage0_sft/config/cr3/

git commit -m "cr3: public-base Dockerfile, sbatch wrappers, env-setup, patches"
git push origin HEAD
```

After push, capture the commit SHA — local Claude will check this out:

```bash
git rev-parse HEAD
# example output: 3a8f2c1...   ← share this SHA with local Claude
```

---

## Phase 2 — build + push from the local machine

Open a Claude Code session **on your local machine**. Tell it:

> "Read `BUILD-LOCAL.md` from <your-fork-url>/blob/main/cr3/BUILD-LOCAL.md
> and execute Phase 2. Use Docker Hub user `<DOCKER_HUB_USER>` and tag
> `omni3-sft:public`. Image platform must be `linux/amd64` (the cluster is
> x86_64 even if my laptop is arm64 Mac)."

The local Claude should perform the steps below. Each step lists the exact
command and the expected outcome.

### 2.1 — Capture inputs

Local Claude prompts the user (or accepts in the initial message) for:

| Variable | Description | Example |
|---|---|---|
| `DOCKER_HUB_USER` | Docker Hub username that owns the destination repo | `christianlin0420` |
| `IMAGE_TAG` | Tag for the image | `omni3-sft:public` (recommended default) |
| `GH_FORK_URL` | URL of the Nemotron fork to clone | `https://github.com/ChristianLin0420/Nemotron.git` |
| `GH_COMMIT_SHA` | (optional) Commit SHA from Phase 1 to pin | `3a8f2c1...` |

Set them as shell variables so the rest of the doc copy-pastes:

```bash
export DOCKER_HUB_USER=<your-docker-hub-username>
export IMAGE_TAG=omni3-sft:public
export FULL_IMAGE=docker.io/${DOCKER_HUB_USER}/${IMAGE_TAG}
export GH_FORK_URL=<https://github.com/your-gh-user/Nemotron.git>
export GH_COMMIT_SHA=    # leave empty to use HEAD of main
```

### 2.2 — Verify host prereqs

```bash
docker --version          # any 20.x or newer is fine
docker buildx version     # required for cross-platform builds
df -h $HOME               # need ≥ 40 GB free
```

If `docker buildx` is missing, install Docker Desktop ≥ 4.0 (macOS/Win)
or `apt install docker-buildx-plugin` (Linux).

If `df -h` shows < 40 GB free, prune Docker first: `docker system prune -af`.

### 2.3 — Authenticate to Docker Hub

```bash
docker login docker.io
# user: $DOCKER_HUB_USER
# pass: a Docker Hub access token (Settings → Security → New Access Token)
#       NOT your account password
```

The login is cached at `~/.docker/config.json`; subsequent `docker push`
commands will use it.

### 2.4 — Clone the fork into a temp workspace

```bash
WORKDIR=$(mktemp -d -t nemotron-build-XXXX)
cd "$WORKDIR"
git clone "$GH_FORK_URL" Nemotron
cd Nemotron
[[ -n "$GH_COMMIT_SHA" ]] && git checkout "$GH_COMMIT_SHA"
git log -1 --oneline      # confirm the commit is the one Phase 1 pushed
```

Verify the public Dockerfile is present:

```bash
test -f src/nemotron/recipes/omni3/stage0_sft/Dockerfile.public \
    && echo "Dockerfile.public OK" \
    || { echo "MISSING Dockerfile.public — Phase 1 wasn't pushed"; exit 1; }
```

### 2.5 — Pre-create a buildx builder pinned to linux/amd64

This is the critical step on Apple Silicon. Without it, the resulting
image will be arm64 and pyxis on the x86 cluster cannot run it.

```bash
docker buildx create --name omni3-builder --platform linux/amd64 --use \
    || docker buildx use omni3-builder
docker buildx inspect --bootstrap
```

### 2.6 — Build + push in one shot

Building and pushing in a single `docker buildx build --push` is faster
than `build` + `push` because buildx streams the layers to Docker Hub as
they finalise.

```bash
cd "$WORKDIR/Nemotron"
docker buildx build \
    --platform linux/amd64 \
    --file src/nemotron/recipes/omni3/stage0_sft/Dockerfile.public \
    --tag "$FULL_IMAGE" \
    --push \
    --progress plain \
    src/nemotron/recipes/omni3/stage0_sft \
    2>&1 | tee "$WORKDIR/build.log"
```

Wall-clock: 30-60 min on a fast laptop, longer on metered networks.
Watch the progress output — the slow steps are:

| Step (in the log) | Why it's slow |
|---|---|
| `[base 2/3] RUN apt-get install ...` | apt mirrors |
| `[torch 1/2] RUN uv pip install torch==2.9.1` | ~3 GB torch wheel |
| `[sources 1/2] RUN git clone --recurse-submodules ... Megatron-Bridge` | submodule fetch |
| `[deps 1/1] RUN uv sync --locked --inexact` | resolves + installs ~2 GB of deps |
| `pushing manifest for ...` | upload of all layers to Docker Hub |

### 2.7 — Verify the push worked

```bash
docker buildx imagetools inspect "$FULL_IMAGE"
```

Expected output: a JSON block confirming the image exists with platform
`linux/amd64`. If you instead see `manifest unknown`, the push didn't
complete — check `$WORKDIR/build.log` for the upload error.

Tell the user the image is published:

```
docker.io/<DOCKER_HUB_USER>/omni3-sft:public
```

### 2.8 — Cleanup (optional)

```bash
docker buildx rm omni3-builder
rm -rf "$WORKDIR"
docker system prune -af   # frees ~20-30 GB cached layers
```

---

## Phase 3 — pull the image back to the cluster (one-time)

> Run on the **cluster login node**, AFTER Phase 2 finishes.

```bash
cd /lustre/fsw/portfolios/edgeai/users/$USER/projects/cr
source Nemotron/cr3/env-setup.sh

# Pull + convert to squashfs in one step. enroot reads the docker:// URI
# directly from Docker Hub.
mkdir -p "$CR3_NEMOTRON_CACHE/containers"
enroot import -o "$CR3_NEMOTRON_CACHE/containers/omni3-sft.sqsh" \
    "docker://docker.io/<DOCKER_HUB_USER>/omni3-sft:public"

# Verify the .sqsh exists and is a sane size (~25-30 GB)
ls -lh "$CR3_NEMOTRON_CACHE/containers/omni3-sft.sqsh"
```

If Docker Hub now requires auth for your repo (i.e. the repo is private),
add credentials to enroot first:

```bash
# Append to ~/.config/enroot/.credentials (one entry per registry)
echo "machine docker.io login <DOCKER_HUB_USER> password <DOCKER_TOKEN>" \
    >> ~/.config/enroot/.credentials
chmod 600 ~/.config/enroot/.credentials
```

The Docker token is **not** your password — generate one at
<https://app.docker.com/settings/personal-access-tokens>.

After the import lands the `.sqsh`, re-source `env-setup.sh` (so
`OMNI3_SFT_SQSH` re-resolves to the new file) and confirm `interactive.sh`
now picks up the real container:

```bash
unset OMNI3_SFT_SQSH                          # drop any stand-in override
source Nemotron/cr3/env-setup.sh
ls -lh "$OMNI3_SFT_SQSH"                      # should print your new file
bash Nemotron/cr3/interactive.sh
```

Inside the container, the import sanity check should match what the
Dockerfile's last layer ran:

```bash
python -c "import megatron.core, megatron.bridge"
python -c "from megatron.bridge.recipes.nemotron_omni import nemotron_omni_valor32k_sft_config"
echo "$? — omni3-sft container is functional"
```

---

## Triage if Phase 2 fails

The Dockerfile has two known fragile spots; both fail loud at build time:

### Symptom: `error: locked file requirements out of sync` or `Failed to spawn: ... torch`

Cause: Megatron-Bridge's `uv.lock` pins a torch that conflicts with the
torch we pre-installed in stage 2.

Fix: drop the `--locked` constraint by editing `Dockerfile.public` line
~100, replacing the OR-fallback block with a single line:

```dockerfile
RUN VIRTUAL_ENV=/workspace/Megatron-Bridge/.venv uv sync \
        --link-mode copy \
        --inexact
```

…and rebuild. The `--inexact` flag keeps the pre-installed torch and
resolves the rest fresh.

### Symptom: `ModuleNotFoundError: No module named 'transformer_engine'` (or `apex`) at the verify step

Cause: the public CUDA base doesn't ship the proprietary NV libraries
that the upstream NeMo image bakes in. Megatron-Bridge's lockfile
expects them to be already present.

Fix: add a build step before the verify in `Dockerfile.public`:

```dockerfile
# After the deps stage, before the release stage:
RUN VIRTUAL_ENV=/workspace/Megatron-Bridge/.venv uv pip install \
        --extra-index-url https://pypi.nvidia.com \
        transformer-engine apex
```

Then rebuild from the `deps` stage:

```bash
docker buildx build --platform linux/amd64 \
    --file src/nemotron/recipes/omni3/stage0_sft/Dockerfile.public \
    --tag "$FULL_IMAGE" \
    --target release \
    --push \
    --cache-from type=registry,ref="$FULL_IMAGE-cache" \
    --cache-to type=registry,ref="$FULL_IMAGE-cache",mode=max \
    src/nemotron/recipes/omni3/stage0_sft
```

The `--cache-from/--cache-to` pair makes the rebuild reuse stage 1-3
from the previous push, so only the new layer rebuilds.

### Symptom: Docker Hub `denied: requested access to the resource is denied`

Cause: `docker login` token expired, or the repo was created as private
without setting collaborators.

Fix: regenerate token at <https://app.docker.com/settings/personal-access-tokens>;
verify the destination repo at <https://hub.docker.com/r/$DOCKER_HUB_USER/omni3-sft>
is set to "public" (or that the cluster pulling user has read access).

### Symptom: Phase 3 `enroot import` fails with `unauthorized`

Same as above — Docker Hub auth missing on the cluster side. Add the
docker.io credential line to `~/.config/enroot/.credentials` as shown
in Phase 3.

---

## What the resulting image actually contains

End-state under `/workspace/Megatron-Bridge` inside the running container:

| Path | Contents |
|---|---|
| `.venv/` | Python 3.12 with torch 2.9.1+cu129, Megatron-Bridge's full dep set |
| `src/megatron/bridge/` | Megatron-Bridge `nemotron_3_omni` branch source |
| `3rdparty/Megatron-LM/` | NVIDIA/Megatron-LM `nemotron_3_omni` branch |
| (env) `PATH` | `.venv/bin` first; `python`, `torchrun` resolve there |
| (env) `PYTHONPATH` | `.venv` + Megatron-Bridge src + Megatron-LM src |

This matches the upstream private-base image identically except for the
choice of system-level base. Anything that imports cleanly in this image
is wire-compatible with the upstream `omni3-sft.sqsh` for the purposes
of running the `nemotron omni3 sft` CLI.
