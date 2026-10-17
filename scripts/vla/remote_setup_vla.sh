#!/usr/bin/env bash
# Build the pi0.5 (LeRobot PyTorch port) + LIBERO environment on robot_dev6.
#
# Runs ON the remote. Idempotent: every step skips or no-ops when already done.
# Everything lives under
#   ~/codes/lerobot-src          LeRobot source (rsync of the pinned clone, commit 9459727e7)
#   ~/codes/LeaP/.venv-vla       Python 3.12 venv (LeRobot 0.6.1 requires >=3.12)
#   ~/codes/LeaP/data/vla        LIBERO config + raw LIBERO-Spatial demos
#   ~/codes/LeaP/logs/vla        setup.log, download logs, setup.done marker
# The HF checkpoint goes to the default Hugging Face cache (~/.cache/huggingface/hub)
# so that `--policy.path=lerobot/pi05_libero_finetuned_v044` resolves offline.
#
# Launch (on the remote):
#   setsid nohup bash ~/codes/LeaP/scripts/vla/remote_setup_vla.sh >/dev/null 2>&1 &
# Then watch ~/codes/LeaP/logs/vla/setup.log; ~/codes/LeaP/logs/vla/setup.done appears at the end.
set -euo pipefail

LEAP="${LEAP_ROOT:-$HOME/codes/LeaP}"
SRC="${LEROBOT_SRC:-$HOME/codes/lerobot-src}"
VENV="$LEAP/.venv-vla"
LOGDIR="$LEAP/logs/vla"
DATA="$LEAP/data/vla"
CKPT="lerobot/pi05_libero_finetuned_v044"
DATASET="yifengzhu-hf/LIBERO-datasets"
PY_VERSION="3.12"

mkdir -p "$LOGDIR" "$DATA/raw" "$DATA/libero_config"
rm -f "$LOGDIR/setup.done"
echo $$ > "$LOGDIR/setup.pid"
exec > >(tee -a "$LOGDIR/setup.log") 2>&1

step() { echo "--- [$(date -Is)] $*"; }
echo "=== [$(date -Is)] setup start on $(hostname) (pid $$)"

# ---------------------------------------------------------------- source check
step "check LeRobot source at $SRC"
if [ ! -f "$SRC/pyproject.toml" ]; then
    echo "ERROR: $SRC/pyproject.toml missing; rsync the LeRobot clone first." >&2
    exit 1
fi
grep -m1 '^version' "$SRC/pyproject.toml"

# ------------------------------------------- background downloads (HF cache / data)
# Both are idempotent (hf download skips files that are already complete). They run
# in an ephemeral uvx env so they overlap with the venv build below.
step "start background downloads"
(
    uvx --from "huggingface_hub>=1.6" hf download "$CKPT" \
        > "$LOGDIR/download-checkpoint.log" 2>&1
    echo "checkpoint download exit=$?" >> "$LOGDIR/download-checkpoint.log"
) &
DL_CKPT=$!
(
    uvx --from "huggingface_hub>=1.6" hf download "$DATASET" --repo-type dataset \
        --include "libero_spatial/*" --local-dir "$DATA/raw" \
        > "$LOGDIR/download-dataset.log" 2>&1
    echo "dataset download exit=$?" >> "$LOGDIR/download-dataset.log"
) &
DL_DATA=$!
echo "checkpoint download pid=$DL_CKPT, dataset download pid=$DL_DATA"

# ----------------------------------------------------------------------- venv
step "python $PY_VERSION via uv"
uv python install "$PY_VERSION"
if [ ! -x "$VENV/bin/python" ]; then
    uv venv --python "$PY_VERSION" "$VENV"
fi
PY="$VENV/bin/python"
"$PY" --version

step "torch 2.11.0 / torchvision 0.26.0 from the cu128 index (A6000 = sm_86)"
uv pip install --python "$PY" "torch==2.11.0" "torchvision==0.26.0" \
    --index-url https://download.pytorch.org/whl/cu128

step "lerobot[pi,libero] (editable) + embedder/audio extras under LeRobot's pins"
# transformers is pinned by LeRobot's `transformers-dep` extra to >=5.4,<5.6 (lock: 5.5.4).
# google/embeddinggemma-2 needs a newer transformers, so the embedder cannot run in
# this venv; sentence-transformers is still installed for everything else it provides.
CONS="$LOGDIR/constraints-vla.txt"
cat > "$CONS" <<EOF
torch==2.11.0
torchvision==0.26.0
transformers==5.5.4
hf-libero==0.1.4
mujoco==3.8.1
robosuite==1.4.0
numpy==2.2.6
EOF
uv pip install --python "$PY" -e "$SRC[pi,libero]" \
    "sentence-transformers>=6.1" piper-tts soundfile librosa h5py \
    -c "$CONS"

# ------------------------------------------------------------------- versions
step "installed versions"
"$PY" - <<'EOF'
import importlib.metadata as md
pkgs = ["torch", "torchvision", "transformers", "lerobot", "hf-libero", "mujoco", "robosuite",
        "robomimic", "bddl", "numpy", "sentence-transformers", "piper-tts", "huggingface-hub",
        "safetensors", "tokenizers", "h5py", "librosa", "soundfile", "gymnasium"]
for p in pkgs:
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p} MISSING")
import torch
print("torch.cuda.is_available", torch.cuda.is_available(), "cuda", torch.version.cuda,
      "devices", torch.cuda.device_count())
EOF

# ------------------------------------------------------- LIBERO config + EGL smoke
step "LIBERO config (LIBERO_CONFIG_PATH=$DATA/libero_config) and EGL render smoke test"
export LIBERO_CONFIG_PATH="$DATA/libero_config"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# Render on whichever GPU currently has the most free memory.
SMOKE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
export CUDA_VISIBLE_DEVICES="$SMOKE_GPU" MUJOCO_EGL_DEVICE_ID="$SMOKE_GPU"
echo "smoke test on GPU $SMOKE_GPU"
# hf-libero's `libero.libero` package prompts with input() on first import when its
# config.yaml is missing, which would raise EOFError under nohup. Write the default
# config (paths inside the installed package) ourselves before anything imports it.
"$PY" - <<'EOF'
import importlib.util, os, yaml
cfg_dir = os.environ["LIBERO_CONFIG_PATH"]
cfg_file = os.path.join(cfg_dir, "config.yaml")
if not os.path.exists(cfg_file):
    pkg = os.path.dirname(importlib.util.find_spec("libero").origin)  # top-level pkg, no prompt
    root = os.path.join(pkg, "libero")
    paths = {
        "benchmark_root": root,
        "bddl_files": os.path.join(root, "./bddl_files"),
        "init_states": os.path.join(root, "./init_files"),
        "datasets": os.path.join(root, "../datasets"),
        "assets": os.path.join(root, "./assets"),
    }
    os.makedirs(cfg_dir, exist_ok=True)
    with open(cfg_file, "w") as f:
        yaml.dump(paths, f)
    print("wrote", cfg_file)
print(open(cfg_file).read())
EOF
"$PY" - <<'EOF'
import os, time
from libero.libero import get_libero_path, benchmark
for k in ["bddl_files", "init_states"]:
    p = get_libero_path(k)
    print(k, p, os.path.exists(p))
print("suites", sorted(benchmark.get_benchmark_dict()))
from lerobot.envs.libero import LiberoEnv, _get_suite
suite = _get_suite("libero_spatial")
env = LiberoEnv(task_suite=suite, task_id=0, task_suite_name="libero_spatial",
                obs_type="pixels_agent_pos", observation_height=256, observation_width=256)
t0 = time.time()
obs, info = env.reset(seed=0)
print("task:", env.task_description)
print("reset %.1fs" % (time.time() - t0), {k: v.shape for k, v in obs["pixels"].items()},
      "eef_pos", obs["robot_state"]["eef"]["pos"])
env.close()
print("EGL render OK")
EOF

# ------------------------------------------------------------------ downloads
step "wait for downloads"
wait "$DL_CKPT" || true
wait "$DL_DATA" || true
tail -2 "$LOGDIR/download-checkpoint.log" "$LOGDIR/download-dataset.log"
"$PY" - <<EOF
from huggingface_hub import snapshot_download
p = snapshot_download("$CKPT")
print("checkpoint snapshot:", p)
import os
for f in sorted(os.listdir(p)):
    print("  ", f, os.path.getsize(os.path.join(p, f)))
EOF
ls -la "$DATA/raw/libero_spatial" | head -20
du -sh "$DATA/raw/libero_spatial"

echo "=== [$(date -Is)] setup done"
touch "$LOGDIR/setup.done"
