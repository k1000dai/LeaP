#!/usr/bin/env bash
# VLA prior-steering campaign on LIBERO-Spatial (docs/superpowers/specs/2026-10-17-vla-prior-steering-design.md).
# Stages: setup -> data -> prior -> eval -> results, sequential on one GPU. Idempotent: a stage whose outputs
# exist is skipped (dataset npz files, prior.pt, summary.json with every episode; the evaluator itself resumes
# per episode). Mirrors scripts/run_campaign.sh / run_steer_campaign.sh.
# Usage: scripts/vla/run_vla_campaign.sh
# Env:   ROOT (repo root; default: the repo holding this script), VENV (.venv-vla; sourced when present), GPU (0),
#        EPISODES (20), TASKS (0-9), SEED (0), SUITE (libero_spatial), CHECKPOINT (lerobot/pi05_libero_finetuned_v044),
#        RAW (data/vla/raw/libero_spatial), DATA (data/vla/libero_spatial_reversal), OUT_DIR (outputs/vla),
#        STRIDE (2), NUM_STEPS (10), N_ACTION_STEPS (10), REVERSAL_BATCH (16), STEPS (12500), BATCH (256), LR (1e-4),
#        MODALITIES ("text sketch audio"), SIGMAS ("1 0"), PRIOR_CKPT (best.pt), VOICE (data/voices/en_US-lessac-medium.onnx),
#        EMBEDDER (google/embeddinggemma-2), EMBED_PYTHON (interpreter for embed_vla_instructions.py and the
#        evaluator's embedder worker; default $ROOT/.venv/bin/python when present, empty = current python),
#        STAGES ("setup data prior eval results"), DRY_RUN (1: print commands only), LOG_DIR (logs/vla),
#        DATA_ARGS / EMBED_ARGS / TRAIN_ARGS / EVAL_ARGS (extra flags for the four Python entry points).
# Every logged command runs with CUDA_VISIBLE_DEVICES=MUJOCO_EGL_DEVICE_ID=$GPU, MUJOCO_GL=PYOPENGL_PLATFORM=egl,
# LIBERO_CONFIG_PATH ($ROOT/data/vla/libero_config) and PYTHONPATH=$ROOT.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV="${VENV:-$ROOT/.venv-vla}"; GPU="${GPU:-0}"; EPISODES="${EPISODES:-20}"; TASKS="${TASKS:-0-9}"; SEED="${SEED:-0}"
SUITE="${SUITE:-libero_spatial}"; CHECKPOINT="${CHECKPOINT:-lerobot/pi05_libero_finetuned_v044}"
RAW="${RAW:-data/vla/raw/libero_spatial}"; DATA="${DATA:-data/vla/libero_spatial_reversal}"; OUT_DIR="${OUT_DIR:-outputs/vla}"
STRIDE="${STRIDE:-2}"; NUM_STEPS="${NUM_STEPS:-10}"; N_ACTION_STEPS="${N_ACTION_STEPS:-10}"; REVERSAL_BATCH="${REVERSAL_BATCH:-16}"
STEPS="${STEPS:-12500}"; BATCH="${BATCH:-256}"; LR="${LR:-1e-4}"
MODALITIES="${MODALITIES:-text sketch audio}"; SIGMAS="${SIGMAS:-1 0}"; PRIOR_CKPT="${PRIOR_CKPT:-best.pt}"
VOICE="${VOICE:-data/voices/en_US-lessac-medium.onnx}"; EMBEDDER="${EMBEDDER:-google/embeddinggemma-2}"
STAGES="${STAGES:-setup data prior eval results}"; LOG_DIR="${LOG_DIR:-logs/vla}"
DATA_ARGS="${DATA_ARGS:-}"; EMBED_ARGS="${EMBED_ARGS:-}"; TRAIN_ARGS="${TRAIN_ARGS:-}"; EVAL_ARGS="${EVAL_ARGS:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR" "$OUT_DIR"
if [[ -z "${EMBED_PYTHON+x}" ]]; then
  EMBED_PYTHON=""; [[ -x "$ROOT/.venv/bin/python" ]] && EMBED_PYTHON="$ROOT/.venv/bin/python"
fi

log() { echo "[$(date '+%F %T')] $*"; }
dry() { [[ -n "${DRY_RUN:-}" ]]; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
run() { log "+ $*"; dry || "$@"; }
run_logged() {  # run_logged <name> <cmd...>: on GPU $GPU with the EGL/LIBERO env below, stdout/stderr to $LOG_DIR/<name>.log
  local name="$1"; shift
  log "+ [gpu $GPU] $* > $LOG_DIR/$name.log"
  dry || env CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" MUJOCO_GL="$MUJOCO_GL" \
    PYOPENGL_PLATFORM="$PYOPENGL_PLATFORM" LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" PYTHONPATH="$PYTHONPATH" \
    "$@" > "$LOG_DIR/$name.log" 2>&1 < /dev/null
}
expand_tasks() {  # "0-9" / "0,3,5" / "2" -> "0 1 2 ..."
  local part out=()
  IFS=',' read -r -a parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then out+=($(seq "${part%-*}" "${part#*-}")); else out+=("$part"); fi
  done
  echo "${out[@]}"
}
TASK_LIST="$(expand_tasks "$TASKS")"; N_TASKS=$(wc -w <<< "$TASK_LIST")
MODALITY_CSV="$(echo "$MODALITIES" | tr ' ' ',')"

data_done() { local t; for t in $TASK_LIST; do [[ -f "$DATA/task_$t.npz" ]] || return 1; done; }
eval_done() {  # <out dir>: summary.json lists every requested episode
  [[ -f "$1/summary.json" ]] || return 1
  python - "$1/summary.json" "$((N_TASKS * EPISODES))" <<'EOF'
import json, sys
summary = json.load(open(sys.argv[1]))
sys.exit(0 if summary.get("overall", {}).get("episodes", 0) >= int(sys.argv[2]) else 1)
EOF
}
summary_line() {
  python - "$1/summary.json" <<'EOF'
import json, sys
o = json.load(open(sys.argv[1]))["overall"]
print(f"{o['episodes']} episodes, success {o['success_rate']:.3f}, bowl-choice {o['bowl_choice_accuracy']:.3f}")
EOF
}

# One campaign at a time per log directory.
if [[ -f "$LOG_DIR/campaign.pid" ]] && kill -0 "$(cat "$LOG_DIR/campaign.pid")" 2>/dev/null; then
  echo "another campaign is running (pid $(cat "$LOG_DIR/campaign.pid"), $LOG_DIR/campaign.pid)" >&2; exit 2
fi
echo $$ > "$LOG_DIR/campaign.pid"
trap 'rm -f "$LOG_DIR/campaign.pid"' EXIT

log "campaign: suite=$SUITE tasks=[$TASK_LIST] episodes=$EPISODES seed=$SEED gpu=$GPU stages=[$STAGES]${DRY_RUN:+ DRY_RUN}"

if want setup; then
  log "=== setup ($VENV)"
  if [[ -f "$LOG_DIR/setup.done" ]]; then log "setup done ($LOG_DIR/setup.done)"
  elif [[ -f scripts/vla/remote_setup_vla.sh ]]; then run env LEAP_ROOT="$ROOT" bash scripts/vla/remote_setup_vla.sh
  else log "scripts/vla/remote_setup_vla.sh not found; skipping"; fi
fi

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  log "warning: $VENV not found; using $(command -v python || echo 'no python')"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/data/vla/libero_config}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
VLA_PY=python; [[ -x "$VENV/bin/python" ]] && VLA_PY="$VENV/bin/python"
log "env: MUJOCO_GL=$MUJOCO_GL PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH PYTHONPATH=$PYTHONPATH python=$VLA_PY embed=${EMBED_PYTHON:-python}"

instr_done() {  # every task npz lists instr_modalities covering the selected modalities
  python - "$DATA" "$MODALITY_CSV" $TASK_LIST <<'EOF'
import sys
import numpy as np
data, modalities, tasks = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
for task in tasks:
    with np.load(f"{data}/task_{task}.npz") as npz:
        built = [str(m) for m in npz["instr_modalities"]] if "instr_modalities" in npz else []
        if any(m not in built for m in modalities):
            sys.exit(1)
EOF
}

if want data; then
  # Step 1 (VLA venv): replay, prefix features and reversal noises -> task_{i}.npz without instr_* keys, plus the
  # instruction artefacts.  Step 2 (phase-2 venv under EMBED_PYTHON): embed_vla_instructions.py adds
  # instr_text/sketch/audio [D, 1, 768] and instr_modalities to each npz (skips tasks already embedded).
  VOICE_ARGS=(); if [[ -f "$VOICE" ]] || dry; then VOICE_ARGS=(--voice "$VOICE"); fi
  log "=== data: reversal -> $DATA (stride $STRIDE, K=$NUM_STEPS)"
  if data_done; then log "reversal dataset present"
  else
    # shellcheck disable=SC2086
    run_logged data "$VLA_PY" scripts/vla/build_reversal_dataset.py --raw "$RAW" --out "$DATA" --checkpoint "$CHECKPOINT" \
      --suite "$SUITE" --tasks "$TASKS" --stride "$STRIDE" --num-steps "$NUM_STEPS" --batch-size "$REVERSAL_BATCH" \
      --device cuda --seed "$SEED" ${VOICE_ARGS[@]+"${VOICE_ARGS[@]}"} $DATA_ARGS
  fi
  log "=== data: embed [$MODALITIES] under ${EMBED_PYTHON:-python}"
  if ! dry && instr_done; then log "embeddings present"
  else
    # shellcheck disable=SC2086
    run_logged embed "${EMBED_PYTHON:-python}" scripts/vla/embed_vla_instructions.py --data "$DATA" \
      ${VOICE_ARGS[@]+"${VOICE_ARGS[@]}"} --device cuda --tasks "$TASKS" --modalities $MODALITIES \
      --embedder-model "$EMBEDDER" $EMBED_ARGS
  fi
fi

if want prior; then
  for MODE in none concat; do
    OUT="$OUT_DIR/prior-$MODE"
    log "=== prior $MODE -> $OUT ($STEPS steps)"
    if [[ -f "$OUT/prior.pt" ]]; then log "prior present"; continue; fi
    if ! dry && ! data_done; then log "skipped: dataset $DATA incomplete"; continue; fi
    dry || rm -rf "$OUT"
    # shellcheck disable=SC2086
    run_logged "train-prior-$MODE" python scripts/vla/train_vla_prior.py --data "$DATA" --out "$OUT" --instr-mode "$MODE" \
      --modalities "$MODALITY_CSV" --steps "$STEPS" --batch "$BATCH" --lr "$LR" --seed "$SEED" --device cuda $TRAIN_ARGS
  done
fi

eval_run() {  # eval_run <name> <condition> <modality> [<sigma> <prior ckpt>]
  local name="$1" condition="$2" modality="$3" sigma="${4:-}" prior="${5:-}" out="$OUT_DIR/$1"
  log "=== eval $name"
  if ! dry && eval_done "$out"; then log "done: $(summary_line "$out")"; return 0; fi
  if [[ -n "$prior" ]] && ! dry && [[ ! -f "$prior" ]]; then log "skipped: prior $prior missing"; return 0; fi
  local args=(--suite "$SUITE" --tasks "$TASKS" --episodes "$EPISODES" --condition "$condition" --instr-modality "$modality"
              --checkpoint "$CHECKPOINT" --num-steps "$NUM_STEPS" --n-action-steps "$N_ACTION_STEPS" --device cuda
              --seed "$SEED" --out "$out")
  [[ -n "$prior" ]] && args+=(--prior "$prior" --sigma-scale "$sigma")
  if [[ "$modality" != none ]]; then
    args+=(--voice "$VOICE" --embedder "$EMBEDDER")
    [[ -n "$EMBED_PYTHON" ]] && args+=(--embedder-python "$EMBED_PYTHON")
  fi
  # shellcheck disable=SC2086
  run_logged "eval-$name" python scripts/vla/eval_vla_steer.py "${args[@]}" $EVAL_ARGS
  dry || log "result: $(summary_line "$out")"
}

if want eval; then
  eval_run full-prompt-none full-prompt none
  eval_run neutral-gauss-none neutral-gauss none
  for S in $SIGMAS; do eval_run "neutral-prior-none-s$S" neutral-prior none "$S" "$OUT_DIR/prior-none/$PRIOR_CKPT"; done
  for M in $MODALITIES; do
    for S in $SIGMAS; do eval_run "neutral-prior-$M-s$S" neutral-prior "$M" "$S" "$OUT_DIR/prior-concat/$PRIOR_CKPT"; done
  done
fi

if want results; then
  log "=== results -> $OUT_DIR/results.{md,json} + results_per_task.csv"
  if dry; then log "+ python scripts/vla/collect_vla_results.py --out-dir $OUT_DIR"
  else python scripts/vla/collect_vla_results.py --out-dir "$OUT_DIR" | tee "$LOG_DIR/results.md"; fi
fi
log "campaign finished"
