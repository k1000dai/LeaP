#!/usr/bin/env bash
# Phase 3b: prior trained on the counterfactual (bowl-1 + bowl-2) reversal dataset, evaluated for either bowl.
# Stages: embed -> prior -> eval -> results, sequential on one GPU; idempotent like scripts/vla/run_vla_campaign.sh
# (the reversal itself is scripts/vla/reverse_rollouts.py, run beforehand).  Evaluation uses the init states
# FIRST_EPISODE.. (default 20..39), held out from the rollouts (states 0..19).
# Usage: scripts/vla/run_vla_counterfactual.sh
# Env:   ROOT, VENV (.venv-vla), GPU (0), EPISODES (20), FIRST_EPISODE (20), TASKS (0-9), SEED (0), SUITE, CHECKPOINT,
#        DATA (data/vla/libero_spatial_counterfactual), DATA_TASKS (npz indices of the dataset; default: the bowl-1 files
#        100-109 plus every bowl-2 file present — layouts without a successful bowl-2 rollout have none),
#        OUT_DIR (outputs/vla_cf), LOG_DIR (logs/vla_cf), NUM_STEPS (10), N_ACTION_STEPS (10), STEPS (12500), BATCH (256),
#        LR (1e-4), MODALITIES ("text sketch audio"), SIGMAS ("1"), PRIOR_MODES ("concat none"), PRIOR_CKPT (best.pt),
#        BOWLS ("2 1"), CELLS (names to evaluate; default: all), VOICE, EMBEDDER, EMBED_PYTHON (as in run_vla_campaign.sh),
#        STAGES ("embed prior eval results"), DRY_RUN (1), EMBED_ARGS / TRAIN_ARGS / EVAL_ARGS.
# Cells: full-prompt-b2, neutral-gauss-b2, neutral-gauss-b1, then neutral-prior-<mode>-<modality>-s<sigma>-b<bowl>
# (mode none has modality none).  Several instances may run at once with different GPU/CELLS (lock per GPU).
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
VENV="${VENV:-$ROOT/.venv-vla}"; GPU="${GPU:-0}"; EPISODES="${EPISODES:-20}"; FIRST_EPISODE="${FIRST_EPISODE:-20}"
TASKS="${TASKS:-0-9}"; SEED="${SEED:-0}"; SUITE="${SUITE:-libero_spatial}"; CHECKPOINT="${CHECKPOINT:-lerobot/pi05_libero_finetuned_v044}"
DATA="${DATA:-data/vla/libero_spatial_counterfactual}"; DATA_TASKS="${DATA_TASKS:-}"
OUT_DIR="${OUT_DIR:-outputs/vla_cf}"; LOG_DIR="${LOG_DIR:-logs/vla_cf}"
NUM_STEPS="${NUM_STEPS:-10}"; N_ACTION_STEPS="${N_ACTION_STEPS:-10}"; STEPS="${STEPS:-12500}"; BATCH="${BATCH:-256}"; LR="${LR:-1e-4}"
MODALITIES="${MODALITIES:-text sketch audio}"; SIGMAS="${SIGMAS:-1}"; PRIOR_MODES="${PRIOR_MODES:-concat none}"
PRIOR_CKPT="${PRIOR_CKPT:-best.pt}"; BOWLS="${BOWLS:-2 1}"; CELLS="${CELLS:-}"
VOICE="${VOICE:-data/voices/en_US-lessac-medium.onnx}"; EMBEDDER="${EMBEDDER:-google/embeddinggemma-2}"
STAGES="${STAGES:-embed prior eval results}"
EMBED_ARGS="${EMBED_ARGS:-}"; TRAIN_ARGS="${TRAIN_ARGS:-}"; EVAL_ARGS="${EVAL_ARGS:-}"
cd "$ROOT"
mkdir -p "$LOG_DIR" "$OUT_DIR"
if [[ -z "${EMBED_PYTHON+x}" ]]; then
  EMBED_PYTHON=""; [[ -x "$ROOT/.venv/bin/python" ]] && EMBED_PYTHON="$ROOT/.venv/bin/python"
fi

log() { echo "[$(date '+%F %T')] $*"; }
dry() { [[ -n "${DRY_RUN:-}" ]]; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
wanted_cell() { [[ -z "$CELLS" ]] || [[ " $CELLS " == *" $1 "* ]]; }
run_logged() {  # run_logged <name> <cmd...>
  local name="$1"; shift
  log "+ [gpu $GPU] $* > $LOG_DIR/$name.log"
  dry || env CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" MUJOCO_GL="$MUJOCO_GL" \
    PYOPENGL_PLATFORM="$PYOPENGL_PLATFORM" LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH" PYTHONPATH="$PYTHONPATH" \
    "$@" > "$LOG_DIR/$name.log" 2>&1 < /dev/null
}
expand_tasks() {
  local part out=()
  IFS=',' read -r -a parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then out+=($(seq "${part%-*}" "${part#*-}")); else out+=("$part"); fi
  done
  echo "${out[@]}"
}
TASK_LIST="$(expand_tasks "$TASKS")"; N_TASKS=$(wc -w <<< "$TASK_LIST")
if [[ -z "$DATA_TASKS" ]]; then
  DATA_TASKS="100-109"
  for f in "$DATA"/task_2[0-9][0-9].npz; do [[ -f "$f" ]] && DATA_TASKS+=",$(basename "$f" .npz | cut -d_ -f2)"; done
fi
DATA_TASK_LIST="$(expand_tasks "$DATA_TASKS")"
MODALITY_CSV="$(echo "$MODALITIES" | tr ' ' ',')"

data_done() { local t; for t in $DATA_TASK_LIST; do [[ -f "$DATA/task_$t.npz" ]] || return 1; done; }
eval_done() {
  [[ -f "$1/summary.json" ]] || return 1
  python - "$1/summary.json" "$((N_TASKS * EPISODES))" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
sys.exit(0 if summary.get("overall", {}).get("episodes", 0) >= int(sys.argv[2]) else 1)
PY
}
summary_line() {
  python - "$1/summary.json" <<'PY'
import json, sys
o = json.load(open(sys.argv[1]))["overall"]
print(f"{o['episodes']} episodes, success {o['success_rate']:.3f}, bowl-choice {o['bowl_choice_accuracy']:.3f}")
PY
}

LOCK="$LOG_DIR/campaign-gpu$GPU.pid"
if [[ -f "$LOCK" ]] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "another campaign is running on gpu $GPU (pid $(cat "$LOCK"), $LOCK)" >&2; exit 2
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

log "counterfactual campaign: suite=$SUITE tasks=[$TASK_LIST] episodes=$EPISODES from $FIRST_EPISODE seed=$SEED gpu=$GPU stages=[$STAGES] cells=[${CELLS:-all}]${DRY_RUN:+ DRY_RUN}"

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  log "warning: $VENV not found; using $(command -v python || echo 'no python')"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/data/vla/libero_config}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
log "env: MUJOCO_GL=$MUJOCO_GL PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH PYTHONPATH=$PYTHONPATH embed=${EMBED_PYTHON:-python}"

instr_done() {
  python - "$DATA" "$MODALITY_CSV" $DATA_TASK_LIST <<'PY'
import sys
import numpy as np
data, modalities, tasks = sys.argv[1], sys.argv[2].split(","), sys.argv[3:]
for task in tasks:
    with np.load(f"{data}/task_{task}.npz") as npz:
        built = [str(m) for m in npz["instr_modalities"]] if "instr_modalities" in npz else []
        if any(m not in built for m in modalities):
            sys.exit(1)
PY
}

if want embed; then
  VOICE_ARGS=(); if [[ -f "$VOICE" ]] || dry; then VOICE_ARGS=(--voice "$VOICE"); fi
  log "=== embed [$MODALITIES] of $DATA (tasks $DATA_TASKS) under ${EMBED_PYTHON:-python}"
  if ! dry && ! data_done; then log "skipped: reversal dataset $DATA incomplete (scripts/vla/reverse_rollouts.py)"
  elif ! dry && instr_done; then log "embeddings present"
  else
    # shellcheck disable=SC2086
    run_logged embed "${EMBED_PYTHON:-python}" scripts/vla/embed_vla_instructions.py --data "$DATA" \
      ${VOICE_ARGS[@]+"${VOICE_ARGS[@]}"} --device cuda --tasks "$DATA_TASKS" --modalities $MODALITIES \
      --embedder-model "$EMBEDDER" $EMBED_ARGS
  fi
fi

if want prior; then
  for MODE in $PRIOR_MODES; do
    OUT="$OUT_DIR/prior-cf-$MODE"
    log "=== prior $MODE -> $OUT ($STEPS steps)"
    if [[ -f "$OUT/prior.pt" ]]; then log "prior present"; continue; fi
    if ! dry && ! data_done; then log "skipped: dataset $DATA incomplete"; continue; fi
    dry || rm -rf "$OUT"
    # shellcheck disable=SC2086
    run_logged "train-prior-cf-$MODE" python scripts/vla/train_vla_prior.py --data "$DATA" --out "$OUT" --instr-mode "$MODE" \
      --modalities "$MODALITY_CSV" --steps "$STEPS" --batch "$BATCH" --lr "$LR" --seed "$SEED" --device cuda $TRAIN_ARGS
  done
fi

eval_run() {  # eval_run <name> <condition> <modality> <bowl> [<sigma> <prior ckpt>]
  local name="$1" condition="$2" modality="$3" bowl="$4" sigma="${5:-}" prior="${6:-}" out="$OUT_DIR/$1"
  wanted_cell "$name" || return 0
  log "=== eval $name"
  if ! dry && eval_done "$out"; then log "done: $(summary_line "$out")"; return 0; fi
  if [[ -n "$prior" ]] && ! dry && [[ ! -f "$prior" ]]; then log "skipped: prior $prior missing"; return 0; fi
  local args=(--suite "$SUITE" --tasks "$TASKS" --episodes "$EPISODES" --first-episode "$FIRST_EPISODE"
              --condition "$condition" --instr-modality "$modality" --target-bowl "$bowl"
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
  eval_run full-prompt-b2 full-prompt none 2
  eval_run neutral-gauss-b2 neutral-gauss none 2
  eval_run neutral-gauss-b1 neutral-gauss none 1
  for B in $BOWLS; do
    for MODE in $PRIOR_MODES; do
      PRIOR="$OUT_DIR/prior-cf-$MODE/$PRIOR_CKPT"
      if [[ "$MODE" == none ]]; then
        for S in $SIGMAS; do eval_run "neutral-prior-none-none-s$S-b$B" neutral-prior none "$B" "$S" "$PRIOR"; done
      else
        for M in $MODALITIES; do
          for S in $SIGMAS; do eval_run "neutral-prior-$MODE-$M-s$S-b$B" neutral-prior "$M" "$B" "$S" "$PRIOR"; done
        done
      fi
    done
  done
fi

if want results; then
  log "=== results -> $OUT_DIR/results.{md,json} + results_per_task.csv"
  if dry; then log "+ python scripts/vla/collect_vla_results.py --out-dir $OUT_DIR"
  else python scripts/vla/collect_vla_results.py --out-dir "$OUT_DIR" | tee "$LOG_DIR/results.md"; fi
fi
log "campaign finished"
