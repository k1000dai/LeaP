#!/usr/bin/env bash
# Multimodal instruction-steering campaign on place_a2b_side (docs/superpowers/specs/2026-10-17-leap-multimodal-steering-design.md).
# Stages: install -> config -> collect -> convert -> bundle -> train -> eval -> results. Idempotent: finished stages
# are skipped. Training and evaluation jobs run up to PARALLEL at a time, spread over GPUS. Mirrors run_campaign.sh.
# Usage: scripts/run_steer_campaign.sh            (venv activated; the rasterizer shim is enabled here)
# Env:   ROBOTWIN_ROOT (required), GPUS ("0 1 2"), GPU (collection/embedding GPU, first of GPUS), PARALLEL (3),
#        TRAIN_PARALLEL / EVAL_PARALLEL (PARALLEL), SEED (0), EPISODES (100), TASK (place_a2b_side),
#        TASK_CONFIG (leap_steer), WORKERS (4), VARIANTS ("noinstr prior cond both"), MODALITIES ("text sketch audio"),
#        EPOCHS (config default), VOICE (data/voices/en_US-lessac-medium.onnx), EMBEDDER (google/embeddinggemma-2),
#        STAGES ("install config collect convert bundle train eval results"), DRY_RUN (1: print commands only),
#        LOG_DIR (logs/steer), OUTPUTS (outputs), STALL_MINUTES (3), MAX_COLLECT_ATTEMPTS (40)
# shellcheck disable=SC2012  # ls | wc -l on our own episode*/timestamp names, as in run_campaign.sh
set -euo pipefail
LEAP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT}"
GPUS="${GPUS:-0 1 2}"; read -r -a GPU_LIST <<< "$GPUS"; GPU="${GPU:-${GPU_LIST[0]}}"
PARALLEL="${PARALLEL:-3}"; TRAIN_PARALLEL="${TRAIN_PARALLEL:-$PARALLEL}"; EVAL_PARALLEL="${EVAL_PARALLEL:-$PARALLEL}"
SEED="${SEED:-0}"; EPISODES="${EPISODES:-100}"; WORKERS="${WORKERS:-4}"
TASK="${TASK:-place_a2b_side}"; TASK_CONFIG="${TASK_CONFIG:-leap_steer}"
VARIANTS="${VARIANTS:-noinstr prior cond both}"; MODALITIES="${MODALITIES:-text sketch audio}"
VOICE="${VOICE:-data/voices/en_US-lessac-medium.onnx}"; EMBEDDER="${EMBEDDER:-google/embeddinggemma-2}"
STAGES="${STAGES:-install config collect convert bundle train eval results}"
LOG_DIR="${LOG_DIR:-logs/steer}"; OUTPUTS="${OUTPUTS:-outputs}"
EPOCH_ARG=(); [[ -n "${EPOCHS:-}" ]] && EPOCH_ARG=(--epochs "$EPOCHS")
cd "$LEAP_ROOT"
mkdir -p "$LOG_DIR" "$OUTPUTS/analysis" data/raw
LOG_DIR="$(cd "$LOG_DIR" && pwd)"   # absolute: evaluations run with cwd=$ROBOTWIN_ROOT
[[ "$VOICE" == /* ]] || VOICE="$LEAP_ROOT/$VOICE"
export LEAP_RASTER="${LEAP_RASTER:-1}" LEAP_FAULTHANDLER="${LEAP_FAULTHANDLER:-1}"
export PYTHONPATH="$LEAP_ROOT/scripts/shims${PYTHONPATH:+:$PYTHONPATH}"

RAW="data/raw/$TASK/$TASK_CONFIG"
ZARR="data/$TASK-$TASK_CONFIG-$EPISODES.zarr"
CFG="$ROBOTWIN_ROOT/task_config/$TASK_CONFIG.yml"
RESULTS="$ROBOTWIN_ROOT/eval_result/$TASK/leap.robotwin.deploy/$TASK_CONFIG"

log() { echo "[$(date '+%F %T')] $*"; }
dry() { [[ -n "${DRY_RUN:-}" ]]; }
want() { [[ " $STAGES " == *" $1 "* ]]; }
run() { log "+ $*"; dry || "$@"; }
config_value() { python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))[sys.argv[2]])" "$CFG" "$1"; }
variant_epochs() {  # EPOCHS, else training.epochs of configs/steer_<variant>.yaml, else the LeaP default
  [[ -n "${EPOCHS:-}" ]] && { echo "$EPOCHS"; return; }
  python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['training']['epochs'])" "configs/steer_$1.yaml" 2>/dev/null || echo 3001
}
job_alive() { [[ -f "$LOG_DIR/$1.pid" ]] && kill -0 "$(cat "$LOG_DIR/$1.pid")" 2>/dev/null; }
train_done() {  # <output dir> <epochs>: DONE marker (written by this script) or metrics.jsonl ending at the last epoch
  [[ -f "$1/checkpoints/latest.ckpt" ]] || return 1
  [[ -f "$1/DONE" ]] && return 0
  [[ -s "$1/metrics.jsonl" ]] || return 1
  tail -n 1 "$1/metrics.jsonl" | python -c "import json,sys;sys.exit(0 if json.loads(sys.stdin.read())['epoch'] == int(sys.argv[1]) - 1 else 1)" "$2" 2>/dev/null
}

# One campaign at a time per log directory; background jobs keep their own pid files.
if [[ -f "$LOG_DIR/campaign.pid" ]] && kill -0 "$(cat "$LOG_DIR/campaign.pid")" 2>/dev/null; then
  echo "another campaign is running (pid $(cat "$LOG_DIR/campaign.pid"), $LOG_DIR/campaign.pid)" >&2; exit 2
fi
echo $$ > "$LOG_DIR/campaign.pid"
trap 'rm -f "$LOG_DIR/campaign.pid"' EXIT

# --- job pool: up to $POOL_LIMIT background jobs, each on the GPU with the fewest running jobs -------------
JOB_PIDS=(); JOB_NAMES=(); JOB_GPUS=(); FAILED=(); FOREIGN_PIDS=(); POOL_LIMIT="$PARALLEL"; DRY_JOBS=0
reap() {
  local i status pids=() names=() gpus=()
  for i in ${JOB_PIDS[@]+"${!JOB_PIDS[@]}"}; do
    if kill -0 "${JOB_PIDS[$i]}" 2>/dev/null; then
      pids+=("${JOB_PIDS[$i]}"); names+=("${JOB_NAMES[$i]}"); gpus+=("${JOB_GPUS[$i]}")
    else
      status=0; wait "${JOB_PIDS[$i]}" || status=$?
      if [[ $status -eq 0 ]]; then log "job ${JOB_NAMES[$i]} finished"
      else log "job ${JOB_NAMES[$i]} FAILED (exit $status); see $LOG_DIR/${JOB_NAMES[$i]}.log"; FAILED+=("${JOB_NAMES[$i]}"); fi
      rm -f "$LOG_DIR/${JOB_NAMES[$i]}.pid"
    fi
  done
  JOB_PIDS=(${pids[@]+"${pids[@]}"}); JOB_NAMES=(${names[@]+"${names[@]}"}); JOB_GPUS=(${gpus[@]+"${gpus[@]}"})
}
acquire_slot() {  # waits for a free slot and sets SLOT_GPU
  local g i count best="" best_count=999999
  if dry; then SLOT_GPU="${GPU_LIST[$((DRY_JOBS % ${#GPU_LIST[@]}))]}"; DRY_JOBS=$((DRY_JOBS + 1)); return 0; fi
  while [[ ${#JOB_PIDS[@]} -ge $POOL_LIMIT ]]; do sleep 30; reap; done
  for g in "${GPU_LIST[@]}"; do
    count=0
    for i in ${JOB_GPUS[@]+"${!JOB_GPUS[@]}"}; do [[ "${JOB_GPUS[$i]}" == "$g" ]] && count=$((count + 1)); done
    if [[ $count -lt $best_count ]]; then best="$g"; best_count=$count; fi
  done
  SLOT_GPU="$best"
}
start_job() {  # start_job <name> <done marker or ""> <cmd...>: runs on $SLOT_GPU, logs to $LOG_DIR/<name>.log
  local name="$1" marker="$2"; shift 2
  log "+ [gpu $SLOT_GPU] $* > $LOG_DIR/$name.log"
  dry && return 0
  ( CUDA_VISIBLE_DEVICES="$SLOT_GPU" "$@" > "$LOG_DIR/$name.log" 2>&1 < /dev/null && { [[ -z "$marker" ]] || touch "$marker"; } ) &
  local pid=$!
  echo "$pid" > "$LOG_DIR/$name.pid"
  JOB_PIDS+=("$pid"); JOB_NAMES+=("$name"); JOB_GPUS+=("$SLOT_GPU")
}
wait_jobs() {  # drain the pool and any job left running by a previous invocation; fail if any job failed
  local pid
  while [[ ${#JOB_PIDS[@]} -gt 0 ]]; do sleep 30; reap; done
  for pid in ${FOREIGN_PIDS[@]+"${FOREIGN_PIDS[@]}"}; do
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
  done
  FOREIGN_PIDS=()
  [[ ${#FAILED[@]} -eq 0 ]] || { log "failed jobs: ${FAILED[*]}"; exit 1; }
}

log "campaign: task=$TASK config=$TASK_CONFIG episodes=$EPISODES seed=$SEED gpus=[$GPUS] parallel=$PARALLEL stages=[$STAGES]${DRY_RUN:+ DRY_RUN}"

if want install; then
  log "=== [$TASK] install task into $ROBOTWIN_ROOT"
  if ! dry && python scripts/install_task.py --robotwin "$ROBOTWIN_ROOT" --check > /dev/null 2>&1; then
    log "task files installed"
  else
    run python scripts/install_task.py --robotwin "$ROBOTWIN_ROOT"
    run python scripts/install_task.py --robotwin "$ROBOTWIN_ROOT" --check
  fi
fi

if want config; then
  log "=== [$TASK] task config $CFG"
  if [[ -f "$CFG" ]]; then
    CFG_SAVE=$(config_value save_path); CFG_EPISODES=$(config_value episode_num)
    [[ "$CFG_SAVE" == "$LEAP_ROOT/data/raw" ]] || { echo "save_path in $CFG is $CFG_SAVE, expected $LEAP_ROOT/data/raw" >&2; exit 2; }
    [[ "$CFG_EPISODES" -ge "$EPISODES" ]] || { echo "episode_num=$CFG_EPISODES in $CFG < EPISODES=$EPISODES" >&2; exit 2; }
    log "config present (episode_num=$CFG_EPISODES)"
  else
    # Same recipe that produced leap_clean.yml (demo_clean.yml + point clouds, no videos, our save_path).
    run python scripts/robotwin.py prepare --robotwin "$ROBOTWIN_ROOT" --name "$TASK_CONFIG" --episodes "$EPISODES" \
      --save-path "$LEAP_ROOT/data/raw"
  fi
fi

if want collect; then
  log "=== [$TASK] collect $EPISODES episodes -> $RAW"
  if [[ -f "$RAW/data/episode$((EPISODES - 1)).hdf5" && $(ls "$RAW"/data/episode*.hdf5 2>/dev/null | wc -l) -ge $EPISODES ]]; then
    log "collection complete"
  elif dry; then
    run python scripts/robotwin.py collect --robotwin "$ROBOTWIN_ROOT" --task "$TASK" --task-config "$TASK_CONFIG" --gpu "$GPU"
  else
    # The official collector resumes from seed.txt and existing episode*.hdf5 files, so a stalled run
    # (observed: SAPIEN/Vulkan hang mid-episode) is killed and restarted; see run_campaign.sh.
    ATTEMPT=0
    while [[ $(ls "$RAW"/data/episode*.hdf5 2>/dev/null | wc -l) -lt "$EPISODES" ]]; do
      ATTEMPT=$((ATTEMPT + 1))
      [[ $ATTEMPT -le "${MAX_COLLECT_ATTEMPTS:-40}" ]] || { log "collect for $TASK stalled $ATTEMPT times; giving up"; exit 1; }
      log "--- collect attempt $ATTEMPT ($(ls "$RAW"/data/episode*.hdf5 2>/dev/null | wc -l)/$EPISODES episodes)"
      setsid python scripts/robotwin.py collect --robotwin "$ROBOTWIN_ROOT" --task "$TASK" \
        --task-config "$TASK_CONFIG" --gpu "$GPU" >> "$LOG_DIR/collect-$TASK.log" 2>&1 < /dev/null &
      CPID=$!
      echo "$CPID" > "$LOG_DIR/collect-$TASK.pid"
      LAST_CHANGE=$(date +%s)
      while kill -0 "$CPID" 2>/dev/null; do
        sleep 30
        NEWEST=$(find "$RAW" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
        [[ -n "$NEWEST" && "$NEWEST" -gt "$LAST_CHANGE" ]] && LAST_CHANGE=$NEWEST
        if [[ $(( $(date +%s) - LAST_CHANGE )) -gt $(( ${STALL_MINUTES:-3} * 60 )) ]]; then
          log "--- collect stalled for ${STALL_MINUTES:-3} min; killing attempt $ATTEMPT" | tee -a "$LOG_DIR/collect-$TASK.log"
          kill -USR1 -- -"$CPID" 2>/dev/null || true; sleep 5   # faulthandler stacks (see scripts/shims)
          kill -- -"$CPID" 2>/dev/null || true; sleep 5; kill -9 -- -"$CPID" 2>/dev/null || true
          break
        fi
      done
      wait "$CPID" 2>/dev/null || true
      rm -f "$LOG_DIR/collect-$TASK.pid"
      rm -rf "$RAW/.cache"
    done
  fi
  # RoboTwin's end-of-collection instruction generator cannot find scene_info.json when save_path is
  # absolute, so no instructions/ directory appears here; scripts/make_instructions.py generates the
  # texts itself from <raw>/scene_info.json with the same official functions (seeded per episode).
fi

if want convert; then
  log "=== [$TASK] convert -> $ZARR"
  if [[ -d "$ZARR" ]]; then log "zarr present"
  else run python scripts/robotwin.py convert --input "$RAW/data" --output "$ZARR" --episodes "$EPISODES"; fi
fi

BUNDLE=$(python -c "import sys; from leap.steer.bundle import bundle_path; print(bundle_path(sys.argv[1]))" "$ZARR" 2>/dev/null \
  || echo "${ZARR%.zarr}.instr")
if want bundle; then
  log "=== [$TASK] instruction bundle $BUNDLE"
  if [[ -f "$BUNDLE/episode$((EPISODES - 1))/sketch.png" ]]; then log "bundle media present"
  else run python scripts/make_instructions.py --raw "$RAW" --zarr "$ZARR" --voice "$VOICE"; fi
  if [[ -f "$BUNDLE/embeddings.npz" ]]; then log "embeddings present"
  else run env CUDA_VISIBLE_DEVICES="$GPU" python scripts/embed_instructions.py --bundle "$BUNDLE" --device cuda; fi
  PROBE="$OUTPUTS/analysis/instr_probe.json"
  if [[ -f "$PROBE" ]]; then log "probe present ($PROBE)"
  elif dry; then log "+ python scripts/instr_probe.py --bundle $BUNDLE --json $PROBE"
  else python scripts/instr_probe.py --bundle "$BUNDLE" --json "$PROBE" 2>&1 | tee "$LOG_DIR/instr_probe.log"; fi
fi

if want train; then
  POOL_LIMIT="$TRAIN_PARALLEL"
  for VARIANT in $VARIANTS; do
    OUT="$OUTPUTS/$TASK-$VARIANT-s$SEED"; VARIANT_CFG="configs/steer_$VARIANT.yaml"; NAME="train-$TASK-$VARIANT-s$SEED"
    TOTAL_EPOCHS=$(variant_epochs "$VARIANT")
    log "=== [$TASK] train $VARIANT -> $OUT ($TOTAL_EPOCHS epochs)"
    if train_done "$OUT" "$TOTAL_EPOCHS"; then log "training complete"; continue; fi
    if job_alive "$NAME"; then
      log "training already running (pid $(cat "$LOG_DIR/$NAME.pid")); waiting for it instead of restarting"
      FOREIGN_PIDS+=("$(cat "$LOG_DIR/$NAME.pid")"); continue
    fi
    INSTR_ARG=(); if [[ "$VARIANT" != noinstr ]]; then INSTR_ARG=(--instr "$BUNDLE"); fi
    acquire_slot
    dry || rm -rf "$OUT"
    start_job "$NAME" "$OUT/DONE" python -m leap.robotwin.train --config "$VARIANT_CFG" --data "$ZARR" --output "$OUT" \
      --device cuda:0 --seed "$SEED" --workers "$WORKERS" ${INSTR_ARG[@]+"${INSTR_ARG[@]}"} ${EPOCH_ARG[@]+"${EPOCH_ARG[@]}"}
  done
  wait_jobs
fi

if want eval; then
  POOL_LIMIT="$EVAL_PARALLEL"
  for VARIANT in $VARIANTS; do
    OUT="$OUTPUTS/$TASK-$VARIANT-s$SEED"; TOTAL_EPOCHS=$(variant_epochs "$VARIANT")
    VARIANT_MODALITIES="$MODALITIES"; [[ "$VARIANT" == noinstr ]] && VARIANT_MODALITIES="none"
    for MODALITY in $VARIANT_MODALITIES; do
      SETTING="$VARIANT-s$SEED-$MODALITY"; NAME="eval-$TASK-$SETTING"
      log "=== [$TASK] eval $SETTING"
      if ls "$RESULTS/$SETTING"/*/_result.txt > /dev/null 2>&1; then
        log "done: success_rate=$(tail -1 "$(ls -td "$RESULTS/$SETTING"/*/ | head -1)_result.txt")"; continue
      fi
      if ! dry && ! train_done "$OUT" "$TOTAL_EPOCHS"; then log "skipped: $OUT is not a finished training run"; continue; fi
      if job_alive "$NAME"; then
        log "evaluation already running (pid $(cat "$LOG_DIR/$NAME.pid"))"; FOREIGN_PIDS+=("$(cat "$LOG_DIR/$NAME.pid")"); continue
      fi
      MODALITY_ARGS=(--instruction-modality "$MODALITY")
      if [[ "$MODALITY" != none ]]; then MODALITY_ARGS+=(--tts-voice "$VOICE" --embedder "$EMBEDDER"); fi
      SIDE_LOG="$LOG_DIR/side_${VARIANT}_${MODALITY}.jsonl"
      acquire_slot
      dry || rm -f "$SIDE_LOG"   # no _result.txt: any previous attempt was incomplete
      start_job "$NAME" "" env LEAP_STEER_LOG="$SIDE_LOG" python scripts/robotwin.py eval --robotwin "$ROBOTWIN_ROOT" \
        --task "$TASK" --task-config "$TASK_CONFIG" --gpu "$SLOT_GPU" --checkpoint "$OUT/checkpoints/latest.ckpt" \
        --seed "$SEED" --ckpt-setting "$SETTING" "${MODALITY_ARGS[@]}"
    done
  done
  wait_jobs
fi

if want results; then
  log "=== [$TASK] results -> $OUTPUTS/analysis/steer_results.json"
  if dry; then
    log "+ python scripts/collect_results.py --robotwin $ROBOTWIN_ROOT --task-config $TASK_CONFIG --side-logs $LOG_DIR --output $OUTPUTS/analysis/steer_results.json"
  else
    python scripts/collect_results.py --robotwin "$ROBOTWIN_ROOT" --task-config "$TASK_CONFIG" --side-logs "$LOG_DIR" \
      --output "$OUTPUTS/analysis/steer_results.json" | tee "$LOG_DIR/results.md"
  fi
fi
log "campaign finished"
