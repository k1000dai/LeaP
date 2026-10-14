#!/usr/bin/env bash
# Sequential LeaP-vs-NoPrior campaign on one GPU. Idempotent: finished stages are skipped.
# Usage: scripts/run_campaign.sh <task> [<task> ...]
# Env:   ROBOTWIN_ROOT (required), GPU (default 0), SEED (0), EPOCHS (config default),
#        EPISODES (50), TASK_CONFIG (leap_clean), WORKERS (4)
set -euo pipefail
LEAP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT}"
GPU="${GPU:-0}"; SEED="${SEED:-0}"; EPISODES="${EPISODES:-50}"
TASK_CONFIG="${TASK_CONFIG:-leap_clean}"; WORKERS="${WORKERS:-4}"
EPOCH_ARG=(); [[ -n "${EPOCHS:-}" ]] && EPOCH_ARG=(--epochs "$EPOCHS")
[[ $# -ge 1 ]] || { echo "usage: $0 <task> [<task> ...]" >&2; exit 2; }
cd "$LEAP_ROOT"
mkdir -p logs outputs data/raw

# The task config decides where collect_data.py writes and how many episodes it records.
CFG="$ROBOTWIN_ROOT/task_config/$TASK_CONFIG.yml"
CFG_SAVE=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['save_path'])" "$CFG")
CFG_EPISODES=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['episode_num'])" "$CFG")
[[ "$CFG_SAVE" == "$LEAP_ROOT/data/raw" ]] || { echo "save_path in $CFG is $CFG_SAVE, expected $LEAP_ROOT/data/raw" >&2; exit 2; }
[[ "$CFG_EPISODES" -ge "$EPISODES" ]] || { echo "episode_num=$CFG_EPISODES in $CFG < EPISODES=$EPISODES" >&2; exit 2; }

for TASK in "$@"; do
  RAW="data/raw/$TASK/$TASK_CONFIG/data"
  ZARR="data/$TASK-$TASK_CONFIG-$EPISODES.zarr"
  echo "=== [$TASK] collect"
  # The official collector resumes from seed.txt and existing episode*.hdf5 files,
  # so a stalled run (observed: SAPIEN/Vulkan hang mid-episode) is killed and restarted.
  ATTEMPT=0
  while [[ $(ls "$RAW"/episode*.hdf5 2>/dev/null | wc -l) -lt "$EPISODES" ]]; do
    ATTEMPT=$((ATTEMPT + 1))
    [[ $ATTEMPT -le "${MAX_COLLECT_ATTEMPTS:-40}" ]] || { echo "collect for $TASK stalled $ATTEMPT times; giving up" >&2; exit 1; }
    echo "--- collect attempt $ATTEMPT"
    setsid python scripts/robotwin.py collect --robotwin "$ROBOTWIN_ROOT" --task "$TASK" \
      --task-config "$TASK_CONFIG" --gpu "$GPU" >> "logs/collect-$TASK.log" 2>&1 < /dev/null &
    CPID=$!
    LAST_CHANGE=$(date +%s)
    while kill -0 "$CPID" 2>/dev/null; do
      sleep 30
      NEWEST=$(find "data/raw/$TASK/$TASK_CONFIG" -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
      [[ -n "$NEWEST" && "$NEWEST" -gt "$LAST_CHANGE" ]] && LAST_CHANGE=$NEWEST
      if [[ $(( $(date +%s) - LAST_CHANGE )) -gt $(( ${STALL_MINUTES:-3} * 60 )) ]]; then
        echo "--- collect stalled for ${STALL_MINUTES:-3} min; killing attempt $ATTEMPT" | tee -a "logs/collect-$TASK.log"
        kill -USR1 -- -"$CPID" 2>/dev/null || true; sleep 5   # faulthandler stacks (see scripts/shims)
        kill -- -"$CPID" 2>/dev/null || true; sleep 5; kill -9 -- -"$CPID" 2>/dev/null || true
        break
      fi
    done
    wait "$CPID" 2>/dev/null || true
    rm -rf "data/raw/$TASK/$TASK_CONFIG/.cache"
  done
  echo "=== [$TASK] convert"
  if [[ ! -d "$ZARR" ]]; then
    python scripts/robotwin.py convert --input "$RAW" --output "$ZARR" --episodes "$EPISODES"
  fi
  for METHOD in leap noprior; do
    OUT="outputs/$TASK-$METHOD-s$SEED"
    echo "=== [$TASK] train $METHOD -> $OUT"
    if [[ ! -f "$OUT/checkpoints/latest.ckpt" || ! -f "$OUT/DONE" ]]; then
      rm -rf "$OUT"
      CUDA_VISIBLE_DEVICES="$GPU" python -m leap.robotwin.train --config "configs/$METHOD.yaml" \
        --data "$ZARR" --output "$OUT" --device cuda:0 --seed "$SEED" --workers "$WORKERS" \
        ${EPOCH_ARG[@]+"${EPOCH_ARG[@]}"} > "logs/train-$TASK-$METHOD-s$SEED.log" 2>&1
      touch "$OUT/DONE"
    fi
    echo "=== [$TASK] eval $METHOD"
    RESULT_DIR="$ROBOTWIN_ROOT/eval_result/$TASK/leap.robotwin.deploy/$TASK_CONFIG/$METHOD-s$SEED"
    if ! ls "$RESULT_DIR"/*/_result.txt >/dev/null 2>&1; then
      python scripts/robotwin.py eval --robotwin "$ROBOTWIN_ROOT" --task "$TASK" \
        --task-config "$TASK_CONFIG" --checkpoint "$OUT/checkpoints/latest.ckpt" \
        --gpu "$GPU" --seed "$SEED" --ckpt-setting "$METHOD-s$SEED" \
        > "logs/eval-$TASK-$METHOD-s$SEED.log" 2>&1
    fi
    echo "$TASK $METHOD success_rate=$(tail -1 "$(ls -td "$RESULT_DIR"/*/ | head -1)_result.txt")"
  done
done
