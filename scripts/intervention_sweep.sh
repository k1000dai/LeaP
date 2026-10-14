#!/usr/bin/env bash
# Official 100-episode evaluations of one LeaP checkpoint under inference-time prior interventions.
# Usage: scripts/intervention_sweep.sh <task> <checkpoint> [<label>]
# Env:   ROBOTWIN_ROOT (required), GPU (0), SEED (0), TASK_CONFIG (leap_clean)
set -euo pipefail
LEAP_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${ROBOTWIN_ROOT:?set ROBOTWIN_ROOT}"
[[ $# -ge 2 ]] || { echo "usage: $0 <task> <checkpoint> [<label>]" >&2; exit 2; }
TASK="$1"; CKPT="$2"; LABEL="${3:-leap-s${SEED:-0}}"
GPU="${GPU:-0}"; SEED="${SEED:-0}"; TASK_CONFIG="${TASK_CONFIG:-leap_clean}"
cd "$LEAP_ROOT"; mkdir -p logs

# name|json  (names must match [A-Za-z0-9_-]+)
INTERVENTIONS=(
  "sigma0|{\"sigma_scale\": 0}"
  "sigma0p5|{\"sigma_scale\": 0.5}"
  "sigma2|{\"sigma_scale\": 2}"
  "sigma4|{\"sigma_scale\": 4}"
  "mu0-sigma1|{\"mu_override\": $(python -c 'print([0.0]*14)'), \"logvar_override\": $(python -c 'print([0.0]*14)')}"
  "directmu|{\"direct_mu\": true}"
)
for ENTRY in "${INTERVENTIONS[@]}"; do
  NAME="${ENTRY%%|*}"; JSON="${ENTRY#*|}"
  SETTING="$LABEL-$NAME"
  RESULT_DIR="$ROBOTWIN_ROOT/eval_result/$TASK/leap.robotwin.deploy/$TASK_CONFIG/$SETTING"
  echo "=== [$TASK] intervention $NAME: $JSON"
  if ! ls "$RESULT_DIR"/*/_result.txt >/dev/null 2>&1; then
    python scripts/robotwin.py eval --robotwin "$ROBOTWIN_ROOT" --task "$TASK" --task-config "$TASK_CONFIG" \
      --checkpoint "$CKPT" --gpu "$GPU" --seed "$SEED" --ckpt-setting "$SETTING" --intervention "$JSON" \
      > "logs/eval-$TASK-$SETTING.log" 2>&1
  fi
  echo "$TASK $NAME success_rate=$(tail -1 "$(ls -td "$RESULT_DIR"/*/ | head -1)_result.txt")"
done
