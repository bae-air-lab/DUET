#!/usr/bin/env bash
# Run the four remaining paper ablations, then build the ablation table.
#
# Abl-Reference already completed (2026-08-07_07-53-22, 6000 iterations) and is
# NOT rerun -- it is the matched-budget control and rerunning it would only cost
# 3.4 h and introduce seed-level differences between the control and the table
# it anchors.
#
# Sequential by design: matched budget requires identical throughput conditions,
# and two runs sharing one GPU would change the wall-clock profile of both.
#
# Usage: duet_remaining_ablations.sh

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ABLATIONS=(
  "Unitree-G1-23Dof-Duet-Abl-NoSymmetry"
  "Unitree-G1-23Dof-Duet-Abl-NoHeightCmd"
  "Unitree-G1-23Dof-Duet-Abl-NoArmCurriculum"
  "Unitree-G1-23Dof-Duet-Abl-NoIdlePrecision"
)

log() { echo "[$(date '+%F %H:%M:%S')] $*"; }

log "starting ${#ABLATIONS[@]} remaining ablations (~3.4 h each, ~13.6 h total)"
for TASK in "${ABLATIONS[@]}"; do
  NAME=$(echo "$TASK" | sed 's/.*Duet-//')
  LOG="logs/duet_runs/abl_${NAME}.log"
  log "=== START $NAME ==="
  python scripts/train.py "$TASK" --agent.seed 1 --env.scene.num-envs 4096 \
    --agent.run-name "abl_${NAME}" > "$LOG" 2>&1
  RC=$?
  LAST=$(grep -o "Learning iteration [0-9]*" "$LOG" 2>/dev/null | tail -1)
  log "=== DONE $NAME (rc=$RC, $LAST) ==="
  if [ "$RC" -ne 0 ]; then
    log "    WARNING: $NAME exited non-zero; continuing with the rest so one "
    log "    failure does not cost the whole set. It will be skipped by the "
    log "    table's --min-iter guard if it did not reach 6000."
  fi
done

log "all ablations finished; building the paper table"
python -u scripts/duet_ablation_table.py \
  --runs-root logs/rsl_rl/DUET_G1_23dof_Ablation \
  --extra-run "$(ls keep/final_*/model_15500.pt 2>/dev/null | head -1)" \
  --num-envs 512 --steps 1400 --probe-envs 128
log "ABLATION TABLE COMPLETE"
