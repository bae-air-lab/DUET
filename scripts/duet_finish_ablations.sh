#!/usr/bin/env bash
# Tomorrow's resume: run the last ablation, then build the paper table.
#
# Abl-Reference, Abl-NoSymmetry, Abl-NoHeightCmd and Abl-NoArmCurriculum are
# already complete (6000 iterations each) and are NOT rerun.
#
# Usage: nohup bash scripts/duet_finish_ablations.sh > logs/duet_runs/finish_ablations.log 2>&1 &

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
log() { echo "[$(date '+%F %H:%M:%S')] $*"; }

TASK="Unitree-G1-23Dof-Duet-Abl-NoIdlePrecision"
NAME="Abl-NoIdlePrecision"
LOG="logs/duet_runs/abl_${NAME}.log"

log "=== START $NAME (~2.5 h) ==="
python scripts/train.py "$TASK" --agent.seed 1 --env.scene.num-envs 4096 \
  --agent.run-name "abl_${NAME}" > "$LOG" 2>&1
RC=$?
log "=== DONE $NAME (rc=$RC, $(grep -o 'Learning iteration [0-9]*' "$LOG" 2>/dev/null | tail -1)) ==="

log "building the paper ablation table (all 5 variants + the deployed policy)"
python -u scripts/duet_ablation_table.py \
  --runs-root logs/rsl_rl/DUET_G1_23dof_Ablation \
  --extra-run "$(ls keep/final_*/model_15500.pt 2>/dev/null | head -1)" \
  --num-envs 512 --steps 1400 --probe-envs 128
log "ABLATION TABLE COMPLETE"
