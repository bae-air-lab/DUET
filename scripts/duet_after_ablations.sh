#!/usr/bin/env bash
# Runs after the ablation chain finishes: builds the paper's ablation table.
# Separate from duet_after_finetune.sh so a failure in either does not take the
# other down, and so the table can be rebuilt without rerunning any training.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CHAIN_PID=${1:?chain pid}
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "waiting for the ablation chain (pid $CHAIN_PID) to finish..."
while [ -e "/proc/$CHAIN_PID" ]; do sleep 120; done
log "ablation chain finished; building the paper table"
python -u scripts/duet_ablation_table.py \
  --runs-root logs/rsl_rl/DUET_G1_23dof_Ablation \
  --extra-run "$(ls keep/final_*/model_15500.pt 2>/dev/null | head -1)" \
  --num-envs 512 --steps 1400 --probe-envs 128
log "ablation table complete."
