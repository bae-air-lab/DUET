#!/usr/bin/env bash
# Stop the ablation chain cleanly once Abl-NoArmCurriculum finishes, leaving
# Abl-NoIdlePrecision for tomorrow.
#
# The chain prints "DONE <name>" and immediately starts the next variant, so by
# the time we see the line the next run has already been spawned. We therefore
# kill the chain first (so it cannot spawn anything further) and then the
# freshly-started training process.
#
# Lives in a file rather than an inline command on purpose: `pgrep -f` matches
# against full command lines, so an inline version would match its own shell and
# kill itself.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CHAIN_PID=${1:?chain pid}
CHAIN_LOG=logs/duet_runs/remaining_ablations.log

log() { echo "[$(date '+%F %H:%M:%S')] $*"; }

log "watching for Abl-NoArmCurriculum to finish (chain pid $CHAIN_PID)"
while true; do
  if grep -q "DONE Abl-NoArmCurriculum" "$CHAIN_LOG" 2>/dev/null; then
    log "Abl-NoArmCurriculum finished."
    break
  fi
  if [ ! -e "/proc/$CHAIN_PID" ]; then
    log "chain exited before Abl-NoArmCurriculum completed; nothing to stop."
    exit 0
  fi
  sleep 20
done

# Kill the chain first so it cannot launch another variant.
if [ -e "/proc/$CHAIN_PID" ]; then
  kill "$CHAIN_PID" 2>/dev/null && log "stopped the chain ($CHAIN_PID)"
fi
sleep 3

# Then stop Abl-NoIdlePrecision if the chain already spawned it.
for _ in $(seq 1 15); do
  TPID=$(pgrep -f "scripts/train.py Unitree-G1-23Dof-Duet-Abl" | head -1)
  [ -z "$TPID" ] && break
  kill "$TPID" 2>/dev/null && log "stopping ablation training ($TPID)"
  sleep 2
done

sleep 3
if pgrep -f "scripts/train.py Unitree-G1-23Dof-Duet-Abl" >/dev/null; then
  log "WARNING: an ablation process is still alive; check manually."
else
  log "all ablation training stopped. 4 of 5 ablations complete."
  log "Remaining: Abl-NoIdlePrecision. Resume with scripts/duet_finish_ablations.sh"
fi
