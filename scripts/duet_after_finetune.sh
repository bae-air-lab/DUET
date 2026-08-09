#!/usr/bin/env bash
# Chain that runs unattended after the refinement fine-tune finishes:
#   1. wait for training to exit
#   2. pick the best checkpoint by the measured battery ranking (not by recency)
#   3. export it to ONNX with the deployment contract, and verify against deploy.yaml
#   4. launch the paper ablations sequentially
#
# Sequential by design: the ablations must be matched-budget and comparable, and
# running two on one GPU changes throughput for both.
#
# Usage: duet_after_finetune.sh <train_pid> <run_dir>

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD
TRAIN_PID=${1:?train pid}
RUN_DIR=${2:?run dir}
RESULTS=logs/duet_runs/v3_results.jsonl
STAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for training pid $TRAIN_PID to finish..."
while [ -e "/proc/$TRAIN_PID" ]; do sleep 60; done
log "training finished."

# Give the battery a chance to finish the last checkpoints it is mid-way through.
log "waiting for the validation battery to drain..."
for _ in $(seq 1 40); do
  pgrep -f "duet_overnight.p[y]" >/dev/null || break
  sleep 60
done

log "selecting best checkpoint from measured results"
BEST=$(python - "$RESULTS" <<'PY'
import json, sys, importlib.util as iu
spec = iu.spec_from_file_location("ov", "scripts/duet_overnight.py")
ov = iu.module_from_spec(spec); spec.loader.exec_module(ov)
rows = []
for line in open(sys.argv[1]):
    try: rows.append(json.loads(line))
    except Exception: pass
rows = [r for r in rows if r.get("iteration")]
if not rows:
    print(""); raise SystemExit
rows.sort(key=ov.score)
print(rows[0]["checkpoint"])
PY
)

if [ -z "$BEST" ]; then
  log "no measured checkpoints; falling back to the latest"
  BEST=$(ls "$RUN_DIR"/model_*.pt | sort -t_ -k2 -n | tail -1)
fi
log "BEST CHECKPOINT: $BEST"

OUT="keep/final_${STAMP}"
mkdir -p "$OUT"
cp "$BEST" "$OUT/" 2>/dev/null
cp "$RUN_DIR"/params/*.yaml "$OUT/" 2>/dev/null
log "exporting + verifying against deploy.yaml"
python scripts/export_duet_onnx.py --task Unitree-G1-23Dof-Duet-Flat \
  --checkpoint "$BEST" --out-dir "$OUT/exported" > "$OUT/export.log" 2>&1
EXPORT_RC=$?
log "export/verify exit code: $EXPORT_RC  (0 = safe to deploy)"
tail -30 "$OUT/export.log" | grep -E "PASS|FAIL|SKIP|WARN|checks|hash" || true

# --- Paper ablations -------------------------------------------------------
# Priority order: the three HOMIE contributions each ablated, our idle-precision
# group, and the reference control. Secondary variants are left for later --
# each run is ~4.7 h at 4096 envs, so the full set of 11 is >50 h on one GPU.
ABLATIONS=(
  "Unitree-G1-23Dof-Duet-Abl-Reference"
  "Unitree-G1-23Dof-Duet-Abl-NoSymmetry"
  "Unitree-G1-23Dof-Duet-Abl-NoHeightCmd"
  "Unitree-G1-23Dof-Duet-Abl-NoArmCurriculum"
  "Unitree-G1-23Dof-Duet-Abl-NoIdlePrecision"
)

log "starting ${#ABLATIONS[@]} paper ablations, sequentially"
for TASK in "${ABLATIONS[@]}"; do
  NAME=$(echo "$TASK" | sed 's/.*Duet-//')
  LOG="logs/duet_runs/abl_${NAME}.log"
  log "=== ablation $NAME ==="
  python scripts/train.py "$TASK" --agent.seed 1 --env.scene.num-envs 4096 \
    --agent.run-name "abl_${NAME}" > "$LOG" 2>&1
  log "  $NAME finished (rc=$?)"
done
log "all ablations complete."
