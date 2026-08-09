#!/usr/bin/env bash
# Launch the DUET training run for one or more seeds.
#
# Seeding is already a first-class CLI argument (`--agent.seed`, applied to both
# the agent and the env by mjlab's train.py), so this script is only a
# convenience wrapper for running the 3 seeds HOMIE reports.
#
# Usage:
#   scripts/train_duet_seeds.sh                       # seeds 1 2 3, GPU 0, sequential
#   scripts/train_duet_seeds.sh "1 2 3" "0 1 2"       # one seed per GPU, parallel
#   TASK=Unitree-G1-23Dof-Duet-Rough scripts/train_duet_seeds.sh "7"
#
# Resume is NOT special-cased: every curriculum reads env.common_step_counter,
# which is written into and restored from the checkpoint, so
#   python scripts/train.py --task <id> --agent.resume True --agent.load-run <run>
# continues on the same distribution it left off. No source edit is required.

set -euo pipefail

SEEDS=${1:-"1 2 3"}
GPUS=${2:-""}
TASK=${TASK:-Unitree-G1-23Dof-Duet-Flat}
NUM_ENVS=${NUM_ENVS:-4096}
EXTRA=${EXTRA:-}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

read -r -a SEED_ARR <<<"$SEEDS"
read -r -a GPU_ARR <<<"$GPUS"

echo "task     : $TASK"
echo "seeds    : ${SEED_ARR[*]}"
echo "num_envs : $NUM_ENVS"

pids=()
for i in "${!SEED_ARR[@]}"; do
  seed="${SEED_ARR[$i]}"
  run_name="duet_seed${seed}_$(date +%Y%m%d_%H%M%S)"
  cmd=(python scripts/train.py
       --task "$TASK"
       --agent.seed "$seed"
       --agent.run-name "$run_name"
       --env.scene.num-envs "$NUM_ENVS")
  [ -n "$EXTRA" ] && read -r -a extra_arr <<<"$EXTRA" && cmd+=("${extra_arr[@]}")

  if [ ${#GPU_ARR[@]} -gt 0 ]; then
    gpu="${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}"
    echo "-> seed $seed on GPU $gpu (background), log: logs/${run_name}.log"
    mkdir -p logs
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"logs/${run_name}.log" 2>&1 &
    pids+=($!)
  else
    echo "-> seed $seed (foreground)"
    "${cmd[@]}"
  fi
done

if [ ${#pids[@]} -gt 0 ]; then
  echo "waiting on ${#pids[@]} run(s): ${pids[*]}"
  wait "${pids[@]}"
fi
echo "done."
