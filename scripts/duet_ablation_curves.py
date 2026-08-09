"""Per-checkpoint ablation curves, in the structure of HOMIE's Fig. 7.

That figure is a grid: one row per ablated component, one column per metric,
each panel plotting the metric across training checkpoints for the ablated
variant against the reference. This produces the underlying data.

Every checkpoint is evaluated under the SAME fixed protocol (the reference task
config) with only the trained policy varying -- evaluating each variant under
its own ablated config would change the test alongside the treatment.

Metrics match HOMIE Table II: linear velocity error, angular velocity error,
height error, symmetry loss, living time.

Usage:
  python scripts/duet_ablation_curves.py [--every 1000]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

REPO = "/home/rbbist.lab/unitree_rl_mjlab"
DUET_EVAL = "/home/rbbist.lab/paper_rl/duet_bench/duet_eval.py"
PROTOCOL_TASK = "Unitree-G1-23Dof-Duet-Flat"

# The variants that appear in the paper. NoIdlePrecision is excluded: those
# reward terms are part of the implementation but are not a claimed
# contribution, and papers ablate claims, not every reward term.
VARIANTS = ["Abl-Reference", "Abl-NoSymmetry", "Abl-NoHeightCmd",
            "Abl-NoArmCurriculum"]


def symmetry_loss_at(run_dir: str, iteration: int) -> float | None:
  """Mean Loss/symmetry in a window around the checkpoint's iteration."""
  try:
    from tensorboard.backend.event_processing import event_accumulator
  except ImportError:
    return None
  ev = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
  if not ev:
    return None
  try:
    ea = event_accumulator.EventAccumulator(ev[0], size_guidance={"scalars": 0})
    ea.Reload()
    tags = [t for t in ea.Tags().get("scalars", []) if "symmetry" in t.lower()]
    if not tags:
      return None
    vals = [(s.step, s.value) for s in ea.Scalars(tags[0])]
    win = [v for st, v in vals if abs(st - iteration) <= 250]
    if not win:
      win = [v for _, v in vals[-10:]]
    return sum(win) / max(len(win), 1)
  except Exception:  # noqa: BLE001
    return None


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs-root", default="logs/rsl_rl/DUET_G1_23dof_Ablation")
  ap.add_argument("--every", type=int, default=1000)
  ap.add_argument("--num-envs", type=int, default=384)
  ap.add_argument("--steps", type=int, default=1300)
  ap.add_argument("--out", default="logs/duet_runs/ablation_curves.jsonl")
  args = ap.parse_args()

  done = set()
  if os.path.exists(args.out):
    for line in open(args.out):
      try:
        r = json.loads(line)
        done.add((r["variant"], r["iteration"]))
      except Exception:  # noqa: BLE001
        pass

  runs = {}
  for d in glob.glob(os.path.join(args.runs_root, "*")):
    for v in VARIANTS:
      if d.endswith(v):
        runs[v] = d

  missing = [v for v in VARIANTS if v not in runs]
  if missing:
    print(f"[warn] no run directory for: {missing}")

  todo = []
  for v in VARIANTS:
    if v not in runs:
      continue
    for p in glob.glob(os.path.join(runs[v], "model_*.pt")):
      it = int(re.search(r"model_(\d+)\.pt$", p).group(1))
      if it > 0 and it % args.every == 0 and (v, it) not in done:
        todo.append((v, it, p, runs[v]))
  todo.sort(key=lambda x: (VARIANTS.index(x[0]), x[1]))
  print(f"{len(todo)} checkpoint evaluations to run\n", flush=True)

  for i, (v, it, path, run_dir) in enumerate(todo, 1):
    tag = f"curve_{v}_{it}"
    print(f"[{i}/{len(todo)}] {v} @ {it}", flush=True)
    rc = subprocess.call([
      sys.executable, DUET_EVAL, "--checkpoint", path, "--task", PROTOCOL_TASK,
      "--tag", tag, "--out", args.out + ".raw",
      "--num-envs", str(args.num_envs), "--steps", str(args.steps),
      "--payload", "1.0",
    ], cwd=REPO)
    if rc != 0:
      print("    [warn] eval failed; skipping", flush=True)
      continue
    try:
      last = json.loads(open(args.out + ".raw").readlines()[-1])
    except Exception:  # noqa: BLE001
      continue
    row = {
      "variant": v, "iteration": it,
      "lin_vel_err": last.get("mean_vxy_err_mps"),
      "ang_vel_err": last.get("mean_vyaw_err_radps"),
      "height_err": last.get("mean_height_err_m"),
      "living_time": last.get("mean_living_time_s"),
      "falls_per_env_min": last.get("falls_per_env_minute"),
      "symmetry_loss": symmetry_loss_at(run_dir, it),
    }
    with open(args.out, "a") as f:
      f.write(json.dumps(row) + "\n")
    print(f"    lin={row['lin_vel_err']:.4f} ang={row['ang_vel_err']:.4f} "
          f"h={row['height_err']:.4f} live={row['living_time']:.1f} "
          f"sym={row['symmetry_loss']}", flush=True)

  print("\ndone ->", args.out)
  return 0


if __name__ == "__main__":
  sys.exit(main())
