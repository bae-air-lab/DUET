"""Build the paper's ablation table from completed ablation runs.

Methodology: every ablation is evaluated under the SAME fixed protocol -- the
reference task config -- with only the trained policy varying. Evaluating each
ablation under its own (ablated) config would change the test alongside the
treatment and measure nothing.

This is only valid because the ablations were designed to preserve the
observation and action layout: `NoHeightCmd` keeps the height observation slot
(pinned constant), `NoArmCurriculum` keeps the zero-width arm action term, and
`NoSymmetry`/`NoIdlePrecision` touch only the runner and the reward set. All
checkpoints are therefore 71 -> 13 and interchangeable at evaluation time.

Reports HOMIE's metric set (linear velocity error, angular velocity error,
height error, symmetry loss, living time) plus the fall rate and standing-drift
measures this project added.

Usage:
  python scripts/duet_ablation_table.py --runs-root logs/rsl_rl/DUET_G1_23dof_Ablation
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
PROTOCOL_TASK = "Unitree-G1-23Dof-Duet-Flat"  # fixed evaluation protocol


def latest_checkpoint(run_dir: str, min_iter: int = 0) -> tuple[str | None, int]:
  """Latest checkpoint in a run, and its iteration.

  ``min_iter`` guards against including a run that is still training. Without
  it, a run three minutes old contributes `model_0.pt` -- an untrained network
  -- and lands in the paper table as a catastrophic-looking ablation result
  (42 falls/min) that is really just an artefact of reading it too early. That
  is a silently wrong number rather than a visible failure, so it is worth an
  explicit check.
  """
  cks = glob.glob(os.path.join(run_dir, "model_*.pt"))
  if not cks:
    return None, -1
  best = max(cks, key=lambda p: int(re.search(r"model_(\d+)\.pt$", p).group(1)))
  it = int(re.search(r"model_(\d+)\.pt$", best).group(1))
  if it < min_iter:
    return None, it
  return best, it


def symmetry_loss(run_dir: str) -> float | None:
  """Mean Loss/symmetry over the last 10% of training, from tfevents.

  Logged in every variant including `NoSymmetry` -- rsl_rl computes it for
  logging even when it is excluded from the objective -- so the ablation that
  removes symmetry still reports the quantity it ablates.
  """
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
    vals = [s.value for s in ea.Scalars(tags[0])]
    tail = vals[max(0, int(len(vals) * 0.9)):]
    return sum(tail) / max(len(tail), 1)
  except Exception:  # noqa: BLE001
    return None


def run(cmd, timeout=2400):
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)
    return p.stdout if p.returncode == 0 else None
  except Exception:  # noqa: BLE001
    return None


def evaluate(ckpt: str, tag: str, out: str, envs: int, steps: int) -> dict | None:
  run([sys.executable, DUET_EVAL, "--checkpoint", ckpt, "--task", PROTOCOL_TASK,
       "--tag", tag, "--out", out, "--num-envs", str(envs), "--steps", str(steps),
       "--payload", "1.0"])
  try:
    with open(out) as f:
      last = json.loads(f.readlines()[-1])
    return last if last.get("tag") == tag else None
  except Exception:  # noqa: BLE001
    return None


def probe_idle(ckpt: str, envs: int) -> dict | None:
  txt = run([sys.executable, "scripts/duet_probe_idle.py", "--checkpoint", ckpt,
             "--heights", "0.73", "--per-height", str(envs),
             "--settle", "250", "--measure", "300"])
  if not txt:
    return None
  rows = [ln for ln in txt.splitlines() if ln.strip().startswith("0.73")]
  if not rows:
    return None
  v = rows[-1].split()
  try:
    return {"drift_mm": float(v[6]), "osc_mm": float(v[5]), "fwd_mm": float(v[9])}
  except (IndexError, ValueError):
    return None


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs-root", default="logs/rsl_rl/DUET_G1_23dof_Ablation")
  ap.add_argument("--extra-run", nargs="*", default=[],
                  help="Additional run dirs or checkpoints to include (e.g. the "
                       "main policy, for reference).")
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=1400)
  ap.add_argument("--probe-envs", type=int, default=128)
  ap.add_argument("--min-iter", type=int, default=6000,
                  help="Skip runs whose latest checkpoint is below this. "
                       "Guards against including a still-training run.")
  ap.add_argument("--out", default="logs/duet_runs/ablation_table.jsonl")
  args = ap.parse_args()

  runs = sorted(glob.glob(os.path.join(args.runs_root, "*")))
  runs = [r for r in runs if os.path.isdir(r)] + list(args.extra_run)
  if not runs:
    print(f"No ablation runs found under {args.runs_root}")
    return 1

  rows = []
  for run_dir in runs:
    if run_dir.endswith(".pt"):
      ckpt, name, sym = run_dir, os.path.basename(run_dir), None
    else:
      ckpt, it = latest_checkpoint(run_dir, args.min_iter)
      name = os.path.basename(run_dir)
      sym = symmetry_loss(run_dir)
      if not ckpt:
        print(f"[SKIP] {os.path.basename(run_dir)}: latest checkpoint is "
              f"iteration {it}, below --min-iter {args.min_iter} "
              f"(run still training or died early)")
        continue
    if not ckpt:
      print(f"[skip] no checkpoint in {run_dir}")
      continue
    tag = re.sub(r"^\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_?", "", name) or name
    print(f"--- {tag} :: {os.path.basename(ckpt)} ---", flush=True)
    r = {"variant": tag, "checkpoint": ckpt, "symmetry_loss": sym}
    e = evaluate(ckpt, f"abl_{tag}", args.out + ".raw", args.num_envs, args.steps)
    if e:
      r.update({k: e.get(k) for k in (
        "mean_vxy_err_mps", "mean_vyaw_err_radps", "mean_height_err_m",
        "mean_living_time_s", "falls_per_env_minute", "mean_tilt_deg")})
    r.update(probe_idle(ckpt, args.probe_envs) or {})
    rows.append(r)
    with open(args.out, "a") as f:
      f.write(json.dumps(r) + "\n")
    print(f"    {json.dumps({k: v for k, v in r.items() if k != 'checkpoint'})}",
          flush=True)

  def g(r, k, f="{:.4f}"):
    v = r.get(k)
    return "  n/a" if v is None else f.format(v)

  print("\n" + "=" * 118)
  print("ABLATION TABLE -- all evaluated under the reference protocol "
        f"({PROTOCOL_TASK}, {args.num_envs} envs, 20 s episodes)")
  print("=" * 118)
  print(f"{'variant':>24} {'lin err':>9} {'ang err':>9} {'h err':>8} "
        f"{'sym loss':>10} {'living s':>9} {'falls/min':>10} {'drift mm':>9}")
  print("-" * 118)
  for r in rows:
    print(f"{r['variant']:>24} {g(r,'mean_vxy_err_mps'):>9} "
          f"{g(r,'mean_vyaw_err_radps'):>9} {g(r,'mean_height_err_m'):>8} "
          f"{g(r,'symmetry_loss','{:.5f}'):>10} {g(r,'mean_living_time_s','{:.1f}'):>9} "
          f"{g(r,'falls_per_env_minute'):>10} {g(r,'drift_mm','{:.1f}'):>9}")
  print("=" * 118)
  print("\nlin/ang err: m/s, rad/s. h err: m. living time saturates at 20 s with no falls.")
  print("Symmetry loss is logged for every variant, including the one that ablates it.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
