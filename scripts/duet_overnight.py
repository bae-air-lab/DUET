"""Unattended overnight validation of DUET checkpoints as training produces them.

Watches a run directory and, for each new checkpoint, runs the full battery and
appends the result. Designed to survive the agent session ending: it is a plain
background process writing to disk, so the results are on disk in the morning
whether or not anyone is watching.

Battery per checkpoint (all headless, all measured -- nothing is assumed):

  1. duet_eval, flat, nominal payload   -- falls, tracking, height error, living time
  2. duet_eval, flat, 1.75 kg/hand      -- the same under max cargo
  3. squat probe                        -- achieved depth + torso pitch vs command
  4. idle probe, arms at default        -- standing drift/wobble
  5. idle probe, ARMS FORWARD           -- the same with the CoM shifted forward,
                                           reproducing the manual hardware test

Every checkpoint is compared against a fixed baseline checkpoint (the one
already validated on hardware), so "better" is always relative to something real.

Env counts are kept modest on purpose: this shares a GPU with the live training
run, and starving it would cost more than the extra statistical power is worth.

Usage:
  python scripts/duet_overnight.py --run-dir logs/rsl_rl/DUET_G1_23dof/<run> \
      --baseline keep/hw_best_7000/model_7000.pt --from-iter 9500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

REPO = "/home/rbbist.lab/unitree_rl_mjlab"
DUET_EVAL = "/home/rbbist.lab/paper_rl/duet_bench/duet_eval.py"
TASK = "Unitree-G1-23Dof-Duet-Flat"


def log(msg: str) -> None:
  print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run(cmd: list[str], timeout: int) -> str | None:
  """Run a sub-job, returning stdout or None. Never raises: one bad checkpoint
  must not end the night."""
  try:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                       cwd=REPO)
    if p.returncode != 0:
      log(f"  [warn] rc={p.returncode}: {' '.join(cmd[:3])}... "
          f"{p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else ''}")
      return None
    return p.stdout
  except subprocess.TimeoutExpired:
    log(f"  [warn] timeout: {' '.join(cmd[:3])}...")
    return None
  except Exception as e:  # noqa: BLE001
    log(f"  [warn] {type(e).__name__}: {e}")
    return None


def eval_one(ckpt: str, tag: str, payload: float, envs: int, steps: int,
             out: str) -> dict | None:
  run([sys.executable, DUET_EVAL, "--checkpoint", ckpt, "--task", TASK,
       "--tag", tag, "--out", out, "--num-envs", str(envs), "--steps", str(steps),
       "--payload", str(payload)], timeout=1800)
  try:
    with open(out) as f:
      last = json.loads(f.readlines()[-1])
    return last if last.get("tag") == tag else None
  except Exception:  # noqa: BLE001
    return None


# Must accept a leading '+': the probe prints signed values with "{:+.3f}", and
# a '-?'-only pattern silently failed to match, reporting squat depth as n/a.
_NUM = r"([+-]?\d+\.?\d*)"


def probe_squat(ckpt: str, envs: int) -> dict | None:
  txt = run([sys.executable, "scripts/duet_probe_squat.py", "--checkpoint", ckpt,
             "--envs-per-height", str(envs), "--settle", "300", "--measure", "120"],
            timeout=1800)
  if not txt:
    return None
  # "  model_x: achieved 0.254 m (short by +0.134), pitch -0.6 deg"
  m = re.search(r"achieved " + _NUM + r" m \(short by " + _NUM + r"\), pitch " + _NUM,
                txt)
  if not m:
    return None
  return {"deep_achieved_m": float(m.group(1)), "deep_short_m": float(m.group(2)),
          "deep_pitch_deg": float(m.group(3))}


def probe_idle(ckpt: str, arm_pose: str, envs: int) -> dict | None:
  txt = run([sys.executable, "scripts/duet_probe_idle.py", "--checkpoint", ckpt,
             "--heights", "0.73", "--per-height", str(envs),
             "--settle", "250", "--measure", "300", "--arm-pose", arm_pose],
            timeout=1800)
  if not txt:
    return None
  rows = [ln for ln in txt.splitlines() if ln.strip().startswith("0.73")]
  if not rows:
    return None
  v = rows[-1].split()
  try:
    return {f"idle_{arm_pose}_vxy": float(v[1]),
            f"idle_{arm_pose}_yaw": float(v[2]),
            f"idle_{arm_pose}_osc_mm": float(v[5]),
            f"idle_{arm_pose}_drift_mm": float(v[6])}
  except (IndexError, ValueError):
    return None


def battery(ckpt: str, tag: str, args) -> dict:
  res: dict = {"checkpoint": ckpt, "tag": tag, "time": datetime.now().isoformat()}
  log(f"  1/5 duet_eval nominal ({args.eval_envs} envs)")
  e = eval_one(ckpt, f"{tag}_nom", 1.0, args.eval_envs, args.eval_steps, args.eval_out)
  if e:
    res.update({k: e[k] for k in (
      "falls_per_env_minute", "mean_height_err_m", "mean_vxy_err_mps",
      "mean_vyaw_err_radps", "mean_tilt_deg", "idle_base_speed_mps",
      "mean_living_time_s") if k in e})
  log("  2/5 duet_eval max cargo (1.75 kg/hand)")
  e2 = eval_one(ckpt, f"{tag}_pay", 1.75, args.eval_envs, args.eval_steps,
                args.eval_out)
  if e2:
    res["falls_per_env_minute_payload"] = e2.get("falls_per_env_minute")
    res["mean_vxy_err_mps_payload"] = e2.get("mean_vxy_err_mps")
  log("  3/5 squat depth")
  res.update(probe_squat(ckpt, args.probe_envs) or {})
  log("  4/6 idle, arms default")
  res.update(probe_idle(ckpt, "default", args.probe_envs * 4) or {})
  log("  5/6 idle, arms forward (held)")
  res.update(probe_idle(ckpt, "forward", args.probe_envs * 4) or {})
  log("  6/6 idle, arms swept default<->forward")
  res.update(probe_idle(ckpt, "sweep", args.probe_envs * 4) or {})
  return res


def score(r: dict) -> float:
  """Rank candidates. Falls dominate; then tracking; then the standing-precision
  terms the operator actually cares about for manipulation."""
  def g(k, d=0.0):
    v = r.get(k)
    return d if v is None else float(v)
  return (
    10.0 * g("falls_per_env_minute")
    + 5.0 * g("falls_per_env_minute_payload")
    + 1.0 * g("mean_vxy_err_mps")
    + 0.5 * g("mean_vyaw_err_radps")
    + 1.0 * g("mean_height_err_m")
    # Standing precision, the property the operator actually needs for
    # manipulation. Weighted so ~50 mm of drift costs about as much as 0.05 m/s
    # of tracking error -- present in the ranking, not dominating it.
    + 0.01 * (g("idle_default_drift_mm") + g("idle_sweep_drift_mm"))
    + 0.01 * (g("idle_default_osc_mm") + g("idle_sweep_osc_mm"))
  )


def summarise(r: dict) -> str:
  def g(k, f="{:.4f}"):
    v = r.get(k)
    return "  n/a" if v is None else f.format(v)
  return (f"{r['tag']}: falls={g('falls_per_env_minute')} "
          f"falls_pay={g('falls_per_env_minute_payload')} "
          f"vxy={g('mean_vxy_err_mps')} vyaw={g('mean_vyaw_err_radps')} "
          f"herr={g('mean_height_err_m')} live={g('mean_living_time_s','{:.1f}')} "
          f"squat={g('deep_achieved_m','{:.3f}')} "
          f"drift[def/fwd/swp]={g('idle_default_drift_mm','{:.1f}')}/"
          f"{g('idle_forward_drift_mm','{:.1f}')}/"
          f"{g('idle_sweep_drift_mm','{:.1f}')}mm "
          f"osc_def={g('idle_default_osc_mm','{:.1f}')}mm "
          f"score={score(r):.4f}")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--baseline", default="keep/hw_best_7000/model_7000.pt")
  ap.add_argument("--from-iter", type=int, default=9500)
  ap.add_argument("--every", type=int, default=500)
  ap.add_argument("--eval-envs", type=int, default=512)
  # Must exceed the 1000-step (20 s) episode, or NO episode completes inside the
  # window and mean_living_time_s is reported as 0.0 for every checkpoint.
  ap.add_argument("--eval-steps", type=int, default=1400)
  ap.add_argument("--probe-envs", type=int, default=32)
  ap.add_argument("--out", default="logs/duet_runs/overnight_results.jsonl")
  ap.add_argument("--eval-out", default="logs/duet_runs/overnight_duet_eval.jsonl")
  ap.add_argument("--poll", type=float, default=180.0)
  ap.add_argument("--train-pid", type=int, default=0)
  args = ap.parse_args()

  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  done: set[str] = set()
  if os.path.exists(args.out):
    with open(args.out) as f:
      for line in f:
        try:
          done.add(json.loads(line)["tag"])
        except Exception:  # noqa: BLE001
          pass

  results: list[dict] = []
  log(f"overnight battery started; baseline={args.baseline}")

  # Baseline first, so every later comparison has something real to sit against.
  if "baseline" not in done and os.path.exists(args.baseline):
    log("BASELINE (hardware-validated checkpoint)")
    r = battery(args.baseline, "baseline", args)
    results.append(r)
    with open(args.out, "a") as f:
      f.write(json.dumps(r) + "\n")
    log("BASELINE " + summarise(r))

  while True:
    alive = args.train_pid and os.path.exists(f"/proc/{args.train_pid}")
    todo = []
    for p in sorted(glob.glob(os.path.join(args.run_dir, "model_*.pt"))):
      m = re.search(r"model_(\d+)\.pt$", p)
      if not m:
        continue
      it = int(m.group(1))
      tag = f"it{it}"
      if it >= args.from_iter and it % args.every == 0 and tag not in done:
        todo.append((it, p, tag))
    for it, path, tag in sorted(todo):
      log(f"--- checkpoint {it} ---")
      r = battery(path, tag, args)
      r["iteration"] = it
      results.append(r)
      done.add(tag)
      with open(args.out, "a") as f:
        f.write(json.dumps(r) + "\n")
      log("RESULT " + summarise(r))

    if not alive and not todo:
      log("training process gone and no new checkpoints; finishing")
      break
    time.sleep(args.poll)

  # Final ranking over everything measured this session.
  all_rows: list[dict] = []
  with open(args.out) as f:
    for line in f:
      try:
        all_rows.append(json.loads(line))
      except Exception:  # noqa: BLE001
        pass
  all_rows.sort(key=score)
  log("=" * 100)
  log("FINAL RANKING (best first)")
  for r in all_rows:
    log("  " + summarise(r))
  if all_rows:
    log(f"BEST: {all_rows[0]['tag']}  ->  {all_rows[0]['checkpoint']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
