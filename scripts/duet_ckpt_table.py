"""Comparison table of DUET training checkpoints, read from tfevents.

Costs nothing: it reads the event files the runs already write, so it needs no
GPU and does not disturb training. Use it to compare checkpoints across the
chain of resumed runs (each resume starts a new run directory but continues the
same iteration counter, so the table stitches them back together and, where two
runs cover the same iteration, keeps the newest).

Usage:
  PYTHONPATH=. python scripts/duet_ckpt_table.py                 # every 1000 iters
  PYTHONPATH=. python scripts/duet_ckpt_table.py --every 500
  PYTHONPATH=. python scripts/duet_ckpt_table.py --from-iter 11000
  PYTHONPATH=. python scripts/duet_ckpt_table.py --only-saved    # only iters with a .pt

Columns, and what a good value looks like:
  return    episodic return. NOT comparable across a reward change (see NOTES).
  std       action std. Rising while return falls is the entropy-runaway signature.
  vloss     value loss. Rising means the critic cannot follow the task.
  h_err     height tracking error, m. Lower is better.
  jitter    mean |2nd difference of actions| -- what reads as visible twitch.
  yaw/lin   tracking errors, rad/s and m/s.
  pitch     |torso pitch|, rad.
  drift     stationary root drift, m.
  com       CoM offset from the support centre (1.0 = edge of the region).
  jlim      joint_pos_limits reward. Growing negative = pushing into the stops.
  falls     fell_over terminations.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EXPERIMENT = "logs/rsl_rl/DUET_G1_23dof"

TAGS = {
  "return": "Train/mean_reward",
  "std": "Policy/mean_std",
  "vloss": "Loss/value",
  "h_err": "Episode_Metrics/height_track_error",
  "jitter": "Episode_Metrics/mean_action_acc",
  "yaw": "Episode_Metrics/yaw_track_error",
  "lin": "Episode_Metrics/lin_vel_track_error",
  "pitch": "Episode_Metrics/torso_pitch_abs",
  "drift": "Metrics/idle_drift_m",
  "com": "Episode_Metrics/com_support_error",
  "jlim": "Episode_Reward/joint_pos_limits",
  "falls": "Episode_Termination/fell_over",
  "_eplen": "Train/mean_episode_length",
}
FMT = {"return": "{:7.1f}", "std": "{:6.3f}", "vloss": "{:6.3f}", "h_err": "{:7.4f}",
       "jitter": "{:7.3f}", "yaw": "{:6.3f}", "lin": "{:6.3f}", "pitch": "{:6.3f}",
       "drift": "{:6.3f}", "com": "{:6.3f}", "jlim": "{:6.3f}", "falls": "{:6.3f}"}

# What changed, and when, so the table is interpretable rather than just numbers.
NOTES = [
  (2000, "entropy 0.01->0.004 by 2500; push re-latches idle anchor; velocity stages blended"),
  (6000, "squat floor frozen at 0.27 m; HEIGHT_RANGE -> (0.27, 0.73)"),
  (11500, "added action_smoothness_l2 (-0.02), the 2nd-difference jitter penalty"),
  (16000, "pose std_standing 0.8/0.8/0.4 -> 1.6/1.6/0.8; squat floor -> 0.24 m"),
  (19000, "idle gates split linear/yaw (anchors position through a turn); vx range -> (-1.0, 1.0)"),
]


def collect() -> tuple[dict, dict]:
  """Merge every run's scalars by iteration; newer run wins on overlap.

  Exception: the FIRST iteration after a resume is a transient, not a
  measurement. Every environment resets at once, so mean_episode_length starts
  near zero and the episode-averaged metrics read nonsense (return ~1, value
  loss ~1). Those samples must not overwrite the real value another run logged
  at the same iteration, so anything with a short mean episode is only used
  when nothing better exists.
  """
  runs = sorted(d for d in glob.glob(f"{EXPERIMENT}/*") if os.path.isdir(d))
  data: dict[int, dict] = {}
  origin: dict[int, str] = {}
  solid: set[int] = set()  # iterations whose values came from a settled episode
  for run in runs:
    ev = sorted(glob.glob(f"{run}/events*"))
    if not ev:
      continue
    ea = EventAccumulator(ev[0], size_guidance={"scalars": 0})
    ea.Reload()
    have = ea.Tags()["scalars"]
    per: dict[int, dict] = {}
    for key, tag in TAGS.items():
      if tag not in have:
        continue
      for e in ea.Scalars(tag):
        per.setdefault(e.step, {})[key] = e.value
    for step, vals in per.items():
      settled = vals.get("_eplen", 0.0) >= 500.0
      if step in solid and not settled:
        continue  # do not let a resume transient clobber a real sample
      data[step] = {**data.get(step, {}), **vals} if not settled else vals
      origin[step] = os.path.basename(run)
      if settled:
        solid.add(step)
  return data, origin


def saved_iters() -> set[int]:
  out = set()
  for p in glob.glob(f"{EXPERIMENT}/*/model_*.pt"):
    m = re.search(r"model_(\d+)\.pt$", p)
    if m:
      out.add(int(m.group(1)))
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--every", type=int, default=1000)
  ap.add_argument("--from-iter", type=int, default=0)
  ap.add_argument("--only-saved", action="store_true",
                  help="restrict to iterations that have a checkpoint on disk")
  a = ap.parse_args()

  data, origin = collect()
  if not data:
    print("no event data found under", EXPERIMENT)
    return 1
  saved = saved_iters()
  steps = [s for s in sorted(data) if s >= a.from_iter and s % a.every == 0]
  if a.only_saved:
    steps = [s for s in steps if s in saved]
  latest = max(data)
  if latest not in steps:
    steps.append(latest)  # always show where the run is right now

  cols = [c for c in TAGS if not c.startswith("_")]
  print(f"{'iter':>6} " + " ".join(f"{c:>7}" for c in cols) + "  ckpt  run")
  prev = None
  for s in steps:
    row = []
    for c in cols:
      v = data[s].get(c)
      row.append(FMT[c].format(v) if v is not None else f"{'-':>7}")
    mark = "  yes" if s in saved else "   no"
    tag = "" if s != latest or s % a.every == 0 else "  <- live"
    print(f"{s:6d} " + " ".join(f"{x:>7}" for x in row) + f"{mark}  {origin[s][11:]}{tag}")
    for it, note in NOTES:
      if prev is not None and prev < it <= s:
        print(f"       {'':>7} ^^ config change at {it}: {note}")
    prev = s
  print("\nNOTES")
  print("  return and the pose reward are NOT comparable across a reward change;")
  print("  compare h_err, jitter, yaw, lin, pitch, drift, com and falls instead.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
