"""Watch a DUET training log and emit one compact health line per milestone.

Used as the command for a Monitor: every stdout line becomes a notification, so
this prints only at milestone iterations plus on any failure signature. Reads
the tail of the log rather than the whole file, which grows to ~100 MB over a
25k-iteration run.

Coverage note: this must emit on death as well as on progress. A watcher that
only reports good news is indistinguishable from a crashed run, so the process
liveness check and the traceback scan are as important as the milestones.

Usage: duet_watch.py <log> <pid> [--first-phase-iters N] [--early N] [--late N]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
FAIL_RE = re.compile(
  r"Traceback|CUDA out of memory|RuntimeError|AssertionError|Killed|"
  r"out of memory|NaN detected|nan detected",
  re.IGNORECASE,
)

# Fields pulled from the latest completed block. Label -> log line prefix.
FIELDS = [
  ("rew", "Mean reward:"),
  ("eplen", "Mean episode length:"),
  ("fall", "Episode_Termination/fell_over:"),
  ("ent", "Mean entropy loss:"),
  ("entc", "Mean entropy_coef loss:"),
  ("sym", "Mean symmetry loss:"),
  ("vxy", "Metrics/twist/error_vel_xy:"),
  ("vyaw", "Metrics/twist/error_vel_yaw:"),
  ("std", "Mean action std:"),
  ("walkmin", "Curriculum/command_mix:"),
  ("R_lin", "Episode_Reward/track_linear_velocity:"),
  ("R_hgt", "Episode_Reward/track_base_height:"),
  ("R_term", "Episode_Reward/is_terminated:"),
  ("R_arate", "Episode_Reward/action_rate_l2:"),
  # Standing stillness. Added after a user-reported wobble at standing height
  # that none of the columns above could have surfaced: return rose while idle
  # behaviour degraded, because the aggregate hides a regression confined to one
  # commanded height. R_idle is the direct training-time measure of it.
  ("R_idle", "Episode_Reward/idle_base_motion:"),
  ("R_bangv", "Episode_Reward/body_ang_vel:"),
  ("R_slip", "Episode_Reward/foot_slip:"),
]


def tail(path: str, nbytes: int = 400_000) -> str:
  size = os.path.getsize(path)
  with open(path, "rb") as f:
    f.seek(max(0, size - nbytes))
    return ANSI.sub("", f.read().decode("utf-8", "replace"))


def parse(text: str) -> tuple[int | None, int | None, dict[str, str]]:
  its = ITER_RE.findall(text)
  if not its:
    return None, None, {}
  cur, total = int(its[-1][0]), int(its[-1][1])
  # Read the block belonging to the LAST iteration header, so values are never
  # mixed across iterations.
  block = text[text.rfind(f"Learning iteration {cur}/") :]
  vals: dict[str, str] = {}
  for label, prefix in FIELDS:
    for line in block.splitlines():
      s = line.strip()
      if s.startswith(prefix):
        vals[label] = s[len(prefix) :].strip()
        break
  return cur, total, vals


def next_milestone(it: int, first_phase: int, early: int, late: int) -> int:
  step = early if it < first_phase else late
  return ((it // step) + 1) * step


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("log")
  ap.add_argument("pid", type=int)
  ap.add_argument("--first-phase-iters", type=int, default=1000)
  ap.add_argument("--early", type=int, default=250)
  ap.add_argument("--late", type=int, default=500)
  ap.add_argument("--poll", type=float, default=45.0)
  a = ap.parse_args()

  reported = -1
  last_iter = -1
  stall_polls = 0
  seen_fail = set()

  while True:
    alive = os.path.exists(f"/proc/{a.pid}")
    try:
      text = tail(a.log)
    except OSError as e:
      print(f"[WATCH] cannot read log: {e}", flush=True)
      return 1

    for m in FAIL_RE.finditer(text):
      ctx = text[max(0, m.start() - 120) : m.start() + 400].strip().splitlines()
      key = m.group(0)[:40]
      if key not in seen_fail:
        seen_fail.add(key)
        print(f"[FAIL] {' | '.join(ctx[-4:])[:600]}", flush=True)

    cur, total, v = parse(text)
    if cur is not None:
      if cur == last_iter:
        stall_polls += 1
        # ~5 min with no new iteration while the process is still up.
        if stall_polls == max(1, int(300 / a.poll)):
          print(f"[STALL] no progress past iteration {cur} for ~5 min", flush=True)
      else:
        stall_polls = 0
        last_iter = cur

      target = next_milestone(reported if reported >= 0 else 0,
                              a.first_phase_iters, a.early, a.late)
      if cur >= target:
        reported = cur
        pct = 100.0 * cur / total if total else 0.0
        body = " ".join(f"{k}={v[k]}" for k, _ in FIELDS if k in v)
        print(f"[{cur}/{total} {pct:.1f}%] {body}", flush=True)

    if not alive:
      print(f"[EXIT] training process {a.pid} is gone (last iteration {last_iter})",
            flush=True)
      return 0
    if cur is not None and total is not None and cur >= total - 1:
      print(f"[DONE] reached iteration {cur}/{total}", flush=True)
      return 0
    time.sleep(a.poll)


if __name__ == "__main__":
  sys.exit(main())
