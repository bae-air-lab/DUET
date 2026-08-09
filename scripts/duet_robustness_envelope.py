"""Robustness envelope of the deployed policy, measured in simulation.

Sweeps hand payload against push magnitude under the fixed duet_eval protocol
and records fall rate, tracking error and height error at each cell. Produces
the data for the paper's envelope figure.

The sweep deliberately runs PAST the trained range (payload to 3.5 kg/hand
against a 1.75 kg training ceiling; pushes to 1.5 m/s against 0.7 m/s in
training) so the figure shows where the policy degrades, not only where it
holds. An envelope that stops at the training boundary tells a reviewer nothing
about the margin.

Usage:
  python scripts/duet_robustness_envelope.py --checkpoint <path.pt>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = "/home/rbbist.lab/unitree_rl_mjlab"
DUET_EVAL = "/home/rbbist.lab/paper_rl/duet_bench/duet_eval.py"
TASK = "Unitree-G1-23Dof-Duet-Flat"

# kg added per hand (training range was 0.25-1.75)
PAYLOADS = [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5]
# m/s planar push impulse (training used 0.7)
PUSHES = [0.0, 0.35, 0.7, 1.0, 1.5]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=1400)
  ap.add_argument("--out", default="logs/duet_runs/robustness_envelope.jsonl")
  ap.add_argument("--raw", default="logs/duet_runs/robustness_raw.jsonl")
  args = ap.parse_args()

  done = set()
  if os.path.exists(args.out):
    with open(args.out) as f:
      for line in f:
        try:
          r = json.loads(line)
          done.add((r["payload"], r["push"]))
        except Exception:  # noqa: BLE001
          pass

  total = len(PAYLOADS) * len(PUSHES)
  i = 0
  t0 = time.time()
  for pay in PAYLOADS:
    for push in PUSHES:
      i += 1
      if (pay, push) in done:
        print(f"[{i}/{total}] skip payload={pay} push={push} (already done)", flush=True)
        continue
      tag = f"env_p{pay}_k{push}"
      print(f"[{i}/{total}] payload={pay} kg/hand  push={push} m/s", flush=True)
      rc = subprocess.call([
        sys.executable, DUET_EVAL, "--checkpoint", args.checkpoint,
        "--task", TASK, "--tag", tag, "--out", args.raw,
        "--num-envs", str(args.num_envs), "--steps", str(args.steps),
        "--payload", str(pay), "--push", str(push),
      ], cwd=REPO)
      if rc != 0:
        print(f"    [warn] rc={rc}, skipping cell", flush=True)
        continue
      try:
        with open(args.raw) as f:
          last = json.loads(f.readlines()[-1])
      except Exception:  # noqa: BLE001
        continue
      row = {
        "payload": pay, "push": push,
        "falls_per_env_min": last.get("falls_per_env_minute"),
        "vxy_err": last.get("mean_vxy_err_mps"),
        "vyaw_err": last.get("mean_vyaw_err_radps"),
        "height_err": last.get("mean_height_err_m"),
        "living_s": last.get("mean_living_time_s"),
        "tilt_deg": last.get("mean_tilt_deg"),
      }
      with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
      print(f"    falls/min={row['falls_per_env_min']:.4f} "
            f"vxy={row['vxy_err']:.4f} living={row['living_s']:.1f}", flush=True)

  print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {args.out}", flush=True)

  # Text grid, so the result is readable without plotting.
  rows = [json.loads(l) for l in open(args.out)]
  grid = {(r["payload"], r["push"]): r for r in rows}
  print("\nFALLS PER ENV-MINUTE   (rows = payload kg/hand, cols = push m/s)")
  print("        " + "".join(f"{p:>10.2f}" for p in PUSHES))
  for pay in PAYLOADS:
    cells = "".join(
      f"{grid[(pay, p)]['falls_per_env_min']:>10.3f}" if (pay, p) in grid else f"{'-':>10}"
      for p in PUSHES)
    star = " *" if pay > 1.75 else ""
    print(f"{pay:>7.2f}{cells}{star}")
  print("  * beyond the 1.75 kg/hand training ceiling")
  return 0


if __name__ == "__main__":
  sys.exit(main())
