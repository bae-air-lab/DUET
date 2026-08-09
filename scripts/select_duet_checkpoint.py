"""Select the deployable DUET checkpoint by measured performance, not recency.

The last checkpoint of a long run is routinely not the best one: entropy drift,
curriculum saturation and late-run degradation all mean the useful policy is
usually somewhere in the middle. This script evaluates the checkpoint sweep
under the FIXED duet_eval protocol and ranks the results, so the deployed policy
is chosen by fall rate and tracking error rather than by filename.

Protocol (fixed, do not vary between checkpoints -- that is the whole point):
  1024 envs, 20 s episodes, arm curriculum pinned at full strength, flat
  terrain, 1.0 kg/hand payload, 0.7 m/s pushes.

Usage:
  python scripts/select_duet_checkpoint.py --run-dir logs/DUET_G1_23dof/<run>
  python scripts/select_duet_checkpoint.py --run-dir <dir> --from-iter 10000 --every 2500
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

DUET_EVAL = "/home/rbbist.lab/paper_rl/duet_bench/duet_eval.py"


def find_checkpoints(run_dir: str, from_iter: int, every: int) -> list[tuple[int, str]]:
  out = []
  for path in glob.glob(os.path.join(run_dir, "model_*.pt")):
    m = re.search(r"model_(\d+)\.pt$", path)
    if not m:
      continue
    it = int(m.group(1))
    if it >= from_iter and it % every == 0:
      out.append((it, path))
  return sorted(out)


def score(row: dict) -> float:
  """Rank by fall rate first, then tracking error.

  Falls dominate deliberately: a policy that tracks 3 cm/s better but falls
  twice as often is not the one to put on hardware. The tracking terms break
  ties among policies that are all effectively stable, and height error is
  included because pelvis height is a commanded quantity here, not a byproduct.
  """
  return (
    10.0 * row["falls_per_env_minute"]
    + 1.0 * row["mean_vxy_err_mps"]
    + 0.5 * row["mean_vyaw_err_radps"]
    + 1.0 * row["mean_height_err_m"]
  )


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--run-dir", required=True)
  ap.add_argument("--task", default="Unitree-G1-23Dof-Duet-Flat")
  ap.add_argument("--from-iter", type=int, default=10_000)
  ap.add_argument("--every", type=int, default=2_500)
  ap.add_argument("--num-envs", type=int, default=1024)
  ap.add_argument("--episode-s", type=float, default=20.0)
  ap.add_argument("--payload", type=float, default=1.0)
  ap.add_argument("--out", default="duet_checkpoint_selection.jsonl")
  args = ap.parse_args()

  ckpts = find_checkpoints(args.run_dir, args.from_iter, args.every)
  if not ckpts:
    print(f"No checkpoints >= {args.from_iter} (every {args.every}) in {args.run_dir}")
    return 1
  print(f"evaluating {len(ckpts)} checkpoints: {[it for it, _ in ckpts]}\n")

  rows: list[dict] = []
  for it, path in ckpts:
    print(f"--- iteration {it} ---")
    rc = subprocess.call([
      sys.executable, DUET_EVAL,
      "--checkpoint", path,
      "--task", args.task,
      "--tag", f"select_{it}",
      "--out", args.out,
      "--num-envs", str(args.num_envs),
      "--episode-s", str(args.episode_s),
      "--payload", str(args.payload),
    ])
    if rc != 0:
      print(f"  [WARN] eval failed for {path} (rc={rc}); excluded from selection")
      continue
    with open(args.out) as f:
      last = json.loads(f.readlines()[-1])
    last["iteration"] = it
    rows.append(last)

  if not rows:
    print("No successful evaluations.")
    return 1

  rows.sort(key=score)
  print("\n" + "=" * 92)
  print(f"{'iter':>7} {'falls/env-min':>14} {'vxy err':>9} {'vyaw err':>9} "
        f"{'h err':>8} {'living s':>9} {'score':>8}")
  print("-" * 92)
  for r in rows:
    print(f"{r['iteration']:>7} {r['falls_per_env_minute']:>14.4f} "
          f"{r['mean_vxy_err_mps']:>9.4f} {r['mean_vyaw_err_radps']:>9.4f} "
          f"{r['mean_height_err_m']:>8.4f} "
          f"{r.get('mean_living_time_s', float('nan')):>9.2f} {score(r):>8.4f}")
  best = rows[0]
  print("=" * 92)
  print(f"\nSELECTED: iteration {best['iteration']}\n  {best['checkpoint']}")
  print("\nExport and verify it with:")
  print(f"  python scripts/export_duet_onnx.py --task {args.task} \\\n"
        f"      --checkpoint {best['checkpoint']} --out-dir exported/duet_"
        f"{best['iteration']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
