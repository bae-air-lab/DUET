"""Rank checkpoints for hardware deployment under one fixed protocol.

Why this exists rather than scripts/select_duet_checkpoint.py: that script shells
out to an evaluator pinned to a sibling repository, which loads a different task
package and an older arm-disturbance generator. Checkpoints trained here must be
judged against the generator they were trained against, so the protocol lives in
this repo.

Protocol, identical for every checkpoint (that is the whole point):
  * the FULL trained task distribution, pinned -- arm trajectories at full
    workspace/velocity/acceleration, pushes at full magnitude, the whole height
    and velocity command range, observation corruption ON
  * ranked by FALL RATE first. A policy that tracks a few cm/s better but falls
    twice as often is not the one to put on a real robot. Tracking terms only
    break ties among policies that are already stable.

Usage:
  PYTHONPATH=. python scripts/duet_rank_checkpoints.py --auto
  PYTHONPATH=. python scripts/duet_rank_checkpoints.py --checkpoint a.pt b.pt
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends
from src.tasks.duet.config.g1_23dof.env_cfgs import pin_duet_full_distribution


def evaluate(ckpt, task, dev, n_env, steps, seed):
  torch.manual_seed(seed)
  cfg = load_env_cfg(task, play=False)
  ag = load_rl_cfg(task)
  cfg.scene.num_envs = n_env
  cfg.episode_length_s = 20.0
  pin_duet_full_distribution(cfg)          # full difficulty from step 0
  cfg.curriculum = {}                      # nothing ramps during the eval
  raw = ManagerBasedRlEnv(cfg=cfg, device=dev)
  env = RslRlVecEnvWrapper(raw, clip_actions=ag.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(ag), device=dev)
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
  pol = r.get_inference_policy(device=dev)

  u = env.unwrapped
  rb = u.scene["robot"]
  fid = u.reward_manager.get_term_cfg("track_base_height").params["asset_cfg"].site_ids
  tid = u.reward_manager.get_term_cfg("body_orientation_l2").params["asset_cfg"].body_ids
  dt = u.step_dt

  falls = torch.zeros(n_env, device=dev)
  acc = {k: 0.0 for k in ("vxy", "yaw", "h", "tilt", "jit")}
  cnt = 0.0
  prev = prev2 = None
  obs = env.get_observations()
  with torch.inference_mode():
    for _ in range(steps):
      a = pol(obs)
      obs, _, dones, _ = env.step(a)
      falls += u.termination_manager.get_term("fell_over").float()
      valid = (~dones.bool()).float()
      w = float(valid.sum())
      if w > 0:
        cmd = u.command_manager.get_command("twist")
        lin = rb.data.root_link_lin_vel_b
        ang = rb.data.root_link_ang_vel_b
        acc["vxy"] += float((torch.norm(cmd[:, :2] - lin[:, :2], dim=1) * valid).sum())
        acc["yaw"] += float(((cmd[:, 2] - ang[:, 2]).abs() * valid).sum())
        hc = u.command_manager.get_command("base_height")[:, 0]
        bh = rb.data.root_link_pos_w[:, 2] - rb.data.site_pos_w[:, fid, 2].min(1).values
        acc["h"] += float(((bh - hc + 0.02).abs() * valid).sum())
        g = quat_apply_inverse(rb.data.body_link_quat_w[:, tid, :].squeeze(1),
                               rb.data.gravity_vec_w)
        acc["tilt"] += float((torch.acos((-g[:, 2]).clamp(-1, 1)) * valid).sum())
        if prev2 is not None:
          acc["jit"] += float(((a - 2 * prev + prev2).abs().mean(1) * valid).sum())
        cnt += w
      prev2, prev = prev, a.clone()
  env_min = n_env * steps * dt / 60.0
  out = {
    "falls_per_env_min": float(falls.sum()) / env_min,
    "vxy_err": acc["vxy"] / max(cnt, 1),
    "yaw_err": acc["yaw"] / max(cnt, 1),
    "h_err": acc["h"] / max(cnt, 1),
    "tilt_deg": math.degrees(acc["tilt"] / max(cnt, 1)),
    "jitter": acc["jit"] / max(cnt, 1),
  }
  env.close()
  return out


def score(r) -> float:
  """Falls dominate; tracking breaks ties among stable policies."""
  return (10.0 * r["falls_per_env_min"] + 1.0 * r["vxy_err"]
          + 0.5 * r["yaw_err"] + 1.0 * r["h_err"])


def auto_candidates() -> list[str]:
  """One checkpoint per 2000 iterations, newest run winning on overlap."""
  best: dict[int, str] = {}
  for run in sorted(glob.glob("logs/rsl_rl/DUET_G1_23dof/*arm_robust*")):
    for p in glob.glob(os.path.join(run, "model_*.pt")):
      m = re.search(r"model_(\d+)\.pt$", p)
      if m and int(m.group(1)) > 0 and int(m.group(1)) % 2000 == 0:
        best[int(m.group(1))] = p
  return [best[k] for k in sorted(best)]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", nargs="*", default=None)
  ap.add_argument("--auto", action="store_true")
  ap.add_argument("--task", default="Unitree-G1-23Dof-Duet-Flat")
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=700)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default=None)
  a = ap.parse_args()
  configure_torch_backends()
  dev = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cks = a.checkpoint or (auto_candidates() if a.auto else [])
  if not cks:
    print("nothing to evaluate"); return 1

  rows = []
  for c in cks:
    it = int(re.search(r"model_(\d+)\.pt$", c).group(1))
    run = os.path.basename(os.path.dirname(c))[11:]
    r = evaluate(c, a.task, dev, a.num_envs, a.steps, a.seed)
    r.update(iter=it, run=run, path=c, score=0.0)
    r["score"] = score(r)
    rows.append(r)
    print(f"  [done] {run} iter {it}", flush=True)

  rows.sort(key=lambda r: r["score"])
  print(f"\nRANKED (full task distribution, {a.num_envs} envs x {a.steps} steps, "
        f"identical seed; lower score = better)")
  print(f"{'rank':>4}{'iter':>7}{'falls/min':>11}{'vxy_err':>9}{'yaw_err':>9}"
        f"{'h_err':>8}{'tilt':>7}{'jitter':>8}{'score':>8}  run")
  for i, r in enumerate(rows, 1):
    print(f"{i:4d}{r['iter']:7d}{r['falls_per_env_min']:11.3f}{r['vxy_err']:9.3f}"
          f"{r['yaw_err']:9.3f}{r['h_err']:8.4f}{r['tilt_deg']:7.2f}{r['jitter']:8.3f}"
          f"{r['score']:8.3f}  {r['run']}")
  b = rows[0]
  print(f"\nbest by this protocol: {b['path']}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
