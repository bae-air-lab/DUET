"""Measure achieved squat depth and torso pitch vs commanded height.

The aggregate `R_hgt` metric averages over the whole commanded height range, and
most commanded heights fall in the walking band [0.60, 0.73] -- so poor tracking
at the DEEP end is invisible in it. This probe holds the robot standing still,
pins the height command to a fixed value per env, and reports what the policy
actually does at each depth.

Reproduces the play-script conditions: arms held at their default pose (a fresh
process has common_step_counter = 0, so the arm curriculum ratio is 0), zero
twist command, no pushes, no observation corruption.

Usage:
  python scripts/duet_probe_squat.py --checkpoint <a.pt> [<b.pt> ...] \
      [--settle 400] [--measure 150]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.torch import configure_torch_backends

# Floor is 0.18 as of 2026-08-06 (0.12 was below the 0.144 m kinematic limit).
HEIGHTS = [0.18, 0.25, 0.30, 0.40, 0.50, 0.60, 0.65, 0.73]


def probe(checkpoint: str, task: str, device: str, settle: int, measure: int,
          envs_per_height: int) -> list[dict]:
  cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  n = len(HEIGHTS) * envs_per_height
  cfg.scene.num_envs = n
  # Never resample the height command -- we pin it ourselves.
  cfg.commands["base_height"].resampling_time_range = (1e9, 1e9)

  env_raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  u = env.unwrapped
  robot = u.scene["robot"]
  foot_ids = u.reward_manager.get_term_cfg("track_base_height").params[
    "asset_cfg"
  ].site_ids
  torso_id = u.reward_manager.get_term_cfg("body_orientation_l2").params[
    "asset_cfg"
  ].body_ids
  h_term = u.command_manager.get_term("base_height")
  tw_term = u.command_manager.get_term("twist")

  want = torch.tensor(
    [h for h in HEIGHTS for _ in range(envs_per_height)], device=device
  ).unsqueeze(1)

  acc_h = torch.zeros(n, device=device)
  acc_pitch = torch.zeros(n, device=device)
  cnt = 0

  obs = env.get_observations()
  with torch.inference_mode():
    for step in range(settle + measure):
      # Pin the command every step: standing still, at the target height.
      tw_term.vel_command_b[:] = 0.0
      if hasattr(tw_term, "is_standing_env"):
        tw_term.is_standing_env[:] = True
      h_term._squat_target[:] = want
      h_term._walk_target[:] = want
      h_term._height[:] = want

      actions = policy(obs)
      obs, _, _, _ = env.step(actions)

      if step >= settle:
        root_z = robot.data.root_link_pos_w[:, 2]
        foot_z = robot.data.site_pos_w[:, foot_ids, 2]
        acc_h += root_z - torch.min(foot_z, dim=1).values
        # Torso pitch: angle of the torso +z axis from world +z, signed so that
        # POSITIVE means leaning forward (+x).
        q = robot.data.body_link_quat_w[:, torso_id, :].squeeze(1)
        z_axis = matrix_from_quat(q)[:, :, 2]
        acc_pitch += torch.atan2(z_axis[:, 0], z_axis[:, 2])
        cnt += 1

  rows = []
  for i, h in enumerate(HEIGHTS):
    sl = slice(i * envs_per_height, (i + 1) * envs_per_height)
    rows.append({
      "cmd": h,
      "achieved": float((acc_h[sl] / cnt).mean()),
      "pitch_deg": math.degrees(float((acc_pitch[sl] / cnt).mean())),
    })
  env.close()
  return rows


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", nargs="+", required=True)
  ap.add_argument("--task", default="Unitree-G1-23Dof-Duet-Flat")
  ap.add_argument("--settle", type=int, default=400)
  ap.add_argument("--measure", type=int, default=150)
  ap.add_argument("--envs-per-height", type=int, default=8)
  ap.add_argument("--device", default=None)
  a = ap.parse_args()

  configure_torch_backends()
  device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  results = {}
  for ckpt in a.checkpoint:
    tag = os.path.basename(ckpt).replace(".pt", "")
    results[tag] = probe(ckpt, a.task, device, a.settle, a.measure,
                         a.envs_per_height)

  tags = list(results)
  print("\n" + "=" * (18 + 26 * len(tags)))
  hdr = f"{'cmd height':>11} |"
  for t in tags:
    hdr += f" {t + ' achieved':>13} {'pitch':>9} |"
  print(hdr)
  print("-" * (18 + 26 * len(tags)))
  for i, h in enumerate(HEIGHTS):
    line = f"{h:>11.2f} |"
    for t in tags:
      r = results[t][i]
      line += f" {r['achieved']:>13.3f} {r['pitch_deg']:>8.1f}d |"
    print(line)
  print("=" * (18 + 26 * len(tags)))
  print("\nachieved = pelvis height above the lower foot (m)")
  print("pitch    = torso lean, POSITIVE is forward (deg)")
  print(f"deep-end error (cmd {HEIGHTS[0]:.2f}):")
  for t in tags:
    r = results[t][0]
    print(f"  {t:>14}: achieved {r['achieved']:.3f} m "
          f"(short by {r['achieved'] - HEIGHTS[0]:+.3f}), "
          f"pitch {r['pitch_deg']:+.1f} deg")
  return 0


if __name__ == "__main__":
  sys.exit(main())
