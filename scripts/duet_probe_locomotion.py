"""Measure locomotion command tracking: forward/backward symmetry and turn quality.

Two things the training curves cannot show:

1. SPEED SYMMETRY. `lin_vel_track_error` is an average over the whole command
   distribution, so a systematic forward/backward asymmetry averages away. This
   sweeps vx and reports achieved vs commanded on each side separately.

2. TURN QUALITY. `Metrics/idle_drift_m` only accumulates while the velocity
   command is ZERO, so it is blind to what happens during a turn. A commanded
   pure yaw (vx = vy = 0, wz != 0) should rotate the base in place; any residual
   base-frame translation is the robot orbiting instead of spinning. That
   residual, and the radius it implies, are what this reports.

Usage:
  PYTHONPATH=. python scripts/duet_probe_locomotion.py --checkpoint <path.pt>
  PYTHONPATH=. python scripts/duet_probe_locomotion.py --checkpoint <a.pt> <b.pt>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

VX = [-0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]
WZ = [-1.0, -0.6, -0.3, 0.3, 0.6, 1.0]


def build(ckpt, task, dev, n_env):
  cfg = load_env_cfg(task, play=True)
  ag = load_rl_cfg(task)
  cfg.scene.num_envs = n_env
  cfg.commands["base_height"].resampling_time_range = (1e9, 1e9)
  raw = ManagerBasedRlEnv(cfg=cfg, device=dev)
  env = RslRlVecEnvWrapper(raw, clip_actions=ag.clip_actions)
  r = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(ag), device=dev)
  r.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
  return env, r.get_inference_policy(device=dev)


def run(env, policy, cmds, per, settle, measure, height):
  """cmds: list of (vx, vy, wz). Returns achieved lin/ang velocity per command."""
  u = env.unwrapped
  rb = u.scene["robot"]
  tw = u.command_manager.get_term("twist")
  h_t = u.command_manager.get_term("base_height")
  dev = u.device
  n = len(cmds) * per
  want = torch.tensor([c for c in cmds for _ in range(per)], device=dev)
  acc = {k: torch.zeros(n, device=dev) for k in ("vx", "vy", "wz", "speed")}
  steps = 0
  p0 = None
  obs = env.get_observations()
  with torch.inference_mode():
    for s in range(settle + measure):
      # Pin the command every step. BOTH overrides are required: the twist
      # term's _update_command() zeroes standing envs, and -- easy to miss --
      # with heading_command=True it OVERWRITES the yaw channel with the output
      # of its own heading controller. Without disabling heading envs the yaw
      # command measured here is not the one being commanded, and the turn
      # results are meaningless.
      if hasattr(tw, "is_heading_env"):
        tw.is_heading_env[:] = False
      if hasattr(tw, "is_standing_env"):
        tw.is_standing_env[:] = False
      tw.vel_command_b[:, :] = want
      h_t._squat_target[:] = height
      h_t._walk_target[:] = height
      h_t._height[:] = height
      obs, _, _, _ = env.step(policy(obs))
      if s == settle:
        p0 = rb.data.root_link_pos_w[:, :2].clone()
      if s >= settle:
        lin = rb.data.root_link_lin_vel_b
        ang = rb.data.root_link_ang_vel_b
        acc["vx"] += lin[:, 0]
        acc["vy"] += lin[:, 1]
        acc["wz"] += ang[:, 2]
        acc["speed"] += torch.norm(lin[:, :2], dim=1)
        steps += 1
  disp = torch.norm(rb.data.root_link_pos_w[:, :2] - p0, dim=1)
  return {k: v / steps for k, v in acc.items()}, disp


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", nargs="+", required=True)
  ap.add_argument("--task", default="Unitree-G1-23Dof-Duet-Flat")
  ap.add_argument("--per-cmd", type=int, default=4)
  ap.add_argument("--settle", type=int, default=250)
  ap.add_argument("--measure", type=int, default=300)
  ap.add_argument("--height", type=float, default=0.70)
  ap.add_argument("--device", default=None)
  a = ap.parse_args()
  configure_torch_backends()
  dev = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  for ck in a.checkpoint:
    tag = ck.split("/")[-1]
    # -- forward / backward sweep -----------------------------------------
    cmds = [(v, 0.0, 0.0) for v in VX]
    env, pol = build(ck, a.task, dev, len(cmds) * a.per_cmd)
    res, _ = run(env, pol, cmds, a.per_cmd, a.settle, a.measure, a.height)
    print(f"\n=== {tag}: forward / backward tracking (height {a.height}) ===")
    print(f"{'cmd vx':>8}{'achieved':>10}{'err':>8}{'ratio':>8}")
    ach = {}
    for i, v in enumerate(VX):
      sl = slice(i * a.per_cmd, (i + 1) * a.per_cmd)
      m = float(res["vx"][sl].mean())
      ach[v] = m
      print(f"{v:8.2f}{m:10.3f}{m - v:+8.3f}{m / v:8.2f}")
    print("  symmetry (|achieved| fwd vs back at the same |cmd|):")
    for v in (0.2, 0.4, 0.6, 0.8):
      if v in ach and -v in ach:
        f, b = ach[v], abs(ach[-v])
        print(f"    |{v:.1f}|  fwd {f:.3f}  back {b:.3f}   back/fwd {b / f:.2f}")
    env.close()
    # -- turn in place ------------------------------------------------------
    cmds = [(0.0, 0.0, w) for w in WZ]
    env, pol = build(ck, a.task, dev, len(cmds) * a.per_cmd)
    res, disp = run(env, pol, cmds, a.per_cmd, a.settle, a.measure, a.height)
    dt = env.unwrapped.step_dt
    win = a.measure * dt
    print(f"\n=== {tag}: turn in place (vx = vy = 0) ===")
    print(f"{'cmd wz':>8}{'achieved':>10}{'err':>8}{'|v_xy|':>9}{'drift_m':>9}{'radius_m':>10}")
    for i, w in enumerate(WZ):
      sl = slice(i * a.per_cmd, (i + 1) * a.per_cmd)
      aw = float(res["wz"][sl].mean())
      sp = float(res["speed"][sl].mean())
      dr = float(disp[sl].mean())
      rad = sp / abs(aw) if abs(aw) > 1e-3 else float("nan")
      print(f"{w:8.2f}{aw:10.3f}{aw - w:+8.3f}{sp:9.3f}{dr:9.3f}{rad:10.3f}")
    print(f"  |v_xy| is residual base-frame translation while turning (0 = spins on the spot).")
    print(f"  radius = |v_xy| / |yaw rate|, the circle the base traces instead.")
    print(f"  drift_m is net displacement over the {win:.0f} s measurement window.")
    env.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
