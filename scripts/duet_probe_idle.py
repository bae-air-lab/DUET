"""Measure standing stillness ("wobble") under play-mode conditions.

The training-log `Episode_Reward/idle_base_motion` averages over all envs with
the arm disturbance running and commands resampling, so it does not describe
what you see when you stand a policy still in the viewer. This probe reproduces
those conditions exactly -- zero twist command, arms held at their default pose,
no pushes, no observation corruption -- and reports how much the base actually
moves.

Reports both a drift measure (net displacement) and an oscillation measure
(RMS about the mean), because "wobble" is the second, not the first: a policy
can hold its average position perfectly while oscillating around it visibly.

Usage:
  python scripts/duet_probe_idle.py --checkpoint <a.pt> <b.pt> [--heights 0.73 0.50]
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
from mjlab.utils.torch import configure_torch_backends


# Arm poses to hold while measuring. "forward" is the deployment reach pose and
# is the condition that matters most: extending the arms shifts the CoM forward,
# which is the disturbance the standing policy has to reject and the one the
# operator reported amplifies drift several-fold.
ARM_POSES = {
  "default": None,   # arms hang at their default pose (the plain play condition)
  "forward": 2,      # forward reach (table pick), held statically
  "lowpick": 3,      # low pick, pairs with a deep squat
  "carry": 0,        # box carry at waist
  # Arms swept default <-> forward on a ~3 s cycle. This is the condition that
  # reproduces the manual test: an operator "putting the arm forward" applies a
  # MOVING mass, which injects momentum the legs must reject. Holding the same
  # pose statically turns out to be calmer than the default hang, so the static
  # condition alone does not reproduce what is seen on hardware.
  "sweep": 2,
}
_SWEEP_PERIOD_STEPS = 150  # ~3 s at 50 Hz


def _arm_targets(env, name: str):
  """Resolve (default_pose, task_pose) joint targets for the driven arm joints.

  Uses the action term's deployment hook (``set_external_target``), the same
  path the VLA drives at runtime, so the arms follow a genuine task pose instead
  of the curriculum's randomised motion.
  """
  idx = ARM_POSES[name]
  if idx is None:
    return None, None
  import torch as _t

  from mjlab.utils.lab_api.string import resolve_matching_names_values

  from src.tasks.duet.config.g1_23dof.env_cfgs import TASK_ARM_POSES

  u = env.unwrapped
  arm = u.action_manager.get_term("upper_body_pose")
  _, _, values = resolve_matching_names_values(
    data=dict(TASK_ARM_POSES[idx]), list_of_strings=arm._joint_names
  )
  goal = _t.tensor(values, device=u.device, dtype=_t.float32)
  goal = goal.unsqueeze(0).repeat(u.num_envs, 1)
  return arm._default.clone(), goal


def _apply_arm_pose(env, name: str, home, goal, step: int) -> None:
  """Write the arm target for this step (static hold, or the sweep cycle)."""
  if goal is None:
    return
  import math as _m

  arm = env.unwrapped.action_manager.get_term("upper_body_pose")
  if name != "sweep":
    arm.set_external_target(goal)
    return
  # Triangle-free smooth cycle: 0 -> 1 -> 0, so the arms accelerate and
  # decelerate rather than snapping (a snap would be an unrealistic impulse).
  phase = (step % _SWEEP_PERIOD_STEPS) / _SWEEP_PERIOD_STEPS
  alpha = 0.5 - 0.5 * _m.cos(2.0 * _m.pi * phase)
  arm.set_external_target(home + alpha * (goal - home))


def probe(checkpoint, task, device, heights, per_h, settle, measure,
          arm_pose="default"):
  cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  n = len(heights) * per_h
  cfg.scene.num_envs = n
  cfg.commands["base_height"].resampling_time_range = (1e9, 1e9)

  env_raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  u = env.unwrapped
  robot = u.scene["robot"]
  h_term = u.command_manager.get_term("base_height")
  tw_term = u.command_manager.get_term("twist")
  want = torch.tensor([h for h in heights for _ in range(per_h)],
                      device=device).unsqueeze(1)

  home, goal = _arm_targets(env, arm_pose)

  traj = {k: [] for k in ("vxy", "yaw", "angxy", "z", "px", "py", "jv", "act")}
  obs = env.get_observations()
  prev_act = None
  with torch.inference_mode():
    for step in range(settle + measure):
      _apply_arm_pose(env, arm_pose, home, goal, step)
      tw_term.vel_command_b[:] = 0.0
      if hasattr(tw_term, "is_standing_env"):
        tw_term.is_standing_env[:] = True
      h_term._squat_target[:] = want
      h_term._walk_target[:] = want
      h_term._height[:] = want

      actions = policy(obs)
      obs, _, _, _ = env.step(actions)

      if step >= settle:
        lin = robot.data.root_link_lin_vel_b
        ang = robot.data.root_link_ang_vel_b
        traj["vxy"].append(torch.norm(lin[:, :2], dim=1))
        traj["yaw"].append(ang[:, 2].abs())
        traj["angxy"].append(torch.norm(ang[:, :2], dim=1))
        traj["z"].append(robot.data.root_link_pos_w[:, 2])
        traj["px"].append(robot.data.root_link_pos_w[:, 0])
        traj["py"].append(robot.data.root_link_pos_w[:, 1])
        traj["jv"].append(torch.norm(robot.data.joint_vel[:, :13], dim=1))
        if prev_act is not None:
          traj["act"].append(torch.norm(actions - prev_act, dim=1))
        prev_act = actions.clone()

  T = {k: torch.stack(v) for k, v in traj.items() if v}  # [T, N]
  out = []
  for i, h in enumerate(heights):
    sl = slice(i * per_h, (i + 1) * per_h)
    px, py, z = T["px"][:, sl], T["py"][:, sl], T["z"][:, sl]
    out.append({
      "cmd": h,
      "vxy": float(T["vxy"][:, sl].mean()),
      "yaw": float(T["yaw"][:, sl].mean()),
      "angxy": float(T["angxy"][:, sl].mean()),
      # Oscillation about each env's own mean -- this is "wobble".
      "z_osc_mm": float((z - z.mean(0)).std(0).mean()) * 1000,
      "xy_osc_mm": float(
        ((px - px.mean(0)).std(0) + (py - py.mean(0)).std(0)).mean()
      ) * 500,
      # Net drift over the window.
      "drift_mm": float(
        torch.sqrt((px[-1] - px[0]) ** 2 + (py[-1] - py[0]) ** 2).mean()
      ) * 1000,
      # SIGNED forward drift, and the fraction of envs sharing that sign.
      # Magnitude alone cannot distinguish random wander from a systematic
      # creep: a policy drifting 15 mm forward in 73% of envs and one wandering
      # 15 mm in random directions score identically on `drift_mm`, but only the
      # first is a bias worth fixing. This was found on hardware, not here,
      # because the probe had no directional channel at all.
      "fwd_mm": float((px[-1] - px[0]).mean()) * 1000,
      "fwd_frac": float(((px[-1] - px[0]) > 0).float().mean()),
      "jv": float(T["jv"][:, sl].mean()),
      "dact": float(T["act"][:, sl].mean()) if "act" in T else float("nan"),
    })
  env.close()
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", nargs="+", required=True)
  ap.add_argument("--task", default="Unitree-G1-23Dof-Duet-Flat")
  ap.add_argument("--heights", nargs="+", type=float, default=[0.73, 0.60, 0.45])
  ap.add_argument("--per-height", type=int, default=8)
  ap.add_argument("--settle", type=int, default=300)
  ap.add_argument("--measure", type=int, default=250)
  ap.add_argument("--device", default=None)
  ap.add_argument("--arm-pose", default="default", choices=sorted(ARM_POSES),
                  help="Arm pose held for the whole measurement.")
  a = ap.parse_args()

  configure_torch_backends()
  device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  res = {}
  for c in a.checkpoint:
    tag = os.path.basename(c).replace(".pt", "") + f"[{a.arm_pose}]"
    res[tag] = probe(c, a.task, device, a.heights, a.per_height, a.settle,
                     a.measure, arm_pose=a.arm_pose)

  cols = [("vxy", "base|v| m/s"), ("yaw", "|yawrate|"), ("angxy", "torso|w|xy"),
          ("z_osc_mm", "z osc mm"), ("xy_osc_mm", "xy osc mm"),
          ("drift_mm", "drift mm"), ("jv", "|qdot| leg"), ("dact", "d(act)"),
          ("fwd_mm", "fwd mm"), ("fwd_frac", "fwd frac")]
  for tag, rows in res.items():
    print(f"\n=== {tag} ===")
    print(f"{'cmd h':>6} " + " ".join(f"{lbl:>12}" for _, lbl in cols))
    for r in rows:
      print(f"{r['cmd']:>6.2f} " + " ".join(f"{r[k]:>12.4f}" for k, _ in cols))
  print("\nosc = std about each env's own mean (the visible wobble)")
  print("drift = net displacement over the measurement window")
  return 0


if __name__ == "__main__":
  sys.exit(main())
