"""Digital-twin acceptance scenarios for the arm-robustness DUET policy.

Runs a checkpoint headless through the eight scenarios of the arm-robustness
acceptance protocol and prints, per scenario, the quantities that define
"stable and approximately upright under arbitrary arm motion":

  torso pitch / roll (mean |.| and RMS, deg), root xy drift while idle (mean and
  max, mm), whole-body CoM support ratio (1 = edge of the double-support
  region), foot slip speed while in contact (mm/s), fraction of idle steps with
  a foot in the air (unnecessary stepping), velocity / yaw / height tracking
  error, and falls.

Everything is measured, nothing assumed. Arms are driven through the action
term's deployment hook (``set_external_target``) for the static / scripted
scenarios and through the training-time trapezoidal generator, pinned at the
full distribution, for the "moving arms" ones -- so the moving-arm scenarios
see exactly the disturbance distribution the policy was trained on.

Scenarios (``--scenario all`` runs every one):

  1 stationary       arms at the default pose; stand, walk fwd/back, strafe, turn, stop
  2 both_forward     standing; both arms extend forward and stay extended
  3 asymmetric       standing; left arm forward, right arm down/back
  4 moving_stand     standing; random trapezoidal arm trajectories
  5 moving_forward   walking forward 0.6 m/s; random arm trajectories
  6 moving_backward  walking backward -0.4 m/s, then stop; random arm trajectories
  7 moving_turn      turning +-0.6 rad/s; random arm trajectories
  8 moving_height    standing, height 0.73 -> 0.45 -> 0.73; random arm trajectories

Usage:
  python scripts/duet_arm_scenarios.py --checkpoint logs/.../model_XXXX.pt
  python scripts/duet_arm_scenarios.py --checkpoint ... --scenario moving_backward --num-envs 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from src.tasks.common.mdp.rewards import _com_support_ratio

# Static arm poses in the action term's joint order:
#   [L_sh_pitch, R_sh_pitch, L_sh_roll, R_sh_roll, L_sh_yaw, R_sh_yaw,
#    L_elbow, R_elbow, L_wrist, R_wrist]  (verified against arm.joint_names at run time)
_DEFAULT = {"pitch": 0.35, "roll": 0.18, "yaw": 0.0, "elbow": 0.87, "wrist": 0.0}
_FORWARD = {"pitch": -1.2, "roll": 0.10, "yaw": 0.0, "elbow": 0.30, "wrist": 0.0}
_DOWN_BACK = {"pitch": 0.8, "roll": 0.15, "yaw": 0.0, "elbow": 0.20, "wrist": 0.0}


def _pose_vec(names: list[str], left: dict, right: dict, device) -> torch.Tensor:
  vals = []
  for n in names:
    side = left if n.startswith("left_") else right
    if "shoulder_pitch" in n:
      v = side["pitch"]
    elif "shoulder_roll" in n:
      v = side["roll"] if n.startswith("left_") else -side["roll"]
    elif "shoulder_yaw" in n:
      v = side["yaw"] if n.startswith("left_") else -side["yaw"]
    elif "elbow" in n:
      v = side["elbow"]
    elif "wrist" in n:
      v = side["wrist"] if n.startswith("left_") else -side["wrist"]
    else:
      raise ValueError(n)
    vals.append(v)
  return torch.tensor(vals, device=device, dtype=torch.float32)


# (name, command schedule [(duration_s, vx, vy, yaw)], height schedule
#  [(duration_s, h)], arm mode: "default" | "forward" | "asymmetric" | "generator")
SCENARIOS = {
  "stationary": (
    [(3, 0, 0, 0), (4, 0.6, 0, 0), (2, 0, 0, 0), (4, -0.4, 0, 0), (2, 0, 0, 0),
     (3, 0, 0.3, 0), (2, 0, 0, 0), (3, 0, 0, 0.6), (3, 0, 0, -0.6), (3, 0, 0, 0)],
    [(1e9, 0.73)], "default",
  ),
  "both_forward": ([(12, 0, 0, 0)], [(1e9, 0.73)], "forward"),
  "asymmetric": ([(12, 0, 0, 0)], [(1e9, 0.73)], "asymmetric"),
  "moving_stand": ([(20, 0, 0, 0)], [(1e9, 0.73)], "generator"),
  "moving_forward": ([(2, 0, 0, 0), (10, 0.6, 0, 0), (3, 0, 0, 0)], [(1e9, 0.73)], "generator"),
  "moving_backward": ([(2, 0, 0, 0), (10, -0.4, 0, 0), (4, 0, 0, 0)], [(1e9, 0.73)], "generator"),
  "moving_turn": ([(2, 0, 0, 0), (5, 0, 0, 0.6), (5, 0, 0, -0.6), (3, 0, 0, 0)], [(1e9, 0.73)], "generator"),
  "moving_height": ([(20, 0, 0, 0)], [(5, 0.73), (7, 0.45), (8, 0.73)], "generator"),
}


_ROW_MARK = "@@SCENARIO_ROW@@ "


def _schedule_value(schedule, t):
  acc = 0.0
  for dur, *vals in schedule:
    acc += dur
    if t < acc:
      return vals
  return schedule[-1][1:]


def run_scenario(name, checkpoint, task, device, num_envs, settle_s):
  cmds, heights, arm_mode = SCENARIOS[name]
  total_s = sum(d for d, *_ in cmds) + settle_s

  cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  cfg.scene.num_envs = num_envs
  cfg.episode_length_s = int(1e9)
  cfg.commands["twist"].resampling_time_range = (1e9, 1e9)
  cfg.commands["base_height"].resampling_time_range = (1e9, 1e9)
  arm_cfg = cfg.actions["upper_body_pose"]
  if arm_mode == "generator":
    arm_cfg.init_ratio = 1.0  # full trained distribution
    arm_cfg.clean_fraction_stages = ((0, 0.0),)

  env_raw = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  u = env.unwrapped
  robot = u.scene["robot"]
  arm = u.action_manager.get_term("upper_body_pose")
  tw = u.command_manager.get_term("twist")
  bh = u.command_manager.get_term("base_height")
  torso_id = u.reward_manager.get_term_cfg("body_orientation_l2").params["asset_cfg"].body_ids
  foot_cfg = u.reward_manager.get_term_cfg("track_base_height").params["asset_cfg"]
  sensor = u.scene["feet_ground_contact"]
  fell_term = u.termination_manager

  names = arm.joint_names
  home = _pose_vec(names, _DEFAULT, _DEFAULT, device).expand(num_envs, -1)
  if arm_mode == "forward":
    goal = _pose_vec(names, _FORWARD, _FORWARD, device).expand(num_envs, -1)
  elif arm_mode == "asymmetric":
    goal = _pose_vec(names, _FORWARD, _DOWN_BACK, device).expand(num_envs, -1)
  else:
    goal = None
  ramp_steps = 75  # 1.5 s smooth move into the static pose after settling

  dt = u.step_dt
  n_steps = int(total_s / dt)
  settle_steps = int(settle_s / dt)
  from mjlab.utils.lab_api.math import quat_apply_inverse

  acc = {k: [] for k in ("pitch", "roll", "drift", "com", "slip", "air_idle",
                          "v_err", "yaw_err", "h_err", "idle")}
  falls = torch.zeros(num_envs, device=device)
  anchor = None
  obs = env.get_observations()
  with torch.inference_mode():
    for step in range(n_steps):
      t = max(0.0, (step - settle_steps) * dt)
      vx, vy, yaw = _schedule_value(cmds, t)
      (h,) = _schedule_value(heights, t)
      tw.vel_command_b[:, 0] = vx
      tw.vel_command_b[:, 1] = vy
      tw.vel_command_b[:, 2] = yaw
      if hasattr(tw, "is_standing_env"):
        tw.is_standing_env[:] = (vx == 0 and vy == 0 and yaw == 0)
      bh._squat_target[:] = h
      bh._walk_target[:] = h
      bh._height[:] = h
      if arm_mode in ("forward", "asymmetric"):
        k = step - settle_steps
        if k < 0:
          arm.set_external_target(home)
        else:
          a = 0.5 - 0.5 * math.cos(math.pi * min(1.0, k / ramp_steps))
          arm.set_external_target(home + a * (goal - home))
      elif arm_mode == "default":
        arm.set_external_target(home)

      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
      if step < settle_steps:
        continue

      falls += fell_term.get_term("fell_over").float()
      quat = robot.data.body_link_quat_w[:, torso_id, :].squeeze(1)
      g = quat_apply_inverse(quat, robot.data.gravity_vec_w)
      acc["pitch"].append(torch.atan2(g[:, 0], -g[:, 2]))
      acc["roll"].append(torch.atan2(g[:, 1], -g[:, 2]))
      idle = (abs(vx) + abs(vy) + abs(yaw)) < 1e-6
      pos = robot.data.root_link_pos_w[:, :2]
      if idle and anchor is None:
        anchor = pos.clone()
      if not idle:
        anchor = None
      acc["drift"].append(torch.norm(pos - anchor, dim=1) if anchor is not None else torch.full((num_envs,), float("nan"), device=device))
      r, ds = _com_support_ratio(u, "feet_ground_contact", foot_cfg, 0.09, 0.04)
      acc["com"].append(torch.where(ds, r, torch.full_like(r, float("nan"))))
      in_c = (sensor.data.found > 0).float()
      sp = torch.norm(robot.data.site_lin_vel_w[:, foot_cfg.site_ids, :2], dim=-1)
      acc["slip"].append((sp * in_c).sum(1) / in_c.sum(1).clamp(min=1))
      acc["air_idle"].append(((in_c.sum(1) < 2).float()) if idle else torch.full((num_envs,), float("nan"), device=device))
      lin = robot.data.root_link_lin_vel_b
      ang = robot.data.root_link_ang_vel_b
      acc["v_err"].append(torch.norm(tw.vel_command_b[:, :2] - lin[:, :2], dim=1))
      acc["yaw_err"].append((tw.vel_command_b[:, 2] - ang[:, 2]).abs())
      root_z = robot.data.root_link_pos_w[:, 2]
      foot_z = robot.data.site_pos_w[:, foot_cfg.site_ids, 2]
      acc["h_err"].append((root_z - foot_z.min(1).values - h + 0.02).abs())
      acc["idle"].append(torch.full((num_envs,), float(idle), device=device))

  T = {k: torch.stack(v) for k, v in acc.items()}
  def nanmean(x):
    return float(torch.nanmean(x))
  drift = T["drift"]
  out = {
    "scenario": name,
    "torso_pitch_mean_deg": math.degrees(nanmean(T["pitch"].abs())),
    "torso_pitch_rms_deg": math.degrees(math.sqrt(nanmean(T["pitch"] ** 2))),
    "torso_pitch_signed_mean_deg": math.degrees(nanmean(T["pitch"])),
    "torso_roll_rms_deg": math.degrees(math.sqrt(nanmean(T["roll"] ** 2))),
    "idle_drift_mean_mm": 1000 * nanmean(drift) if not torch.isnan(drift).all() else float("nan"),
    "idle_drift_max_mm": 1000 * float(torch.nan_to_num(drift, nan=0.0).max()) if not torch.isnan(drift).all() else float("nan"),
    "com_support_ratio_mean": nanmean(T["com"]),
    "com_support_ratio_p95": float(torch.nanquantile(T["com"].flatten()[::max(1, T["com"].numel() // 200000)], 0.95)),
    "foot_slip_mm_s": 1000 * nanmean(T["slip"]),
    "idle_foot_in_air_frac": nanmean(T["air_idle"]) if not torch.isnan(T["air_idle"]).all() else float("nan"),
    "vxy_err_mps": nanmean(T["v_err"]),
    "yaw_err_radps": nanmean(T["yaw_err"]),
    "height_err_m": nanmean(T["h_err"]),
    "falls_per_env": float(falls.mean()),
  }
  env.close()
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--task", default="Unitree-G1-23Dof-Duet-Flat")
  ap.add_argument("--scenario", default="all", help="all | " + " | ".join(SCENARIOS))
  ap.add_argument("--num-envs", type=int, default=128)
  ap.add_argument("--settle-s", type=float, default=3.0)
  ap.add_argument("--device", default=None)
  ap.add_argument("--json-row", action="store_true", help=argparse.SUPPRESS)
  a = ap.parse_args()
  configure_torch_backends()
  device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  names = list(SCENARIOS) if a.scenario == "all" else [a.scenario]
  if len(names) == 1:
    rows = [run_scenario(names[0], a.checkpoint, a.task, device, a.num_envs, a.settle_s)]
    if a.json_row:
      print(_ROW_MARK + json.dumps(rows[0]))
      return 0
  else:
    # One process per scenario. Measured on the rough-blind-tall tasks (which
    # carry the terrain-scan raycast sensor): building the sixth env + runner in
    # one process fails CUDA graph capture of the sensor graph ("Warp CUDA error
    # 901"), while -Flat runs all eight in one process and any single scenario
    # runs fine. Separate processes give every scenario a clean CUDA context.
    rows = []
    for n in names:
      cmd = [sys.executable, os.path.abspath(__file__), "--checkpoint", a.checkpoint,
             "--task", a.task, "--scenario", n, "--num-envs", str(a.num_envs),
             "--settle-s", str(a.settle_s), "--json-row"]
      if a.device:
        cmd += ["--device", a.device]
      out = subprocess.run(cmd, capture_output=True, text=True)
      row = [ln for ln in out.stdout.splitlines() if ln.startswith(_ROW_MARK)]
      if out.returncode != 0 or not row:
        sys.stderr.write(out.stdout[-4000:] + out.stderr[-4000:])
        raise RuntimeError(f"scenario {n} failed (rc={out.returncode})")
      rows.append(json.loads(row[-1][len(_ROW_MARK):]))
  keys = [k for k in rows[0] if k != "scenario"]
  print(f"\n{'scenario':>16} " + " ".join(f"{k[:14]:>14}" for k in keys))
  for r in rows:
    print(f"{r['scenario']:>16} " + " ".join(f"{r[k]:>14.4f}" for k in keys))
  print("\npitch signed: positive = torso leaning FORWARD (gravity x-component in torso frame).")
  print("com_support_ratio: 0 = CoM at support centre, 1 = edge of the double-support ellipse.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
