"""Architecture comparison: decoupled DUET vs a whole-body velocity policy.

This answers a question the ablations do not. The ablations remove one DUET
component at a time and are matched-budget, so they establish what each
contribution is worth. This instead compares two *different architectures*:

  DUET        13 actions, legs + waist only; the arms are an external
              disturbance the legs must reject (and at deployment, a VLA's
              output the policy has no control over).
  whole-body  23 actions, the policy commands the arms itself.

Only metrics that mean the same thing for both are reported: velocity tracking
error, fall rate, living time, and standing drift. Height error and
arm-disturbance rejection are undefined for a policy with no height command and
no external arms, and are deliberately omitted rather than faked.

Conditions are matched physically -- same commands, pushes, friction and hand
payload -- by injecting the payload event into the whole-body task, which never
had one. That is the point of the payload condition: the real robot always
carries Dex3 hands, so "was never trained for payload" is a property of the
baseline worth measuring, not an unfairness to correct.

Usage:
  python scripts/duet_vs_wholebody.py \
      --duet keep/final_20260807_044803/model_15500.pt \
      --wholebody <path>/model_7000.pt --payload 0.0 1.75
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

DUET_TASK = "Unitree-G1-23Dof-Duet-Flat"
WB_TASK = "Unitree-G1-23Dof-Flat"
# Command range both policies cover: DUET trained to 1.2 m/s, the whole-body
# baseline to 2.0, so 1.2 is inside both and neither is extrapolating.
RANGES = {"lin_vel_x": (-0.8, 1.2), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)}


def build(task: str, payload: float, envs: int, episode_s: float,
          locomotion_only: bool = False):
  """Build a matched evaluation config.

  ``locomotion_only`` pins DUET's height command at standing and freezes its
  arms, so both policies are doing the *same* job. Without it DUET is squatting
  through the commanded height range and rejecting a full-amplitude arm
  disturbance while the whole-body baseline just walks -- which makes DUET look
  worse on tracking and far worse on torso tilt for reasons that are task
  differences, not architecture differences.
  """
  cfg = load_env_cfg(task, play=False)
  agent = load_rl_cfg(task)
  cfg.scene.num_envs = envs
  cfg.episode_length_s = episode_s
  cfg.curriculum = {}

  tw = cfg.commands["twist"]
  tw.rel_standing_envs = 0.30
  tw.ranges.lin_vel_x = RANGES["lin_vel_x"]
  tw.ranges.lin_vel_y = RANGES["lin_vel_y"]
  tw.ranges.ang_vel_z = RANGES["ang_vel_z"]

  # DUET-only: either pin the arm disturbance at full strength (deployment
  # condition) or freeze it and stand tall (locomotion-only isolation).
  if "upper_body_pose" in cfg.actions:
    cfg.actions["upper_body_pose"].enabled = not locomotion_only
    cfg.actions["upper_body_pose"].init_ratio = 0.0 if locomotion_only else 1.0
  if "base_height" in cfg.commands:
    bh = cfg.commands["base_height"]
    if locomotion_only:
      bh.enabled = False  # pinned at height_range[1], i.e. standing
    else:
      bh.floor_curriculum_start = bh.height_range[0]

  # Match the hand payload on both. The whole-body baseline never had this
  # event; adding it is what makes the physical condition identical.
  hand = SceneEntityCfg("robot", body_names=(r".*_wrist_roll_rubber_hand",))
  cfg.events["hand_payload"] = EventTermCfg(
    mode="reset", func=dr.body_mass,
    params={"asset_cfg": hand, "operation": "add", "ranges": (payload, payload)},
  )
  cfg.events.pop("torso_payload", None)
  cfg.events.pop("hand_com", None)
  return cfg, agent


def evaluate(task, ckpt, payload, envs, steps, episode_s, device, loco_only=False):
  cfg, agent = build(task, payload, envs, episode_s, locomotion_only=loco_only)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device),
                           clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(env, asdict(agent),
                                                          device=device)
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  u = env.unwrapped
  robot = u.scene["robot"]
  n = envs
  falls = torch.zeros(n, device=device)
  alive = torch.zeros(n, device=device)
  ep_steps, ep_count = 0.0, 0.0
  acc = {k: 0.0 for k in ("vxy", "vyaw", "tilt", "idle_v")}
  tot, idle_n = 0.0, 0.0

  obs = env.get_observations()
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, dones, _ = env.step(policy(obs))
      falls += u.termination_manager.get_term("fell_over").float()
      alive += 1.0
      dm = dones.bool()
      if dm.any():
        ep_steps += float(alive[dm].sum()); ep_count += float(dm.sum()); alive[dm] = 0.0
      twist = u.command_manager.get_command("twist")
      lin, ang = robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b
      valid = (~dm).float()
      w = valid.sum().item()
      if w > 0:
        acc["vxy"] += float((torch.norm(twist[:, :2] - lin[:, :2], dim=1) * valid).sum())
        acc["vyaw"] += float(((twist[:, 2] - ang[:, 2]).abs() * valid).sum())
        pg = robot.data.projected_gravity_b
        acc["tilt"] += float((torch.acos(torch.clamp(-pg[:, 2], -1, 1)) * valid).sum())
        tot += w
      idle = ((torch.norm(twist[:, :2], dim=1) + twist[:, 2].abs()) <= 0.1).float() * valid
      if idle.sum() > 0:
        acc["idle_v"] += float((torch.norm(lin[:, :2], dim=1) * idle).sum())
        idle_n += float(idle.sum())

  dt = u.step_dt
  env_s = float(alive.sum()) * dt + ep_steps * dt
  res = {
    "falls_per_env_min": float(falls.sum()) / max(env_s / 60.0, 1e-9),
    "vxy_err": acc["vxy"] / max(tot, 1),
    "vyaw_err": acc["vyaw"] / max(tot, 1),
    "tilt_deg": math.degrees(acc["tilt"] / max(tot, 1)),
    "idle_speed": acc["idle_v"] / max(idle_n, 1),
    "living_s": (ep_steps * dt) / max(ep_count, 1.0),
  }
  env.close()
  return res


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--duet", required=True)
  ap.add_argument("--wholebody", required=True)
  ap.add_argument("--payload", nargs="+", type=float, default=[0.0, 1.75])
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--steps", type=int, default=1400)
  ap.add_argument("--episode-s", type=float, default=20.0)
  ap.add_argument("--locomotion-only", action="store_true",
                  help="Pin DUET height at standing and freeze its arms, so both "
                       "policies do the same job (isolates architecture).")
  a = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  rows = []
  for pay in a.payload:
    for name, task, ck in (("DUET(13a)", DUET_TASK, a.duet),
                           ("WholeBody(23a)", WB_TASK, a.wholebody)):
      r = evaluate(task, ck, pay, a.num_envs, a.steps, a.episode_s, device,
                   loco_only=a.locomotion_only)
      r["policy"], r["payload"] = name, pay
      rows.append(r)
      print(f"DONE {name} payload={pay}", flush=True)

  print("\n" + "=" * 104)
  print(f"{'payload':>8} {'policy':>16} {'falls/env-min':>14} {'vxy err':>9} "
        f"{'vyaw err':>9} {'tilt deg':>9} {'idle m/s':>9} {'living s':>9}")
  print("-" * 104)
  for r in rows:
    print(f"{r['payload']:>8.2f} {r['policy']:>16} {r['falls_per_env_min']:>14.4f} "
          f"{r['vxy_err']:>9.4f} {r['vyaw_err']:>9.4f} {r['tilt_deg']:>9.2f} "
          f"{r['idle_speed']:>9.4f} {r['living_s']:>9.2f}")
  print("=" * 104)
  print("\nMatched: command range, pushes, friction, hand payload, episode length.")
  print("NOT comparable and omitted: height error (whole-body has no height")
  print("command) and arm-disturbance rejection (whole-body commands its arms).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
