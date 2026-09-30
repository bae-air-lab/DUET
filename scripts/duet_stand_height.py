"""Measure the robot's natural standing height in the DUET height convention.

The height command, ``track_base_height`` and the ``height_track_error`` metric
all use

    h = root_z - min(foot_site_z) + 0.02        (ankle_sole_distance = 0.02)

i.e. the pelvis origin above the LOWER foot site, plus 2 cm. This script
reports that ``h`` two ways:

Mode 1 (sim, default). Builds ``Unitree-G1-23Dof-Duet-Flat`` with 1 env, strips
every randomisation event (encoder bias, CoM offset, payloads, friction,
pushes) so the result is the nominal robot, holds the default pose under PD
with zero action on the rigid plane for ``--seconds`` (default 10 s) and prints
``h``. It also
prints ``z_rest``: the foot site's height above rigid flat ground at rest, which
the terrain-relative foot-clearance reward needs (``feet_clearance``,
``reference="stance_foot"``). This is how ``height_max`` of the rough-blind-tall
tasks was chosen (rounded down to 0.01 m).

The pelvis is held LEVEL and fixed in x/y after every physics substep; its
height and vertical velocity stay free. Measured: with zero action the default
pose is not statically stable on the soft first-principles gains (ankle kp
28.5 Nm/rad per side against a gravity toppling stiffness of roughly
m*g*l ~ 200 Nm/rad), so the unconstrained robot pitches forward and trips
``fell_over`` about 1.2 s after every reset, and a 3 s average is an average
over falls. Holding only the tipping mode keeps the quantity being measured --
the vertical sag of the default pose under PD and gravity -- exact: at rest the
hold applies no vertical force, and it is the same "pelvis level, feet flat"
assumption Mode 2 makes. ``--free`` runs the unconstrained version for
comparison.

Mode 2 (hardware pose). ``--joints`` takes the 12 leg angles (rad) in SDK /
deploy.yaml order -- left hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch,
ankle_roll, then the same for the right leg -- as recorded from ``rt/lowstate``
while the stock firmware stands, and computes the same ``h`` by forward
kinematics with the pelvis level (identity orientation). It also prints each
foot's tilt from horizontal, which should be near zero for a pose that stands
flat-footed; a large tilt means the pelvis-level assumption does not hold for
that pose. The arms are left at the default pose (they do not enter ``h``).

Usage:
  PYTHONPATH=. python scripts/duet_stand_height.py
  PYTHONPATH=. python scripts/duet_stand_height.py --joints \\
      -0.10 0 0 0.30 -0.20 0  -0.10 0 0 0.30 -0.20 0
"""

from __future__ import annotations

import argparse
import math
import sys

import mujoco
import numpy as np

ANKLE_SOLE_DISTANCE = 0.02
LEG_JOINTS = tuple(
  f"{side}_{j}_joint"
  for side in ("left", "right")
  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)
FOOT_SITES = ("left_foot", "right_foot")
TASK = "Unitree-G1-23Dof-Duet-Flat"

# Events kept in Mode 1: the two reset terms that place the robot. Everything
# else is randomisation of the robot or its load and would make the number a
# sample instead of the nominal value.
_KEEP_EVENTS = ("reset_base", "reset_robot_joints")


def sim_height(seconds: float, device: str, hold_level: bool = True) -> dict[str, float]:
  import torch

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  cfg.events = {k: v for k, v in cfg.events.items() if k in _KEEP_EVENTS}
  # A reset during the measurement would restart the settling; the hold makes
  # a fall impossible, and --free reports falls explicitly below.
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env.reset()
  robot = env.scene["robot"]
  site_ids, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  n_steps = int(round(seconds / env.step_dt))
  tail = max(1, int(round(0.5 / env.step_dt)))
  zero = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
  qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
  xy0 = qpos[:, 0:2].clone()
  level = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
  hist: list[tuple[float, float, float, float]] = []
  falls = 0
  with torch.inference_mode():
    for _ in range(n_steps):
      if hold_level:
        # env.step() with the same substep loop, plus the pelvis hold after
        # every physics substep (free joint: qpos[0:3] pos, [3:7] quat;
        # qvel[0:3] lin vel, [3:6] ang vel).
        env.action_manager.process_action(zero)
        for _ in range(env.cfg.decimation):
          env.action_manager.apply_action()
          env.scene.write_data_to_sim()
          env.sim.step()
          qpos[:, 0:2] = xy0
          qpos[:, 3:7] = level
          qvel[:, 0:2] = 0.0
          qvel[:, 3:6] = 0.0
          env.scene.update(dt=env.physics_dt)
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
      else:
        _, _, term, _, _ = env.step(zero)
        falls += int(term.any())
      root_z = float(robot.data.root_link_pos_w[0, 2])
      fz = robot.data.site_pos_w[0, site_ids, 2].tolist()
      hist.append((root_z, fz[0], fz[1], float(robot.data.root_link_lin_vel_w[0].norm())))
  env.close()
  arr = np.array(hist[-tail:])
  root_z, fl, fr, vel = arr.mean(axis=0)
  foot_min = min(fl, fr)
  return {
    "h": root_z - foot_min + ANKLE_SOLE_DISTANCE,
    "root_z": root_z,
    "left_foot_site_z": fl,
    "right_foot_site_z": fr,
    "z_rest": foot_min,
    "root_speed": vel,
    "h_std_last_0.5s": float(np.std(arr[:, 0] - arr[:, 1:3].min(axis=1))),
    "falls": falls,
  }


def fk_height(leg_angles: list[float]) -> dict[str, float]:
  from src.assets.robots import get_g1_23dof_robot_cfg

  robot_cfg = get_g1_23dof_robot_cfg()
  model = robot_cfg.spec_fn().compile()
  data = mujoco.MjData(model)
  data.qpos[:] = 0.0
  data.qpos[2] = 1.0  # pelvis high above anything; only differences are used
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # pelvis level
  init = robot_cfg.init_state.joint_pos or {}
  import re

  for jid in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
    if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    for pat, val in init.items():
      if re.fullmatch(pat, name):
        data.qpos[model.jnt_qposadr[jid]] = val
  for name, q in zip(LEG_JOINTS, leg_angles, strict=True):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = q
  mujoco.mj_kinematics(model, data)
  out: dict[str, float] = {}
  zs = []
  for side, site in zip(("left", "right"), FOOT_SITES, strict=True):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    zs.append(float(data.site_xpos[sid, 2]))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
    z_axis = data.xmat[bid].reshape(3, 3)[:, 2]
    out[f"{side}_foot_tilt_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, z_axis[2]))))
  out["h"] = float(data.qpos[2]) - min(zs) + ANKLE_SOLE_DISTANCE
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--joints", type=float, nargs=12, default=None,
                  help="12 leg angles (rad), SDK order: left hip_pitch, hip_roll, "
                       "hip_yaw, knee, ankle_pitch, ankle_roll, then right.")
  # 10 s, not 3: the sag creeps for several seconds (measured h 0.7967 m at
  # 3 s, 0.7949 m at 10 s and at 20 s, i.e. converged by 10 s).
  ap.add_argument("--seconds", type=float, default=10.0)
  ap.add_argument("--free", action="store_true",
                  help="Mode 1 without the pelvis hold (the robot tips; for comparison).")
  ap.add_argument("--device", default=None)
  a = ap.parse_args()

  if a.joints is not None:
    r = fk_height(list(a.joints))
    print("Mode 2 (FK, pelvis level)")
    print(f"  h = {r['h']:.4f} m   (root_z - min(foot_site_z) + {ANKLE_SOLE_DISTANCE})")
    print(f"  foot tilt from horizontal: left {r['left_foot_tilt_deg']:.2f} deg, "
          f"right {r['right_foot_tilt_deg']:.2f} deg")
    return 0

  import torch

  device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  r = sim_height(a.seconds, device, hold_level=not a.free)
  default_fk = fk_height([-0.1, 0, 0, 0.3, -0.2, 0] * 2)
  print(f"Mode 1 (sim, {TASK}, default pose, zero action, rigid plane, "
        f"{a.seconds:.1f} s, mean of last 0.5 s, "
        f"{'NO pelvis hold' if a.free else 'pelvis held level, z free'})")
  if a.free:
    print(f"  resets from fell_over during the window: {r['falls']}")
  print(f"  h        = {r['h']:.4f} m   (root_z - min(foot_site_z) + {ANKLE_SOLE_DISTANCE})")
  print(f"  root_z   = {r['root_z']:.4f} m")
  print(f"  foot site z: left {r['left_foot_site_z']:+.5f} m, right {r['right_foot_site_z']:+.5f} m")
  print(f"  z_rest   = {r['z_rest']:+.5f} m   (lower foot site above the rigid plane)")
  print(f"  |root v| = {r['root_speed']:.5f} m/s, std(h) over last 0.5 s = {r['h_std_last_0.5s']*1000:.3f} mm")
  print(f"  height_max (h rounded down to 0.01) = {math.floor(r['h'] * 100 + 1e-9) / 100:.2f} m")
  print(f"  cross-check, Mode 2 FK of the same default pose (no PD sag): h = {default_fk['h']:.4f} m")
  return 0


if __name__ == "__main__":
  sys.exit(main())
