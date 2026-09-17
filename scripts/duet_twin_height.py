"""Measure the ACHIEVED pelvis height while the deployed controller runs.

The C++ controller already prints the height it is *commanding*
(``[base_height] current cmd = ...`` in the g1_ctrl terminal). That is the
setpoint, not what the robot reached -- and the two differ, especially near the
bottom of the range where the policy saturates.

This subscribes to ``rt/lowstate`` and reports the achieved height, defined
exactly as the training reward defines it (``track_base_height``):

    pelvis height above the LOWER foot = pelvis_z - min(left_foot_z, right_foot_z)

It is computed by forward kinematics from the reported joint angles and the IMU
orientation, so it needs nothing from the simulator beyond the standard state
topic and works unchanged on hardware. Using the IMU attitude matters: a leaning
robot's feet are not directly below its pelvis, and ignoring the tilt overstates
the height.

Usage (with g1_ctrl + the sim + keyboard.py already running):

    PYTHONPATH=. python scripts/duet_twin_height.py
    PYTHONPATH=. python scripts/duet_twin_height.py --iface lo --hz 5
    PYTHONPATH=. python scripts/duet_twin_height.py --selftest   # no DDS needed

``--selftest`` runs the same maths on the model's default standing pose and
should print ~0.79 m; use it to confirm the kinematics before trusting a
live number.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import mujoco
import numpy as np

from src.assets.robots.unitree_g1.g1_23dof_constants import get_spec

# Policy joint order -> Unitree SDK motor index. Copied from
# deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml
# (`joint_ids_map`); the 23 policy joints are not contiguous in the SDK's
# 29-motor array, so this mapping is what keeps the angles in the right slots.
JOINT_IDS_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]

FOOT_SITES = ("left_foot", "right_foot")


class HeightFK:
  """Forward kinematics for pelvis-height-above-the-lower-foot."""

  def __init__(self) -> None:
    self.model = get_spec().compile()
    self.data = mujoco.MjData(self.model)
    self.site_ids = [
      mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in FOOT_SITES
    ]
    if any(i < 0 for i in self.site_ids):
      raise RuntimeError(f"foot sites {FOOT_SITES} not found in the model")
    # qpos layout: [0:3] pelvis xyz, [3:7] pelvis quat (w,x,y,z), [7:] 23 joints.
    njoint = self.model.nq - 7
    if njoint != len(JOINT_IDS_MAP):
      raise RuntimeError(f"model has {njoint} joints, map has {len(JOINT_IDS_MAP)}")

  def compute(self, q: np.ndarray, quat_wxyz: np.ndarray) -> dict:
    """q: 23 joint angles in policy order. quat: pelvis attitude (w,x,y,z)."""
    self.data.qpos[:3] = 0.0  # pelvis at the origin; only relative z matters
    self.data.qpos[3:7] = quat_wxyz
    self.data.qpos[7:] = q
    mujoco.mj_kinematics(self.model, self.data)
    foot_z = np.array([self.data.site_xpos[i][2] for i in self.site_ids])
    # Pelvis is at z = 0, so the height above the lower foot is -min(foot_z).
    height = 0.0 - float(foot_z.min())
    # Torso pitch/roll from the same attitude, positive pitch = leaning forward.
    w, x, y, z = quat_wxyz
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return {
      "height": height,
      "left_foot_z": float(-foot_z[0]),
      "right_foot_z": float(-foot_z[1]),
      "pitch_deg": math.degrees(pitch),
      "roll_deg": math.degrees(roll),
    }


def selftest() -> int:
  fk = HeightFK()
  m = fk.model
  # Default standing pose = the model's own keyframe joint values.
  d0 = mujoco.MjData(m)
  mujoco.mj_resetData(m, d0)
  q = np.array(d0.qpos[7:]).copy()
  r = fk.compute(q, np.array([1.0, 0.0, 0.0, 0.0]))
  print(f"selftest, model default pose: height = {r['height']:.3f} m "
        f"(pitch {r['pitch_deg']:+.1f} deg)")
  print("expected ~0.79 m for the G1 standing pose; a wildly different number "
        "means the joint order or site lookup is wrong.")
  return 0


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--iface", default="lo",
                  help="network interface for DDS (keyboard.py uses 'lo')")
  ap.add_argument("--topic", default="rt/lowstate")
  ap.add_argument("--hz", type=float, default=5.0, help="print rate")
  ap.add_argument("--csv", default=None, help="also append samples to this CSV")
  ap.add_argument("--selftest", action="store_true")
  a = ap.parse_args()

  if a.selftest:
    return selftest()

  from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

  fk = HeightFK()
  latest = {"msg": None}

  def on_msg(msg) -> None:
    latest["msg"] = msg

  ChannelFactoryInitialize(0, a.iface)
  sub = ChannelSubscriber(a.topic, LowState_)
  sub.Init(on_msg, 10)
  print(f"listening on {a.topic} (iface {a.iface}); Ctrl-C to stop")
  print(f"{'height_m':>9} {'min_foot':>9} {'pitch_deg':>10} {'roll_deg':>9}")

  csv = open(a.csv, "a") if a.csv else None
  if csv:
    csv.write("t,height_m,pitch_deg,roll_deg\n")
  period = 1.0 / max(a.hz, 0.1)
  lo_seen, hi_seen = 9.9, -9.9
  t0 = time.time()
  try:
    while True:
      time.sleep(period)
      msg = latest["msg"]
      if msg is None:
        print("  (no lowstate yet -- is the sim running, and is --iface right?)")
        continue
      q = np.array([msg.motor_state[i].q for i in JOINT_IDS_MAP], dtype=float)
      quat = np.array(msg.imu_state.quaternion, dtype=float)
      if not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1.0) > 0.2:
        quat = np.array([1.0, 0.0, 0.0, 0.0])  # fall back to upright
      r = fk.compute(q, quat)
      lo_seen = min(lo_seen, r["height"]); hi_seen = max(hi_seen, r["height"])
      print(f"{r['height']:9.3f} {min(r['left_foot_z'], r['right_foot_z']):9.3f} "
            f"{r['pitch_deg']:10.1f} {r['roll_deg']:9.1f}   "
            f"[min {lo_seen:.3f} / max {hi_seen:.3f}]")
      if csv:
        csv.write(f"{time.time()-t0:.3f},{r['height']:.4f},"
                  f"{r['pitch_deg']:.2f},{r['roll_deg']:.2f}\n")
        csv.flush()
  except KeyboardInterrupt:
    print(f"\nlowest height reached: {lo_seen:.3f} m   highest: {hi_seen:.3f} m")
  finally:
    if csv:
      csv.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
