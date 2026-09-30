"""Calibrate foot contact softness: static sinkage vs ``geom_solref`` timeconst.

The rough-blind-tall tasks randomise the foot geoms' ``solref`` time constant
in ``[0.02, tc_max]`` (``mdp.geom_solref``). ``tc_max`` must be CALIBRATED,
not guessed: static penetration grows with timeconst^2, so a range that looks
modest can be either rigid or a swamp. This script measures it.

For every timeconst in ``--timeconsts`` on the rigid plane, 16 envs, zero
action, default pose, all 14 foot collision geoms set to that timeconst
(dampratio 1, default solimp), it reports:

  sinkage_mm       settled static sinkage: foot-site z relative to the 0.02 s
                   baseline (positive = deeper), mean of both feet and all envs
  osc_mm           std of the foot-site z over the last 0.5 s (0 = settled)
  drop_sink_mm     deepest foot-site z during a 5 cm drop onto the same ground,
                   relative to the rigid rest height (the tunnelling check:
                   must stay under 50 mm)
  max_con/world    most contacts any one world had at any step
  max_nefc/world   most constraint rows any one world had (njmax is the cap)
  env_steps/s      policy steps per second at this batch size (plain env.step)
  unstable         NaN, non-settling, or the drop sinking past 5 cm

As in ``duet_stand_height.py``, the pelvis is held level and fixed in x/y after
every physics substep (height free): the default pose is not statically stable
under zero action on these soft gains and would otherwise tip within ~1.2 s.
The hold applies no vertical force, so the sinkage is the one gravity and the
contact produce.

Usage:
  PYTHONPATH=. python scripts/duet_contact_softness_sweep.py
  PYTHONPATH=. python scripts/duet_contact_softness_sweep.py --target-mm 25
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg

import src.tasks.duet.mdp as mdp

TASK = "Unitree-G1-23Dof-Duet-Flat"
TIMECONSTS = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30)
FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
FOOT_SITES = ("left_foot", "right_foot")
_KEEP_EVENTS = ("reset_base", "reset_robot_joints")


def build(num_envs: int, device: str) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = num_envs
  cfg.events = {k: v for k, v in cfg.events.items() if k in _KEEP_EVENTS}
  cfg.events["foot_solref"] = EventTermCfg(
    mode="reset",
    func=mdp.geom_solref,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
      "operation": "abs",
      "ranges": (0.02, 0.02),
      "shared_random": True,
    },
  )
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def held_step(env: ManagerBasedRlEnv, action: torch.Tensor, xy0: torch.Tensor) -> None:
  """``env.step`` physics with the pelvis held level and fixed in x/y."""
  qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
  env.action_manager.process_action(action)
  for _ in range(env.cfg.decimation):
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.step()
    qpos[:, 0:2] = xy0
    qpos[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    qvel[:, 0:2] = 0.0
    qvel[:, 3:6] = 0.0
    env.scene.update(dt=env.physics_dt)
  env.sim.forward()
  env.scene.update(dt=env.physics_dt)


def contact_counts(env: ManagerBasedRlEnv) -> tuple[int, int]:
  d = env.sim.data
  n = int(d.nacon[0])
  if n > 0:
    wid = d.contact.worldid[:n].long()
    max_con = int(torch.bincount(wid, minlength=env.num_envs).max())
  else:
    max_con = 0
  return max_con, int(d.nefc[:].max())


def set_timeconst(env: ManagerBasedRlEnv, tc: float) -> None:
  env.event_manager.get_term_cfg("foot_solref").params["ranges"] = (tc, tc)
  env.reset()
  robot = env.scene["robot"]
  gids = robot.indexing.geom_ids[robot.find_geoms(FOOT_GEOMS, preserve_order=True)[0]]
  got = env.sim.model.geom_solref[:, gids, 0]
  assert torch.allclose(got, torch.full_like(got, tc)), (tc, got)


def measure(env: ManagerBasedRlEnv, tc: float, settle_s: float) -> dict[str, float]:
  robot = env.scene["robot"]
  sid, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  set_timeconst(env, tc)
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  xy0 = env.sim.data.qpos[:, 0:2].clone()
  n = int(round(settle_s / env.step_dt))
  tail = int(round(0.5 / env.step_dt))
  zs = []
  max_con = max_nefc = 0
  for _ in range(n):
    held_step(env, zero, xy0)
    zs.append(robot.data.site_pos_w[:, sid, 2].mean(dim=1).clone())
    c, e = contact_counts(env)
    max_con, max_nefc = max(max_con, c), max(max_nefc, e)
  z = torch.stack(zs[-tail:])  # [T, N]
  settled_z = float(z.mean())
  osc = float(z.std(dim=0).mean())
  finite = bool(torch.isfinite(env.sim.data.qpos[:]).all())
  # Drop test: lift the whole robot 5 cm (zero velocity) and let it land.
  env.sim.data.qpos[:, 2] = env.sim.data.qpos[:, 2] + 0.05
  env.sim.data.qvel[:] = 0.0
  env.sim.forward()
  env.scene.update(dt=env.physics_dt)
  drop_min = float("inf")
  for _ in range(int(round(1.5 / env.step_dt))):
    held_step(env, zero, xy0)
    drop_min = min(drop_min, float(robot.data.site_pos_w[:, sid, 2].min()))
    c, e = contact_counts(env)
    max_con, max_nefc = max(max_con, c), max(max_nefc, e)
  return {
    "tc": tc,
    "z": settled_z,
    "osc_mm": osc * 1000,
    "drop_min_z": drop_min,
    "max_con": max_con,
    "max_nefc": max_nefc,
    "finite": finite,
  }


def throughput(env: ManagerBasedRlEnv, tc: float, steps: int = 200) -> float:
  set_timeconst(env, tc)
  zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  for _ in range(20):
    env.step(zero)
  torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(steps):
    env.step(zero)
  torch.cuda.synchronize()
  return env.num_envs * steps / (time.perf_counter() - t0)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--timeconsts", type=float, nargs="+", default=list(TIMECONSTS))
  ap.add_argument("--num-envs", type=int, default=16)
  ap.add_argument("--settle-s", type=float, default=5.0)
  ap.add_argument("--target-mm", type=float, default=25.0,
                  help="Static sinkage tc_max should reproduce (operator litter "
                       "sinkage x1.25; 25 mm when no measurement exists).")
  ap.add_argument("--device", default=None)
  a = ap.parse_args()
  device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env = build(a.num_envs, device)
  # One inference-mode scope for everything, resets included: tensors created
  # inside it cannot be updated in place outside it.
  with torch.inference_mode():
    rows = [measure(env, tc, a.settle_s) for tc in a.timeconsts]
    thr = {tc: throughput(env, tc) for tc in a.timeconsts}
  njmax = env.cfg.sim.njmax
  env.close()

  base = next((r for r in rows if abs(r["tc"] - 0.02) < 1e-9), None)
  if base is None:
    print("the sweep must include the 0.02 s baseline")
    return 1
  print(f"\n{TASK}, rigid plane, {a.num_envs} envs, zero action, default pose, "
        f"pelvis held level; foot-site z at rest on the 0.02 baseline "
        f"= {base['z'] * 1000:+.2f} mm (njmax {njmax})")
  print(f"{'timeconst':>9} {'sinkage_mm':>10} {'osc_mm':>7} {'drop_sink_mm':>12} "
        f"{'max_con/world':>13} {'max_nefc/world':>14} {'env_steps/s':>11}  unstable")
  sink = []
  for r in rows:
    s = (base["z"] - r["z"]) * 1000
    drop = (base["z"] - r["drop_min_z"]) * 1000
    unstable = (not r["finite"]) or r["osc_mm"] > 1.0 or drop > 50.0
    sink.append(s)
    print(f"{r['tc']:9.2f} {s:10.2f} {r['osc_mm']:7.3f} {drop:12.2f} "
          f"{r['max_con']:13d} {r['max_nefc']:14d} {thr[r['tc']]:11.0f}  "
          f"{'YES' if unstable else 'no'}")
  tcs = np.array([r["tc"] for r in rows])
  sink = np.array(sink)
  order = np.argsort(tcs)
  if a.target_mm <= sink[order][-1]:
    tc_star = float(np.interp(a.target_mm, sink[order], tcs[order]))
    print(f"\ntimeconst for {a.target_mm:.1f} mm static sinkage (linear interpolation "
          f"of the measured curve): {tc_star:.4f} s")
  else:
    print(f"\n{a.target_mm:.1f} mm is deeper than the sweep reaches.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
