"""Deployment metadata written into every exported DUET policy.

Why this exists: a stale policy was once deployed against a mismatched
``deploy.yaml`` and the robot misbehaved. The ONNX file alone is opaque -- it
carries no record of the PD gains, action scale or joint order it was trained
against, so nothing could have caught the mismatch before it reached hardware.

Every field here is read back by ``scripts/check_deploy_consistency.py`` and
compared against the ``deploy.yaml`` the controller will actually load. The
``deploy_config_hash`` is a single value summarising the safety-critical subset,
so a human can eyeball one string instead of six arrays.
"""

from __future__ import annotations

import hashlib
import json

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction

# The fields that MUST agree between training and deployment for the policy to
# behave as trained. Anything outside this set may differ (log paths, run names,
# episode length, ...) without affecting the closed loop.
SAFETY_CRITICAL_FIELDS = (
  "joint_names",
  "joint_stiffness",
  "joint_damping",
  "default_joint_pos",
  "action_joint_names",
  "action_scale",
  "action_offset",
  "observation_names",
  "observation_dims",
  "obs_dim",
  "action_dim",
  "step_dt",
)


def _round(values, decimals: int = 4) -> list[float]:
  return [round(float(v), decimals) for v in values]


def _final_stage_value(env: ManagerBasedRlEnv, key: str, fallback):
  """Value of ``key`` at the last stage of any staged curriculum term.

  The command curricula widen the sampling ranges over training, mutating the
  command term's cfg in place. A policy exported at the end of a run was trained
  on the FINAL stage, but a freshly-built cfg (and any play-mode cfg, which
  clears the curriculum entirely) still holds the stage-0 values. Walking the
  declared stages recovers the right answer in both cases.
  """
  best = fallback
  for term in env.cfg.curriculum.values():
    for param in ("velocity_stages", "stages"):
      stages = term.params.get(param)
      if not stages:
        continue
      for stage in stages:
        if key in stage and stage[key] is not None:
          best = stage[key]
  return best


def build_deploy_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  """Collect the full training-side deployment contract for ONNX metadata."""
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, JointPositionAction)

  # Actuator gains in NATURAL JOINT ORDER (robot.joint_names), which is the
  # order deploy.yaml's stiffness/damping arrays are written in. Each spec
  # actuator drives exactly one joint via its target field.
  joint_name_to_ctrl_id = {
    a.target.split("/")[-1]: a.id for a in robot.spec.actuators
  }
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[j] for j in robot.joint_names if j in joint_name_to_ctrl_id
  ]
  stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]

  def _as_list(x, n: int) -> list[float]:
    if isinstance(x, torch.Tensor):
      return _round(x[0].cpu().tolist())
    return _round([float(x)] * n)

  # With actor history H, mjlab reports each term's width as D*H (the term's H
  # frames, oldest first, then the next term). The contract records per-FRAME
  # widths -- what deploy.yaml's per-term ``scale`` lists describe -- plus H;
  # obs_dim stays the full ONNX input width, sum(widths) * H.
  om = env.observation_manager
  history = max(1, int(om.cfg["actor"].history_length or 1))
  obs_names = list(om.active_terms["actor"])
  full_dims = [int(d[0]) for d in om.group_obs_term_dim["actor"]]
  assert all(d % history == 0 for d in full_dims), (full_dims, history)
  obs_dims = [d // history for d in full_dims]

  metadata: dict[str, list | str | float] = {
    "run_path": run_path,
    "task": "DUET-G1-23Dof",
    # -- joint / actuator contract -------------------------------------------
    "joint_names": list(robot.joint_names),
    "joint_stiffness": _round(stiffness.tolist()),
    "joint_damping": _round(damping.tolist()),
    "default_joint_pos": _round(robot.data.default_joint_pos[0].cpu().tolist()),
    # -- action contract (the 13 RL-controlled joints, in action order) ------
    "action_joint_names": list(joint_action.target_names),
    "action_scale": _as_list(joint_action.scale, joint_action.action_dim),
    "action_offset": _as_list(joint_action.offset, joint_action.action_dim),
    "action_dim": int(joint_action.action_dim),
    # -- observation contract -------------------------------------------------
    "observation_names": obs_names,
    "observation_dims": obs_dims,
    "obs_dim": int(sum(full_dims)),
    # Not in SAFETY_CRITICAL_FIELDS: obs_dim together with observation_dims
    # already fixes it, and leaving it out keeps the hash of every 1-frame
    # export identical to what it was before history existed.
    "history_length": history,
    "command_names": list(env.command_manager.active_terms),
    "step_dt": round(float(env.step_dt), 6),
  }

  # Command ranges the policy was actually trained on, taken from the FINAL
  # curriculum stage rather than the term's initial value. Reading the live cfg
  # would record the stage-0 ranges -- the narrowest ones, seen only in the
  # first few thousand iterations -- and the checker would then wrongly warn
  # that a correctly deployed policy is extrapolating.
  height_term = env.command_manager.get_term("base_height")
  if height_term is not None:
    metadata["height_command_range"] = _round(height_term.cfg.height_range)
    metadata["height_walk_min"] = round(
      _final_stage_value(env, "walk_min_height", height_term.cfg.walk_min_height), 4
    )
  twist_term = env.command_manager.get_term("twist")
  if twist_term is not None:
    r = twist_term.cfg.ranges
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
      metadata[f"twist_range_{axis}"] = _round(
        _final_stage_value(env, axis, getattr(r, axis))
      )

  metadata["deploy_config_hash"] = config_hash(metadata)
  return metadata


def config_hash(metadata: dict) -> str:
  """SHA-256 over the safety-critical subset, in a fixed key order.

  Deterministic across processes and machines: keys are sorted, floats are
  already rounded by the caller, and the encoding is canonical JSON.
  """
  payload = {k: metadata[k] for k in SAFETY_CRITICAL_FIELDS if k in metadata}
  blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
  return hashlib.sha256(blob.encode()).hexdigest()[:16]
