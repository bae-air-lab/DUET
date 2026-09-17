"""Diagnostic metrics for the loco-manipulation task.

Each function returns a per-env scalar (shape ``[B]``) and is registered as a
``MetricsTermCfg``; the metrics manager averages it over the episode and logs
``Episode_Metrics/<name>``. No weights, no dt scaling -- the logged value is in
the unit stated in each docstring. These exist so that "the policy got more
robust" is a number on the training curves rather than an impression from the
viewer: torso tilt, stationary drift, CoM support error, foot slip, tracking
errors and the arm disturbance magnitude are all read side by side.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .rewards import _com_support_ratio

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

__all__ = [
  "torso_pitch_abs",
  "torso_roll_abs",
  "idle_root_drift",
  "com_support_error",
  "foot_slip_speed",
  "arm_traj_speed",
  "arm_traj_accel",
  "lin_vel_track_error",
  "yaw_track_error",
  "height_track_error",
]


def _body_tilt(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Gravity in the body frame of ``asset_cfg.body_ids`` (one body). [B, 3]."""
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  return quat_apply_inverse(quat, asset.data.gravity_vec_w)


def torso_pitch_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """|pitch| of the given body (rad), from its projected gravity."""
  g = _body_tilt(env, asset_cfg)
  return torch.atan2(g[:, 0], -g[:, 2]).abs()


def torso_roll_abs(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """|roll| of the given body (rad), from its projected gravity."""
  g = _body_tilt(env, asset_cfg)
  return torch.atan2(g[:, 1], -g[:, 2]).abs()


def idle_root_drift(
  env: ManagerBasedRlEnv, anchor_term_name: str = "idle_position_anchor"
) -> torch.Tensor:
  """Root xy displacement (m) from the idle anchor while idle, else 0.

  Reads the anchor latched by the ``idle_position_anchor`` reward term so the
  metric and the reward agree on what "where it stopped" means. Zero if the
  term is not in the reward set (e.g. the no-idle-precision ablation).
  """
  if anchor_term_name not in env.reward_manager.active_terms:
    return torch.zeros(env.num_envs, device=env.device)
  term = env.reward_manager.get_term_cfg(anchor_term_name).func
  asset: Entity = env.scene["robot"]
  dist = torch.norm(asset.data.root_link_pos_w[:, :2] - term.anchor_xy, dim=1)
  return dist * term.was_lin_idle.float()


def com_support_error(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  foot_half_length: float = 0.09,
  foot_half_width: float = 0.04,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Normalised CoM offset from the support centre in double support (1 = edge).

  Same quantity the ``com_support_region`` reward penalises; zero when not in
  double support so the episode average reads as "how close to the edge the
  CoM sat while both feet were down".
  """
  r, double_support = _com_support_ratio(
    env, sensor_name, asset_cfg, foot_half_length, foot_half_width
  )
  return r * double_support.float()


def foot_slip_speed(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Mean horizontal speed (m/s) of the feet that are in contact."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()
  speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  return (speed * in_contact).sum(dim=1) / in_contact.sum(dim=1).clamp(min=1.0)


def arm_traj_speed(
  env: ManagerBasedRlEnv, action_term_name: str = "upper_body_pose"
) -> torch.Tensor:
  """Mean |target velocity| over the arm joints (rad/s)."""
  term = env.action_manager.get_term(action_term_name)
  return term.traj_vel.abs().mean(dim=1)


def arm_traj_accel(
  env: ManagerBasedRlEnv, action_term_name: str = "upper_body_pose"
) -> torch.Tensor:
  """Mean |target acceleration| over the arm joints (rad/s^2)."""
  term = env.action_manager.get_term(action_term_name)
  return term.traj_acc.abs().mean(dim=1)


def lin_vel_track_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """|v_xy,cmd - v_xy| in the base frame (m/s)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2], dim=1)


def yaw_track_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """|yaw_rate,cmd - yaw_rate| (rad/s)."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]).abs()


def height_track_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  ankle_sole_distance: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """|h_cmd - pelvis height above the lower foot| (m), as ``track_base_height``."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  root_z = asset.data.root_link_pos_w[:, 2]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  base_height = root_z - torch.min(foot_z, dim=1).values
  return (base_height - command[:, 0] + ankle_sole_distance).abs()
