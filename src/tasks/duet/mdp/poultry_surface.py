"""Poultry-only sole geometry, marching rewards and optional litter resistance.

All geometry is training-only. No observations, existing reward implementations,
or existing terrain classes are modified. The passive action term owns the two
feet's external-wrench slots and refreshes them on every physics substep.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.terrains import BoxFlatTerrainCfg
from mjlab.utils.lab_api.math import matrix_from_quat

from src.tasks.common.terrains import SoftFlatTerrainCfg


def point_kinematics(position, rotation, linear_velocity, angular_velocity, endpoints, radii):
  """World-space capsule support points and material velocities there.

  Capsule endpoints are body-local sphere centres. Subtract radius along WORLD
  z, then use v + omega cross r at that surface point, including foot rotation.
  The minimum point height is the exact capsule-union minimum on horizontal
  terrain. Sampling individual points also catches a moving toe beside a heel
  that is still planted; a midfoot or minimum-point velocity alone misses that.
  """
  offset = torch.einsum("nfij,fkj->nfki", rotation, endpoints)
  offset = offset.clone()
  offset[..., 2] -= radii
  points = position.unsqueeze(-2) + offset
  velocity = linear_velocity.unsqueeze(-2) + torch.cross(
    angular_velocity.unsqueeze(-2).expand_as(offset), offset, dim=-1
  )
  return points, velocity


def litter_wrench(points, velocity, com, layer_height, strength, max_force):
  """Dissipative horizontal resistance; endpoint average avoids mesh-count gain.

  strength is effective per-foot N s / m^2, depth and layer height are metres.
  Below the substrate the full layer thickness participates, without unbounded
  force growth from soft-contact penetration. No vertical support is added:
  ground contact and the existing contact sensors remain authoritative.
  """
  depth = torch.minimum((layer_height - points[..., 2]).clamp_min(0), layer_height)
  force = -strength[:, None, None, None] * depth[..., None] * velocity
  force = force.clone()
  force[..., 2] = 0
  force /= points.shape[-2]
  total = force.sum(-2)
  scale = (max_force / total.norm(dim=-1).clamp_min(1e-9)).clamp_max(1)
  force *= scale[..., None, None]
  torque = torch.cross(points - com.unsqueeze(-2), force, dim=-1).sum(-2)
  return force.sum(-2), torque


@dataclass(kw_only=True)
class PoultrySurfaceActionCfg(ActionTermCfg):
  """Zero policy dimensions; geometry is also available with litter disabled."""

  litter_enabled: bool = True
  max_layer_height: float = 0.04
  drag_strength_range: tuple[float, float] = (500.0, 1500.0)
  max_drag_force: float = 150.0
  period: float = 0.6
  stance_fraction: float = 0.56

  def build(self, env):
    return PoultrySurfaceAction(self, env)


class PoultrySurfaceAction(ActionTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if cfg.max_layer_height < 0 or cfg.max_drag_force <= 0:
      raise ValueError("Layer height must be nonnegative and force cap positive")
    lo, hi = cfg.drag_strength_range
    if not 0 <= lo <= hi:
      raise ValueError("Invalid litter drag strength range")
    self.body_ids, _ = self._entity.find_bodies(
      ("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True
    )
    # Read the actual compiled capsules, including left/right differences.
    model = env.sim.mj_model
    endpoints, radii = [], []
    for side in ("left", "right"):
      pts, rad = [], []
      for i in range(1, 8):
        geom = model.geom(f"{cfg.entity_name}/{side}_foot{i}_collision")
        if geom.type[0] != mujoco.mjtGeom.mjGEOM_CAPSULE:
          raise ValueError("Poultry sole measurement requires foot capsules")
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, geom.quat)
        axis = rot.reshape(3, 3)[:, 2] * geom.size[1]
        pts.extend((geom.pos - axis, geom.pos + axis))
        rad.extend((geom.size[0], geom.size[0]))
      endpoints.append(pts)
      radii.append(rad)
    self.endpoints = torch.tensor(np.array(endpoints), device=self.device, dtype=torch.float32)
    self.radii = torch.tensor(np.array(radii), device=self.device, dtype=torch.float32)
    self.toe_index = self.endpoints[..., 0].argmax(-1)
    self._raw_action = torch.zeros((self.num_envs, 0), device=self.device)
    self.strength = torch.empty(self.num_envs, device=self.device).uniform_(lo, hi)
    self.last_force = torch.zeros((self.num_envs, 2, 3), device=self.device)
    self.last_torque = torch.zeros_like(self.last_force)
    self.peak = torch.zeros((self.num_envs, 2), device=self.device)
    self.last_apex = torch.zeros_like(self.peak)
    self.last_liftoff_angle = torch.zeros_like(self.peak)
    self.was_air = torch.zeros_like(self.peak, dtype=torch.bool)
    self.complete_swing = torch.zeros_like(self.was_air)
    self.has_landed = torch.zeros_like(self.was_air)
    self._sample_key = None
    self.terrain_cfg = env.cfg.scene.terrain.terrain_generator
    if self.terrain_cfg is not None:
      tg = self.terrain_cfg
      subs = list(tg.sub_terrains.values())
      if not tg.curriculum and len(subs) != 1:
        raise ValueError("Poultry masking needs curriculum columns or a single terrain type")
      proportions = np.array([s.proportion for s in subs], dtype=float)
      indices = np.searchsorted(np.cumsum(proportions / proportions.sum()),
                                np.arange(tg.num_cols) / tg.num_cols + 0.001, side="right")
      self.flat_columns = torch.tensor(
        [isinstance(subs[i], (BoxFlatTerrainCfg, SoftFlatTerrainCfg)) for i in indices],
        device=self.device,
      )
      self.soft_columns = torch.tensor(
        [isinstance(subs[i], SoftFlatTerrainCfg) for i in indices], device=self.device,
      )

  @property
  def action_dim(self):
    return 0

  @property
  def raw_action(self):
    return self._raw_action

  def process_actions(self, actions):
    pass

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.strength[ids] = torch.empty_like(self.strength[ids]).uniform_(*self.cfg.drag_strength_range)
    for buf in (self.peak, self.last_apex, self.last_liftoff_angle, self.was_air,
                self.complete_swing, self.has_landed, self.last_force, self.last_torque):
      buf[ids] = 0
    self._entity.write_external_wrench_to_sim(
      self.last_force[ids], self.last_torque[ids], env_ids=env_ids, body_ids=self.body_ids
    )
    self._sample_key = None

  def terrain_at(self, points):
    """Actual point location, not spawn column: safe across tile boundaries.

    Poultry's play config uses the same deterministic column layout as training.
    The legacy Rough play config randomizes tile types, so terrain_types alone
    is not a terrain label there. Heightfield and border points are excluded.
    """
    tg = self.terrain_cfg
    if tg is None:
      return torch.ones_like(points[..., 2], dtype=torch.bool), torch.zeros_like(points[..., 2])
    row = torch.floor(points[..., 0] / tg.size[0] + tg.num_rows / 2).long()
    col = torch.floor(points[..., 1] / tg.size[1] + tg.num_cols / 2).long()
    inside = (row >= 0) & (row < tg.num_rows) & (col >= 0) & (col < tg.num_cols)
    col = col.clamp(0, tg.num_cols - 1)
    valid = inside & self.flat_columns[col]
    soft = inside & self.soft_columns[col]
    height = self.cfg.max_layer_height * row.clamp(0, tg.num_rows - 1) / max(tg.num_rows - 1, 1)
    height = height * soft * self.cfg.litter_enabled
    return valid, height

  def geometry(self):
    data = self._entity.data
    rotation = matrix_from_quat(data.body_link_quat_w[:, self.body_ids])
    points, velocity = point_kinematics(
      data.body_link_pos_w[:, self.body_ids], rotation,
      data.body_link_lin_vel_w[:, self.body_ids], data.body_link_ang_vel_w[:, self.body_ids],
      self.endpoints, self.radii,
    )
    valid, height = self.terrain_at(points)
    return points, velocity, rotation, valid, height

  def apply_actions(self):
    if self.cfg.litter_enabled:
      points, velocity, _, _, height = self.geometry()
      force, torque = litter_wrench(
        points, velocity, self._entity.data.body_com_pos_w[:, self.body_ids],
        height, self.strength, self.cfg.max_drag_force,
      )
      self.last_force.copy_(force)
      self.last_torque.copy_(torque)
    else:
      self.last_force.zero_()
      self.last_torque.zero_()
    # Only the new task owns these foot slots. Other bodies' external forces
    # are preserved. Zero is written above the layer and when Phase B is off.
    self._entity.write_external_wrench_to_sim(
      self.last_force, self.last_torque, body_ids=self.body_ids
    )

  def sample(self):
    key = (self._env.common_step_counter, self._env._sim_step_counter)
    if key == self._sample_key:
      return self._sample
    points, velocity, rotation, valid, height = self.geometry()
    # Two references. The stance foot rests on the floor (z = 0 on every valid
    # tile), so swing height is measured from there: a 9 cm target stays 9 cm
    # on every litter row, inside the ~10 cm level-foot limit of the ankle and
    # hip measured by IK. Scooping happens in the loose layer above the floor,
    # so plow cost and scoop metrics use the litter surface instead.
    clearance = points[..., 2] - height
    lowest = points[..., 2].min(-1).values
    lowest_litter = clearance.min(-1).values
    foot_valid = valid.all(-1)
    contact = self._env.scene["feet_ground_contact"].data.found > 0
    air = ~contact
    command = self._env.command_manager.get_command("twist")
    moving = command[:, :2].norm(dim=-1) + command[:, 2].abs() > 0.1
    active = foot_valid & moving[:, None]
    liftoff = air & ~self.was_air & active
    landing = ~air & self.was_air & self.complete_swing & active
    self.complete_swing |= liftoff
    self.complete_swing &= active
    # In-place: rollouts run under torch.inference_mode, and a buffer rebound
    # there becomes an inference tensor that a later reset outside it cannot
    # write to.
    self.peak.copy_(torch.where(
      liftoff, lowest, torch.where(air, torch.maximum(self.peak, lowest), self.peak)
    ))
    self.last_apex.copy_(torch.where(landing, self.peak, self.last_apex))
    self.has_landed |= landing
    toe_down = -torch.asin(rotation[..., 2, 0].clamp(-1, 1))
    self.last_liftoff_angle.copy_(torch.where(liftoff, toe_down, self.last_liftoff_angle))
    self.complete_swing &= air
    self.was_air.copy_(air)
    phase = self._env.episode_length_buf[:, None] * self._env.step_dt / self.cfg.period
    phase = (phase + torch.tensor([0., 0.5], device=self.device)) % 1
    planned_swing = phase >= self.cfg.stance_fraction
    self._sample = dict(points=points, velocity=velocity, rotation=rotation,
                        valid=valid, layer=height, clearance=clearance, lowest=lowest,
                        lowest_litter=lowest_litter, active=active, air=air,
                        landing=landing, contact=contact,
                        planned_swing=planned_swing, toe_down=toe_down)
    self._sample_key = key
    return self._sample


def _surface(env):
  return env.action_manager.get_term("poultry_surface")


def poultry_clearance(env, target_height=0.09):
  """Lowest sole above the stance floor, travel-weighted; moving, valid tiles.

  Replaces the old midfoot WORLD-z clearance only in PoultrySurface. No contact
  gate: low travel while the foot still touches is also costly.
  """
  s = _surface(env).sample()
  speed = s["velocity"][..., :2].norm(dim=-1).mean(-1)
  return ((s["lowest"] - target_height).abs() * speed * s["active"]).sum(-1)


def poultry_plow(env, threshold=0.04):
  """Mean over sole points, summed over feet: depth deficit times WORLD xy speed.

  Every point uses its own ground-relative clearance and material velocity.
  Contact-independent, gated to moving commands and flat/soft tile points.
  """
  s = _surface(env).sample()
  cost = s["velocity"][..., :2].norm(dim=-1) * (threshold - s["clearance"]).clamp_min(0)
  return (cost.mean(-1) * s["active"]).sum(-1)


def poultry_swing_height(env, target_height=0.09):
  """Lowest-sole peak above the stance floor, evaluated once on completed landing.

  Normalize by target and by policy dt so the configured weight represents a
  per-landing cost, independent of control frequency. State resets per episode.
  """
  state = _surface(env)
  s = state.sample()
  error = (state.last_apex / target_height - 1).square().clamp_max(4)
  return (error * s["landing"]).sum(-1) / env.step_dt


def poultry_foot_level(env, threshold=0.05, deadband_deg=5.0):
  """Foot orientation near the litter surface, during scheduled or actual swing.

  Flat/soft surfaces have world-up normals. The stance foot remains free to
  balance; near-ground swing allows a small level/toe-up/down deadband.
  """
  s = _surface(env).sample()
  tilt = torch.acos(s["rotation"][..., 2, 2].clamp(-1, 1))
  error = (tilt - np.deg2rad(deadband_deg)).clamp_min(0).square()
  gate = s["active"] & (s["lowest_litter"] < threshold) & (s["air"] | s["planned_swing"])
  return (error * gate).sum(-1)


def poultry_metric(env, name):
  """Per-step diagnostics only; actor and critic observation layouts unchanged."""
  state = _surface(env)
  s = state.sample()
  active = s["active"]
  if name == "plow":
    return poultry_plow(env)
  if name == "apex":
    mask = active & state.has_landed
    return (state.last_apex * mask).sum(-1) / mask.sum(-1).clamp_min(1)
  if name == "clearance":
    mask = active & s["air"]
    return (s["lowest"] * mask).sum(-1) / mask.sum(-1).clamp_min(1)
  if name == "liftoff_toe_down":
    return (torch.rad2deg(state.last_liftoff_angle) * active).sum(-1) / active.sum(-1).clamp_min(1)
  if name in ("low_travel", "contact_slide"):
    idx = state.toe_index[None, :, None].expand(env.num_envs, -1, -1)
    toe_speed = s["velocity"][..., :2].norm(dim=-1).gather(-1, idx).squeeze(-1)
    toe_clearance = s["clearance"].gather(-1, idx).squeeze(-1)
    mask = toe_clearance < 0.03 if name == "low_travel" else s["contact"]
    return (toe_speed * mask * active).sum(-1)
  if name == "drag_force":
    return state.last_force.norm(dim=-1).sum(-1)
  if name == "layer":
    return s["layer"].mean(dim=(-1, -2))
  if name == "falls":
    return env.reset_terminated.float()
  raise ValueError(name)
