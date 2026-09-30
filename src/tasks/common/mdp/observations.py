from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase



def arm_traj_vel(
  env: ManagerBasedRlEnv, action_term_name: str = "upper_body_pose"
) -> torch.Tensor:
  """Target velocity of the externally driven arm joints (rad/s). [B, J].

  Arm trajectory INTENT, as opposed to the measured arm joint velocity that is
  already in ``joint_vel``: it is the generator's commanded velocity during
  training and the finite difference of the external targets at deployment.
  Given to the CRITIC only in the DUET config -- it lets the value function
  anticipate an arm-induced disturbance without widening the actor's 71-D
  deployment interface. Adding it to the actor would require the C++ controller
  to publish dq_ref alongside q_ref; that is documented as a future step.
  """
  term = env.action_manager.get_term(action_term_name)
  return term.traj_vel


def foot_friction_coef(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Current sliding friction of the foot geoms, ``geom_friction[..., 0]``. [B, 1].

  Privileged (critic only). The foot-friction event draws one value per env and
  shares it across all 14 foot geoms, so the mean is that value. Read from the
  live model, so it is whatever the latest randomisation wrote.
  """
  asset: Entity = env.scene[asset_cfg.name]
  gids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  mu = env.sim.model.geom_friction[:, gids, 0].mean(dim=1, keepdim=True)
  return mu.expand(env.num_envs, 1)


def foot_softness(
  env: ManagerBasedRlEnv,
  left_cfg: SceneEntityCfg,
  right_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Current contact time constant under each foot, ``geom_solref[..., 0]``. [B, 2].

  Privileged (critic only), [left, right]. Each foot's seven collision geoms
  share one sampled value (``mdp.geom_solref`` with ``shared_random``), and it
  is re-drawn mid-episode, so this is read from the live model every step.
  """
  model = env.sim.model
  out = []
  for cfg in (left_cfg, right_cfg):
    asset: Entity = env.scene[cfg.name]
    gids = asset.indexing.geom_ids[cfg.geom_ids]
    out.append(model.geom_solref[:, gids, 0].mean(dim=1))
  return torch.stack(out, dim=1).expand(env.num_envs, 2)
