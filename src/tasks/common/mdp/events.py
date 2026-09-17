"""Event terms specific to the loco-manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

__all__ = ["push_and_relatch_anchor"]


def push_and_relatch_anchor(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  velocity_range: dict[str, tuple[float, float]],
  anchor_term_name: str = "idle_position_anchor",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """``push_by_setting_velocity`` that also re-latches the idle position anchor.

  Why. The anchor latches the base xy once, when the command goes idle, and
  charges displacement from that point for the rest of the idle window. A
  push is a velocity impulse the policy cannot undo without stepping: at the
  final 0.7 m/s it carries a standing robot 15-25 cm. Measured in the first
  arm-robustness run, mean idle drift went 0.07 m -> 0.52 m and the anchor
  cost 0.05 -> 0.9 per second as the push scale ramped 0.1 -> 1.0, i.e. the
  term was mostly charging for pushes, an error the policy cannot reduce and
  which the idle feet-still term forbids it from correcting by stepping. That
  is pure advantage noise. Re-latching after the push keeps the anchor's
  purpose (no creep while standing) and drops the unlearnable part.
  """
  push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
  if anchor_term_name in env.reward_manager.active_terms:
    env.reward_manager.get_term_cfg(anchor_term_name).func.reset(env_ids)
