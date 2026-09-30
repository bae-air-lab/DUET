"""Event terms specific to the loco-manipulation task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

# _randomize_model_field is mjlab's private DR engine (every public dr.geom_*
# function is a thin wrapper around it). Imported anyway: mjlab is pinned at
# 1.2.0 in requirements-train.txt, and reusing the engine keeps geom_solref's
# indexing, shared_random and operation semantics identical to geom_friction.
from mjlab.envs.mdp.dr._core import _randomize_model_field
from mjlab.envs.mdp.dr._types import Distribution, Operation
from mjlab.envs.mdp.events import push_by_setting_velocity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

__all__ = ["push_and_relatch_anchor", "geom_solref"]


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


@requires_model_fields("geom_solref")
def geom_solref(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float] | dict[int, tuple[float, float]],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Distribution | str = "uniform",
  operation: Operation | str = "abs",
  axes: list[int] | None = None,
  shared_random: bool = False,
) -> None:
  """Randomize geom contact softness, modelled on ``mjlab``'s ``geom_friction``.

  ``geom_solref`` is ``(timeconst, dampratio)`` per geom. Default axis 0 (the
  time constant); axis 1 (damping ratio) only when asked for.

  Why the FOOT geoms. They carry ``priority=1`` and the terrain 0, and with
  unequal priorities MuJoCo uses the higher-priority geom's ``solref`` outright
  (no mixing), so randomising the foot is what changes foot-ground contact. The
  static penetration of a soft contact is ``r = a_u (1-d) timeconst^2
  dampratio^2``: it grows with the SQUARE of the time constant, which is why
  MuJoCo's default 0.02 s is effectively rigid (sub-millimetre sinkage).

  Writing this field mid-episode (``interval`` mode) is safe with CUDA graphs:
  the event manager expands the field per world once at startup (that is what
  re-captures the graph); later calls write in place into the same arrays.
  """
  _randomize_model_field(
    env,
    env_ids,
    "geom_solref",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation=operation,
    asset_cfg=asset_cfg,
    axes=axes,
    shared_random=shared_random,
    default_axes=[0],
    valid_axes=[0, 1],
  )
