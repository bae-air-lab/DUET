from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def piecewise_linear(step: int | float, knots: Sequence[Sequence[float]]) -> float:
  """Piecewise-linear schedule through ``knots`` = [(step, value), ...].

  Constant before the first knot and after the last. Used by every ramped
  curriculum quantity so a schedule is a list of (iteration, value) pairs in
  the config rather than arithmetic buried in an update function. Scalar
  Python arithmetic: it is evaluated once per step, never per env.
  """
  if not knots:
    raise ValueError("piecewise_linear needs at least one knot")
  if step <= knots[0][0]:
    return float(knots[0][1])
  for (s0, v0), (s1, v1) in zip(knots[:-1], knots[1:], strict=False):
    if step <= s1:
      if s1 <= s0:
        return float(v1)
      return float(v0 + (v1 - v0) * (step - s0) / (s1 - s0))
  return float(knots[-1][1])


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


class CommandMixStage(TypedDict, total=False):
  step: int
  rel_standing_envs: float
  walk_min_height: float
  nominal_height_fraction: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
  ramp_steps: int = 0,
) -> dict[str, torch.Tensor]:
  """Staged velocity-command envelope.

  ``ramp_steps`` > 0 blends each stage in linearly over that many policy steps
  after its ``step`` instead of switching at once. Why: in the first
  arm-robustness run the 1500-iteration switch (every axis roughly doubled in
  one iteration, on top of the arm ramp starting) produced a -16 return drop, a
  fall spike and the onset of the action-std runaway within 250 iterations. A
  blended envelope gives the value function a moving target it can follow.
  ``ramp_steps=0`` reproduces the original step behaviour.
  """
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  step = env.common_step_counter
  current: dict[str, tuple[float, float] | None] = {
    "lin_vel_x": None, "lin_vel_y": None, "ang_vel_z": None
  }
  for stage in velocity_stages:
    if step <= stage["step"] and stage["step"] > 0:
      continue
    alpha = 1.0
    if ramp_steps > 0 and stage["step"] > 0:
      alpha = min(1.0, max(0.0, (step - stage["step"]) / ramp_steps))
    for axis in current:
      target = stage.get(axis)
      if target is None:
        continue
      prev = current[axis]
      if prev is None or alpha >= 1.0:
        current[axis] = tuple(target)
      else:
        current[axis] = (
          prev[0] + alpha * (target[0] - prev[0]),
          prev[1] + alpha * (target[1] - prev[1]),
        )
  for axis, rng in current.items():
    if rng is not None:
      setattr(cfg.ranges, axis, rng)
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def command_mix(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  twist_command_name: str,
  height_command_name: str,
  stages: list[CommandMixStage],
) -> torch.Tensor:
  """Curriculum by command distribution (walk-only -> squat -> mixed).

  Each stage adjusts (a) ``rel_standing_envs`` on the twist command -- the
  fraction of envs that stand still (where deep squats happen) -- (b)
  ``walk_min_height`` on the base-height command -- how deep the robot may
  squat *while walking* -- and optionally (c) ``nominal_height_fraction``: the
  fraction of envs whose height command is pinned at the nominal standing
  height. (c) is the height half of the mixture curriculum: it keeps a slice of
  nominal-height envs alive all the way through training. All cfg fields are
  read at resample time, so mutating them takes effect on the next resample.
  """
  del env_ids  # Unused.
  twist_term = env.command_manager.get_term(twist_command_name)
  height_term = env.command_manager.get_term(height_command_name)
  assert twist_term is not None and height_term is not None
  twist_cfg = cast(UniformVelocityCommandCfg, twist_term.cfg)
  for stage in stages:
    if env.common_step_counter > stage["step"]:
      if "rel_standing_envs" in stage:
        twist_cfg.rel_standing_envs = stage["rel_standing_envs"]
      if "walk_min_height" in stage:
        height_term.cfg.walk_min_height = stage["walk_min_height"]
      if "nominal_height_fraction" in stage:
        height_term.cfg.nominal_env_fraction = stage["nominal_height_fraction"]
  return torch.tensor([height_term.cfg.walk_min_height])


def push_magnitude(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  base_velocity_range: dict[str, tuple[float, float]],
  scale_stages: Sequence[Sequence[float]],
) -> torch.Tensor:
  """Ramp an interval push event's ``velocity_range`` over training.

  ``scale_stages`` are (policy_step, scale) knots, linearly interpolated; the
  event's range is ``scale * base_velocity_range`` on every axis. The base
  range is passed explicitly (not read back from the event) so the term is
  stateless and idempotent. Pushes start weak so the locomotion foundation is
  learned before disturbance rejection, and reach full strength when the rest
  of the task distribution does.
  """
  del env_ids  # Unused.
  scale = piecewise_linear(env.common_step_counter, scale_stages)
  term_cfg = env.event_manager.get_term_cfg(event_name)
  term_cfg.params["velocity_range"] = {
    k: (scale * lo, scale * hi) for k, (lo, hi) in base_velocity_range.items()
  }
  return torch.tensor([scale])


def arm_curriculum_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  action_term_name: str = "upper_body_pose",
) -> torch.Tensor:
  """Log-only term: the arm generator's current curriculum ratio.

  The ratio itself is computed inside the action term (it reads the same
  checkpointed step counter); this term only surfaces it on the training
  curves as ``Curriculum/arm_curriculum_state``.
  """
  del env_ids  # Unused.
  term = env.action_manager.get_term(action_term_name)
  return torch.tensor([float(getattr(term, "curriculum_ratio", 0.0))])


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
