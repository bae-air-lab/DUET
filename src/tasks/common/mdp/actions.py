"""Action terms specific to the loco-manipulation task.

The upper body (arms) is carved out of the RL policy's action space: the policy
controls only the lower body, while the arm joints are driven externally. During
training they follow a randomized "upper-body pose curriculum" (random targets
whose magnitude ramps from 0 -> 1 over training), so the lower body learns to
stay balanced under arbitrary arm motion (ExBody / HOMIE style). At deployment the
external targets are supplied by the imitation-learning manipulation policy instead.

The term consumes ZERO dimensions of the policy action vector: it is a passive
action term whose ``apply_actions`` writes joint position targets from an internal
buffer that is advanced (sampled + interpolated) once per policy step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["UpperBodyPoseActionCfg", "UpperBodyPoseAction"]


@dataclass(kw_only=True)
class UpperBodyPoseActionCfg(ActionTermCfg):
  """Configuration for the externally-driven upper-body (arm) joint targets.

  This term does not consume any policy actions (``action_dim == 0``). It holds
  position targets for the matched joints and applies them every physics substep.
  """

  joint_names: tuple[str, ...]
  """Regex patterns (matched against joint names) for the joints to drive."""

  enabled: bool = True
  """Ablation switch for HOMIE contribution (a). When False the arms are held at
  their default pose for the whole run, so the lower body never sees an
  upper-body disturbance. Used by the ``Duet-Abl-NoArmCurriculum`` variant."""

  resample_steps: int = 50
  """Number of policy steps between sampling new random goal poses. With the
  default control dt (0.02s) this is ~1s, matching HOMIE's upper_interval_s."""

  init_ratio: float = 0.0
  """Initial curriculum ratio (0 = arms held at default pose)."""

  ratio_curriculum_steps: int = 100_000
  """Policy steps over which the curriculum ratio ramps from ``init_ratio`` to 1.0.
  Larger = slower introduction of arm motion."""

  sample_range_scale: float = 0.6
  """Fraction of each joint's available range (on each side of the default pose)
  that goal poses are sampled from. < 1.0 keeps the arms in a realistic reach
  workspace instead of flailing to the joint limits -- this matches the IL
  manipulation distribution and avoids gratuitous arm self-collision and angular
  momentum injection that hurts locomotion."""

  task_poses: tuple[dict[str, float], ...] = ()
  """Task-pose anchors: each entry maps joint-name patterns to target angles (rad)
  and must cover every driven joint. When non-empty, a fraction
  ``task_pose_prob`` of resampled goals is drawn as (anchor + uniform noise)
  instead of uniformly over the workspace, focusing training on the poses the
  robot will actually hold at deployment (box carry, forward reach, low/high
  pick) rather than arbitrary flailing."""

  task_pose_prob: float = 0.0
  """Probability (per env, per resample) that the new goal is a task-pose anchor
  rather than a uniform workspace sample. The uniform remainder preserves
  robustness to out-of-distribution arm motion."""

  task_pose_noise: dict[str, float] | float = 0.15
  """Half-width (rad) of the uniform noise added around a task-pose anchor,
  either shared or per joint pattern. Keeps a dense cloud of poses around each
  anchor so the policy generalizes across grasp heights/widths."""

  velocity_command_name: str = "twist"
  velocity_threshold: float = 0.1
  walking_motion_scale: float = 0.15
  """How much of the full arm-motion *rate* is allowed while the robot is walking
  (|twist| > velocity_threshold). Standing still -> full motion (1.0) for IL-style
  manipulation. Walking -> arms move slowly (calm), so the legs do not have to
  reject a thrashing upper body (this is what restores Flat-policy walking
  robustness). The arms still drift across the whole pose workspace, so the policy
  stays robust to walking while holding the arm at any pose (e.g. carrying an
  object) -- it just never sees violent arm motion while walking."""

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyPoseAction:
    return UpperBodyPoseAction(self, env)


class UpperBodyPoseAction(ActionTerm):
  """Drive a subset of joints to externally-specified position targets.

  Training: random goal poses within (curriculum-scaled) joint limits, linearly
  interpolated over ``resample_steps``. Deployment: call :meth:`set_external_target`
  every control step with the manipulation policy's joint targets.
  """

  cfg: UpperBodyPoseActionCfg

  def __init__(self, cfg: UpperBodyPoseActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, joint_names = self._entity.find_joints(cfg.joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = joint_names
    self._num_joints = len(joint_ids)

    default = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._joint_ids, :]
    self._default = default
    self._lo = soft_limits[..., 0]
    self._hi = soft_limits[..., 1]

    # Task-pose anchors (deployment-relevant poses), resolved to joint order.
    self._task_anchors: torch.Tensor | None = None
    self._task_noise: torch.Tensor | None = None
    if cfg.task_poses:
      anchors = []
      for pose in cfg.task_poses:
        _, matched, values = resolve_matching_names_values(
          data=dict(pose), list_of_strings=joint_names
        )
        if len(matched) != self._num_joints:
          missing = set(joint_names) - set(matched)
          raise ValueError(
            f"Task pose {pose} does not cover all driven joints; missing {missing}."
          )
        anchors.append(torch.tensor(values, device=self.device, dtype=torch.float32))
      self._task_anchors = torch.stack(anchors)  # [K, J]
      if isinstance(cfg.task_pose_noise, dict):
        _, _, noise_values = resolve_matching_names_values(
          data=dict(cfg.task_pose_noise), list_of_strings=joint_names
        )
        self._task_noise = torch.tensor(
          noise_values, device=self.device, dtype=torch.float32
        )
      else:
        self._task_noise = torch.full(
          (self._num_joints,), float(cfg.task_pose_noise), device=self.device
        )

    # Buffers.
    self._current = default.clone()  # currently applied target
    self._goal = default.clone()  # goal being interpolated toward
    self._delta = torch.zeros_like(default)  # per-step interpolation increment
    self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    # Optional external override (deployment). Shape (num_envs, num_joints) or None.
    self._external_target: torch.Tensor | None = None

  # Properties required by ActionTerm.

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def _step_count(self) -> int:
    """Global policy-step count, read from the environment.

    Deliberately NOT a private counter on this term. ``common_step_counter`` is
    persisted into every checkpoint by ``MjlabOnPolicyRunner.save()`` and
    restored by ``load()``, so keying the curriculum off it makes ``--resume``
    continue the arm-motion ramp where it left off. A term-local counter
    restarts at zero in a new process and silently re-softens the task for
    thousands of iterations -- the bug that ``RESUME_AT_FULL_DISTRIBUTION`` used
    to paper over.
    """
    return int(self._env.common_step_counter)

  @property
  def curriculum_ratio(self) -> float:
    frac = self._step_count / max(self.cfg.ratio_curriculum_steps, 1)
    return float(min(1.0, self.cfg.init_ratio + frac))

  # Deployment hook.

  def set_external_target(self, target: torch.Tensor | None) -> None:
    """Override the arm targets (e.g. with IL policy output). Pass None to
    return to the training-time curriculum."""
    self._external_target = target

  # ActionTerm interface.

  def process_actions(self, actions: torch.Tensor) -> None:
    """Advance the arm targets once per policy step (actions is empty)."""
    del actions  # zero-width slice
    if self._external_target is not None:
      self._current = self._external_target
      return

    if not self.cfg.enabled:
      return  # ablation: arms pinned at the default pose, no disturbance
    if self._step_count % self.cfg.resample_steps == 0:
      self._resample_goal()

    # Calm the arm motion while walking: scale the per-step increment per-env by
    # whether the robot is (near) standing. Standing -> full motion; walking ->
    # slow drift, so walking is not disturbed by violent arm motion.
    scale = 1.0
    twist = self._env.command_manager.get_command(self.cfg.velocity_command_name)
    if twist is not None:
      speed = torch.norm(twist[:, :3], dim=1, keepdim=True)
      moving = speed > self.cfg.velocity_threshold
      scale = torch.where(
        moving,
        torch.full_like(speed, self.cfg.walking_motion_scale),
        torch.ones_like(speed),
      )
    self._current = self._current + self._delta * scale

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._current, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._current[env_ids] = self._default[env_ids]
    self._goal[env_ids] = self._default[env_ids]
    self._delta[env_ids] = 0.0

  # Internal.

  def _resample_goal(self) -> None:
    ratio = self.curriculum_ratio
    # Restrict sampling to a realistic workspace around the default pose.
    s = self.cfg.sample_range_scale
    lo = self._default - s * (self._default - self._lo)
    hi = self._default + s * (self._hi - self._default)
    u = torch.rand(self.num_envs, self._num_joints, device=self.device)
    sampled = lo + u * (hi - lo)
    # Mix in task-pose anchors: with prob task_pose_prob an env's goal is a
    # deployment pose (box carry / reach / pick) + noise instead of a uniform
    # workspace sample. Anchors may sit outside the scaled workspace (e.g. a
    # full forward reach), so they are clamped to the soft limits only.
    if self._task_anchors is not None and self.cfg.task_pose_prob > 0.0:
      assert self._task_noise is not None
      k = torch.randint(
        self._task_anchors.shape[0], (self.num_envs,), device=self.device
      )
      anchor = self._task_anchors[k]  # [B, J]
      noise = (torch.rand_like(anchor) * 2.0 - 1.0) * self._task_noise
      task_goal = torch.clamp(anchor + noise, self._lo, self._hi)
      use_task = (
        torch.rand(self.num_envs, 1, device=self.device) < self.cfg.task_pose_prob
      )
      sampled = torch.where(use_task, task_goal, sampled)
    # ratio scales how far from the default pose the goal can deviate.
    self._goal = self._default + ratio * (sampled - self._default)
    self._delta = (self._goal - self._current) / self.cfg.resample_steps
