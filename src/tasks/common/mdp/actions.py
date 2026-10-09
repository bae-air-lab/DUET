"""Action terms specific to the loco-manipulation task.

The upper body (arms) is carved out of the RL policy's action space: the policy
controls only the lower body, while the arm joints are driven externally. During
training they follow a randomized **acceleration-limited trajectory generator**
(see :class:`UpperBodyPoseAction`), so the lower body learns to stay balanced
under arbitrary, asynchronous, asymmetric arm motion. At deployment the external
targets are supplied by whatever drives ``rt/arm_targets`` (a VLA, teleop, a
script) -- the balance policy is agnostic to the source.

The term consumes ZERO dimensions of the policy action vector: it is a passive
action term whose ``apply_actions`` writes joint position targets from an
internal buffer that is advanced once per policy step.

Why a trapezoidal generator instead of linear interpolation to a goal that every
env changed in lock-step (the previous design):

* A VLA emits joint targets that accelerate, cruise and decelerate. Linear
  interpolation has no acceleration phase, so the policy never saw the momentum
  transients that actually knock a standing robot over.
* A global ``step % resample_steps`` timer made every env change target on the
  same step, which correlates the disturbance across the whole batch and makes
  its timing predictable. Timing is now a per-env random variable.
* The old generator slowed the arms to 15% while walking. That protected the
  legs from exactly the disturbance the deployed policy meets (a manipulation
  policy does not know the legs are walking). The arm distribution here is
  independent of the locomotion command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .curriculums import piecewise_linear

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["UpperBodyPoseActionCfg", "UpperBodyPoseAction"]

# Curriculum knots are (policy_step, value) pairs, linearly interpolated.
Knots = tuple[tuple[int, float], ...]


@dataclass(kw_only=True)
class UpperBodyPoseActionCfg(ActionTermCfg):
  """Configuration for the externally-driven upper-body (arm) joint targets.

  This term does not consume any policy actions (``action_dim == 0``). It holds
  position targets for the matched joints and applies them every physics substep.
  """

  joint_names: tuple[str, ...]
  """Regex patterns (matched against joint names) for the joints to drive."""

  enabled: bool = True
  """Ablation switch. When False the arms are held at their default pose for
  the whole run, so the lower body never sees an upper-body disturbance. Used
  by the ``Duet-Abl-NoArmCurriculum`` variant."""

  # -- Workspace -------------------------------------------------------------

  workspace_limits: dict[str, tuple[float, float]] | None = None
  """Absolute safe joint ranges (rad) per joint-name pattern. Goals are sampled
  inside these ranges (intersected with the soft joint limits). ``None`` falls
  back to the soft joint limits alone. Use this to keep independently sampled
  left/right targets mechanically reasonable (no arm swung through the torso)
  without hand-picking poses."""

  sample_range_scale: float = 1.0
  """Fraction of the safe workspace (on each side of the default pose) that
  goals may occupy once the curriculum is at full strength. This is the
  ``arm_workspace_scale`` of the final task distribution."""

  # -- Trajectory dynamics ---------------------------------------------------

  traj_vmax_range: tuple[float, float] = (0.3, 3.5)
  """[min, max] of the per-joint peak velocity (rad/s) sampled for each segment
  at full curriculum strength."""

  traj_vmax_range_init: tuple[float, float] = (0.2, 0.8)
  """Same, at curriculum ratio 0. The range is linearly interpolated with the
  curriculum ratio between ``_init`` and the full value."""

  traj_amax_range: tuple[float, float] = (1.0, 20.0)
  """[min, max] of the per-joint acceleration limit (rad/s^2) at full strength."""

  traj_amax_range_init: tuple[float, float] = (0.5, 3.0)
  """Same, at curriculum ratio 0."""

  hold_prob: float = 0.5
  """Probability that a reached goal is held for a random duration before the
  next goal is drawn (otherwise the next goal follows immediately, which is
  what a continuously moving manipulation policy looks like)."""

  hold_time_range: tuple[float, float] = (0.2, 2.5)
  """[min, max] hold duration (s) when a hold is drawn."""

  stationary_arm_prob: float = 0.2
  """Per arm, per goal draw: probability that this arm keeps its current pose
  while the other one moves. Gives the "one arm moving, one arm still"
  configurations independently of the joint-level sampling."""

  max_segment_time: float = 8.0
  """Safety valve: a segment that has not reached its goal after this long
  (very slow draw toward a far goal) is re-drawn, so every env keeps sampling
  new configurations at a reasonable rate."""

  goal_tolerance: float = 0.01
  """Position (rad) and velocity (rad/s) tolerance for "goal reached"."""

  # -- Curriculum ------------------------------------------------------------

  init_ratio: float = 0.0
  """Constant added to the scheduled ratio. ``1.0`` pins the generator at the
  full distribution regardless of the step counter (used by the evaluation
  protocol and the play config)."""

  ratio_stages: Knots = ((0, 0.0), (100_000, 1.0))
  """(policy_step, ratio) knots. The curriculum ratio is the piecewise-linear
  interpolation of these, plus ``init_ratio``, clamped to [0, 1]. It scales
  the workspace (0 -> goals at the default pose) and interpolates the
  velocity/acceleration ranges from ``_init`` to full."""

  clean_fraction_stages: Knots = ((0, 0.0),)
  """(policy_step, fraction) knots: the fraction of envs that run a "clean"
  (mild-arm) condition at that point in training. Mixture curriculum: keeping
  a slice of clean locomotion envs alive throughout training stops the
  robustness training from degrading gait quality. Each env draws its slot
  uniformly at reset, so membership is per-env and re-drawn every episode."""

  clean_env_ratio: float = 0.1
  """Curriculum ratio applied to the clean slice (capped by the global ratio)."""

  # -- Optional task-pose anchors (minority) ---------------------------------

  task_poses: tuple[dict[str, float], ...] = ()
  """Task-pose anchors: each entry maps joint-name patterns to target angles
  (rad) and must cover every driven joint. A fraction ``task_pose_prob`` of
  goals is drawn as (anchor + independent per-joint noise) instead of uniformly
  over the workspace. Kept as a minority option; the default task distribution
  is uniform over the safe workspace so the policy is not biased toward a
  handful of symmetric poses."""

  task_pose_prob: float = 0.0
  """Probability (per env, per goal draw) that the goal is a task-pose anchor."""

  task_pose_noise: dict[str, float] | float = 0.15
  """Half-width (rad) of the uniform noise around a task-pose anchor, either
  shared or per joint pattern. Sampled independently per joint, so anchored
  goals are asymmetric too."""

  # -- Optional squat-coupled reach poses -------------------------------------

  squat_reach_poses: tuple[dict[str, float], ...] = ()
  """Reach anchors used while the robot is squatting: an env whose height
  command is below ``squat_reach_below`` draws its goal from these (plus
  ``task_pose_noise``) with probability ``squat_reach_prob``. Couples arm
  extension to squat depth, which the independent per-joint draw almost never
  produces (deep squat + both arms forward was <1% of training time)."""

  squat_reach_prob: float = 0.0
  """Probability (per squatting env, per goal draw) of a squat-reach goal."""

  squat_reach_below: float = 0.45
  """Height command (m) below which an env counts as squatting."""

  height_command_name: str = "base_height"

  def build(self, env: ManagerBasedRlEnv) -> UpperBodyPoseAction:
    return UpperBodyPoseAction(self, env)


class UpperBodyPoseAction(ActionTerm):
  """Drive a subset of joints along random acceleration-limited trajectories.

  Per env and per joint the term keeps a goal, a trajectory position, a
  trajectory velocity, a peak velocity and an acceleration limit. Each policy
  step it advances the trajectory with the classic stopping-distance rule::

      v_stop = sqrt(2 * a_max * |goal - pos|)
      v_des  = sign(goal - pos) * min(v_max, v_stop)
      vel   += clamp(v_des - vel, -a_max*dt, +a_max*dt)
      pos   += vel * dt

  which yields trapezoidal profiles for long moves and triangular ones for
  short moves, with no overshoot (the step that would cross the goal snaps to
  it). Goal changes are driven by per-env timers, so envs are never in phase.

  Deployment: call :meth:`set_external_target` every control step with the
  manipulation policy's joint targets; the trajectory velocity is then the
  finite difference of those targets, so observers of ``traj_vel`` see the
  same quantity in both regimes.
  """

  cfg: UpperBodyPoseActionCfg

  def __init__(self, cfg: UpperBodyPoseActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, joint_names = self._entity.find_joints(cfg.joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = joint_names
    self._num_joints = len(joint_ids)
    self._dt = float(env.step_dt)

    default = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    soft_limits = self._entity.data.soft_joint_pos_limits[:, self._joint_ids, :]
    self._default = default  # [B, J]
    lo = soft_limits[..., 0].clone()
    hi = soft_limits[..., 1].clone()
    if cfg.workspace_limits is not None:
      _, matched, values = resolve_matching_names_values(
        data=dict(cfg.workspace_limits), list_of_strings=joint_names
      )
      if len(matched) != self._num_joints:
        missing = set(joint_names) - set(matched)
        raise ValueError(f"workspace_limits does not cover joints {missing}.")
      ws = torch.tensor(values, device=self.device, dtype=torch.float32)  # [J, 2]
      lo = torch.maximum(lo, ws[:, 0].unsqueeze(0))
      hi = torch.minimum(hi, ws[:, 1].unsqueeze(0))
    if bool((lo > default).any()) or bool((hi < default).any()):
      raise ValueError("Arm workspace must contain the default pose.")
    self._lo = lo  # [B, J] safe workspace (already inside the soft limits)
    self._hi = hi

    # Arm grouping (left / right) for the stationary-arm draw. Joints that
    # match neither prefix form their own group.
    groups = []
    for n in joint_names:
      groups.append(0 if n.startswith("left_") else 1 if n.startswith("right_") else 2)
    self._arm_group = torch.tensor(groups, device=self.device, dtype=torch.long)
    self._num_groups = int(self._arm_group.max().item()) + 1  # init-time only

    # Task-pose and squat-reach anchors, resolved to joint order.
    def _resolve_anchors(poses) -> torch.Tensor:
      anchors = []
      for pose in poses:
        _, matched, values = resolve_matching_names_values(
          data=dict(pose), list_of_strings=joint_names
        )
        if len(matched) != self._num_joints:
          missing = set(joint_names) - set(matched)
          raise ValueError(
            f"Task pose {pose} does not cover all driven joints; missing {missing}."
          )
        anchors.append(torch.tensor(values, device=self.device, dtype=torch.float32))
      return torch.stack(anchors)  # [K, J]

    self._task_anchors: torch.Tensor | None = None
    self._squat_anchors: torch.Tensor | None = None
    self._task_noise: torch.Tensor | None = None
    if cfg.task_poses:
      self._task_anchors = _resolve_anchors(cfg.task_poses)
    if cfg.squat_reach_poses:
      self._squat_anchors = _resolve_anchors(cfg.squat_reach_poses)
    if cfg.task_poses or cfg.squat_reach_poses:
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

    # Per-env, per-joint trajectory state (all GPU-resident).
    B, J = self.num_envs, self._num_joints
    self._goal = default.clone()
    self._traj_pos = default.clone()  # the target actually applied
    self._traj_vel = torch.zeros(B, J, device=self.device)
    self._traj_acc = torch.zeros(B, J, device=self.device)
    self._vmax = torch.full((B, J), cfg.traj_vmax_range_init[0], device=self.device)
    self._amax = torch.full((B, J), cfg.traj_amax_range_init[0], device=self.device)
    # Per-env timers.
    self._holding = torch.ones(B, dtype=torch.bool, device=self.device)
    self._hold_timer = torch.zeros(B, device=self.device)
    self._segment_time = torch.zeros(B, device=self.device)
    # Per-env mixture slot in [0, 1): env is "clean" iff slot < clean fraction.
    self._env_slot = torch.rand(B, device=self.device)
    self._raw_actions = torch.zeros(B, 0, device=self.device)

    # Optional external override (deployment). Shape (num_envs, num_joints) or None.
    self._external_target: torch.Tensor | None = None

  # -- Properties required by ActionTerm ------------------------------------

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  # -- Public state (read by observations, metrics, eval scripts) -----------

  @property
  def joint_names(self) -> list[str]:
    return list(self._joint_names)

  @property
  def joint_ids(self) -> torch.Tensor:
    return self._joint_ids

  @property
  def default_pose(self) -> torch.Tensor:
    return self._default

  @property
  def traj_pos(self) -> torch.Tensor:
    """Currently applied arm target. Shape (num_envs, num_joints)."""
    return self._traj_pos

  @property
  def traj_vel(self) -> torch.Tensor:
    """Target velocity (rad/s): the generator's, or the finite difference of
    external targets. Shape (num_envs, num_joints)."""
    return self._traj_vel

  @property
  def traj_acc(self) -> torch.Tensor:
    return self._traj_acc

  @property
  def goal(self) -> torch.Tensor:
    return self._goal

  @property
  def _step_count(self) -> int:
    """Global policy-step count, read from the environment.

    Deliberately NOT a private counter on this term. ``common_step_counter`` is
    persisted into every checkpoint by ``MjlabOnPolicyRunner.save()`` and
    restored by ``load()``, so keying the curriculum off it makes ``--resume``
    continue the arm-motion ramp where it left off.
    """
    return int(self._env.common_step_counter)

  @property
  def curriculum_ratio(self) -> float:
    r = self.cfg.init_ratio + piecewise_linear(self._step_count, self.cfg.ratio_stages)
    return float(min(1.0, max(0.0, r)))

  @property
  def clean_fraction(self) -> float:
    if self.cfg.init_ratio >= 1.0:
      # Pinned at the full distribution (evaluation): use the final fraction.
      return float(self.cfg.clean_fraction_stages[-1][1])
    return float(piecewise_linear(self._step_count, self.cfg.clean_fraction_stages))

  # -- Deployment hook --------------------------------------------------------

  def set_external_target(self, target: torch.Tensor | None) -> None:
    """Override the arm targets (e.g. with a VLA's output). Pass None to
    return to the training-time generator."""
    self._external_target = target

  # -- ActionTerm interface ---------------------------------------------------

  def process_actions(self, actions: torch.Tensor) -> None:
    """Advance the arm targets once per policy step (actions is empty)."""
    del actions  # zero-width slice
    if self._external_target is not None:
      target = self._external_target.to(self._traj_pos)
      vel = (target - self._traj_pos) / self._dt
      self._traj_acc = (vel - self._traj_vel) / self._dt
      self._traj_vel = vel
      self._traj_pos = target.clone()
      return
    if not self.cfg.enabled:
      return  # ablation: arms pinned at the default pose, no disturbance
    self._update_timers()
    self._advance_trajectory()

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._traj_pos, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    # The robot is reset to its default joint pose, so starting the trajectory
    # at the default with zero velocity means the first applied target equals
    # the joint state: no first-step jump.
    self._traj_pos[env_ids] = self._default[env_ids]
    self._goal[env_ids] = self._default[env_ids]
    self._traj_vel[env_ids] = 0.0
    self._traj_acc[env_ids] = 0.0
    # Hold the default pose for a random short time so the first arm motion is
    # not synchronised with the episode start.
    self._holding[env_ids] = True
    self._hold_timer[env_ids] = (
      torch.rand_like(self._hold_timer[env_ids]) * self.cfg.hold_time_range[1]
    )
    self._segment_time[env_ids] = 0.0
    self._env_slot[env_ids] = torch.rand_like(self._env_slot[env_ids])

  # -- Internal ---------------------------------------------------------------

  def _update_timers(self) -> None:
    """Per-env goal state machine: move -> (hold) -> re-draw."""
    cfg = self.cfg
    tol = cfg.goal_tolerance
    reached = ((self._goal - self._traj_pos).abs() <= tol).all(dim=1) & (
      self._traj_vel.abs() <= tol
    ).all(dim=1)
    newly_reached = reached & ~self._holding
    # Draw hold durations for every env (vectorised); keep them only where a
    # goal was just reached. Zero hold = the next goal follows immediately.
    u = torch.rand(self.num_envs, device=self.device)
    lo, hi = cfg.hold_time_range
    hold = torch.where(
      u < cfg.hold_prob,
      lo + torch.rand(self.num_envs, device=self.device) * (hi - lo),
      torch.zeros_like(u),
    )
    self._hold_timer = torch.where(newly_reached, hold, self._hold_timer - self._dt)
    self._holding = self._holding | newly_reached
    self._segment_time = self._segment_time + self._dt
    redraw = (self._holding & (self._hold_timer <= 0.0)) | (
      self._segment_time >= cfg.max_segment_time
    )
    self._resample_goal(redraw)

  def _advance_trajectory(self) -> None:
    dt = self._dt
    err = self._goal - self._traj_pos
    # Stopping-distance velocity, in its discrete-time form: the largest speed
    # from which a deceleration of a_max per step still stops exactly at the
    # goal, v_stop = sqrt((a dt/2)^2 + 2 a |e|) - a dt/2 (the continuous
    # sqrt(2 a |e|) lets the sampled trajectory arrive with a residual speed of
    # up to ~10 a dt and forces a hard stop). Capping at |e|/dt never commands
    # more than what reaches the goal this step. Together these bound the
    # residual speed at arrival by 2 a dt. The epsilon keeps sqrt finite at 0.
    h = 0.5 * self._amax * dt
    v_stop = torch.sqrt(h * h + 2.0 * self._amax * err.abs() + 1e-12) - h
    v_des = torch.sign(err) * torch.minimum(
      torch.minimum(self._vmax, v_stop), err.abs() / dt
    )
    dv_max = self._amax * dt
    dv = (v_des - self._traj_vel).clamp(-dv_max, dv_max)
    vel = self._traj_vel + dv
    pos = self._traj_pos + vel * dt
    # No overshoot: a step that would cross (or exactly land on) the goal snaps
    # to it and stops. This is also what makes "reached" exact.
    crossed = (self._goal - pos) * err <= 0.0
    pos = torch.where(crossed, self._goal, pos)
    vel = torch.where(crossed, torch.zeros_like(vel), vel)
    self._traj_acc = (vel - self._traj_vel) / dt
    self._traj_vel = vel
    self._traj_pos = torch.minimum(torch.maximum(pos, self._lo), self._hi)

  def _resample_goal(self, mask: torch.Tensor) -> None:
    """Draw new goals + dynamics for the envs in ``mask`` (bool, [B])."""
    cfg = self.cfg
    B, J = self.num_envs, self._num_joints
    dev = self.device

    # Per-env curriculum ratio: the clean slice is capped at clean_env_ratio.
    ratio = self.curriculum_ratio
    clean = self._env_slot < self.clean_fraction
    ratio_env = torch.where(
      clean,
      torch.full((B,), min(ratio, cfg.clean_env_ratio), device=dev),
      torch.full((B,), ratio, device=dev),
    ).unsqueeze(1)  # [B, 1]

    # Workspace: ratio * sample_range_scale of the safe range on each side of
    # the default pose. Sampled independently per joint, so left and right
    # arms (and every joint within an arm) are uncorrelated.
    s = ratio_env * cfg.sample_range_scale
    lo = self._default - s * (self._default - self._lo)
    hi = self._default + s * (self._hi - self._default)
    goal = lo + torch.rand(B, J, device=dev) * (hi - lo)

    # Minority task-pose anchors, with independent per-joint noise, clamped to
    # the current (curriculum-scaled) workspace.
    if self._task_anchors is not None and cfg.task_pose_prob > 0.0:
      assert self._task_noise is not None
      k = torch.randint(self._task_anchors.shape[0], (B,), device=dev)
      anchor = self._task_anchors[k]
      noise = (torch.rand(B, J, device=dev) * 2.0 - 1.0) * self._task_noise
      task_goal = torch.minimum(torch.maximum(anchor + noise, lo), hi)
      use_task = torch.rand(B, 1, device=dev) < cfg.task_pose_prob
      goal = torch.where(use_task, task_goal, goal)

    # Squat-coupled reach anchors for envs currently commanded into a squat.
    cmd_mgr = getattr(self._env, "command_manager", None)
    if self._squat_anchors is not None and cfg.squat_reach_prob > 0.0 and cmd_mgr is not None:
      assert self._task_noise is not None
      height = cmd_mgr.get_command(cfg.height_command_name)[:, 0]
      k = torch.randint(self._squat_anchors.shape[0], (B,), device=dev)
      noise = (torch.rand(B, J, device=dev) * 2.0 - 1.0) * self._task_noise
      squat_goal = torch.minimum(torch.maximum(self._squat_anchors[k] + noise, lo), hi)
      use_squat = (height < cfg.squat_reach_below).unsqueeze(1) & (
        torch.rand(B, 1, device=dev) < cfg.squat_reach_prob
      )
      goal = torch.where(use_squat, squat_goal, goal)

    # Stationary arm: with some probability an arm keeps its current pose while
    # the other one moves.
    stat_group = torch.rand(B, self._num_groups, device=dev) < cfg.stationary_arm_prob
    stat = stat_group[:, self._arm_group]  # [B, J]
    goal = torch.where(stat, self._traj_pos, goal)

    # Segment dynamics: peak velocity and acceleration limit per joint, drawn
    # from ranges that open up with the curriculum ratio.
    def _lerp_range(init, full):
      lo_r = init[0] + ratio_env * (full[0] - init[0])
      hi_r = init[1] + ratio_env * (full[1] - init[1])
      return lo_r + torch.rand(B, J, device=dev) * (hi_r - lo_r)

    vmax = _lerp_range(cfg.traj_vmax_range_init, cfg.traj_vmax_range)
    amax = _lerp_range(cfg.traj_amax_range_init, cfg.traj_amax_range)

    m = mask.unsqueeze(1)
    self._goal = torch.where(m, goal, self._goal)
    self._vmax = torch.where(m, vmax, self._vmax)
    self._amax = torch.where(m, amax, self._amax)
    self._holding = torch.where(mask, torch.zeros_like(mask), self._holding)
    self._segment_time = torch.where(
      mask, torch.zeros_like(self._segment_time), self._segment_time
    )

