"""Command terms specific to the loco-manipulation task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["BaseHeightCommandCfg", "BaseHeightCommand"]


@dataclass(kw_only=True)
class BaseHeightCommandCfg(CommandTermCfg):
  """Command for a target base (pelvis) height above the feet.

  Lets the policy squat to reach low objects while staying balanced. The target
  is an absolute height in meters, sampled uniformly within ``height_range`` and
  held until resampled. In the viewer it can be driven interactively with a
  slider (the sim analog of a gamepad squat/stand control).
  """

  entity_name: str = "robot"
  height_range: tuple[float, float] = (0.20, 0.78)

  enabled: bool = True
  """Ablation switch for HOMIE contribution (b). When False the commanded height
  is pinned at ``height_range[1]`` for every env at all times, i.e. height stops
  being a task variable and the robot only ever stands. The observation slot is
  still emitted (constant), so the 71-D ONNX interface is unchanged and the
  ablated policy remains deployable. Used by ``Duet-Abl-NoHeightCmd``."""

  # Velocity/height decoupling. The robot may WALK at any height in
  # [walk_min_height, top] (comfortable bent-knee band), but a DEEP squat
  # (< walk_min_height) is only commanded while (near) standing still -- it is
  # never asked to walk while deep-folded. Standing-still envs sample the full
  # squat band [floor, top]; moving envs sample the walk band [walk_min, top].
  velocity_command_name: str = "twist"
  velocity_threshold: float = 0.1
  walk_min_height: float = 0.60

  # Squat-depth curriculum: the squat floor starts shallow and deepens to the
  # full height_range[0] over ``floor_curriculum_steps`` policy steps. Essential
  # when the floor is near the kinematic limit (e.g. 0.15 m) -- the policy masters
  # shallow squats first, then progressively deeper ones. Set the start equal to
  # height_range[0] to disable.
  floor_curriculum_start: float = 0.45
  floor_curriculum_steps: int = 100_000

  # Staged alternative to the linear ramp: (policy_step, floor) knots, linearly
  # interpolated (see ``curriculums.piecewise_linear``). When set it replaces
  # the ``floor_curriculum_start/steps`` ramp, EXCEPT that setting
  # ``floor_curriculum_start <= height_range[0]`` still pins the floor at full
  # depth -- that is how the evaluation protocol pins the curriculum, and it
  # must keep working.
  floor_stages: tuple[tuple[int, float], ...] | None = None

  # Mixture curriculum (height half): fraction of envs whose height command is
  # pinned at the nominal standing height ``height_range[1]`` at resample time.
  # Driven over training by ``curriculums.command_mix``; keeps a slice of
  # nominal-height envs alive so height randomisation never crowds out plain
  # standing/walking.
  nominal_env_fraction: float = 0.0

  def build(self, env: ManagerBasedRlEnv) -> BaseHeightCommand:
    return BaseHeightCommand(self, env)


class BaseHeightCommand(CommandTerm):
  cfg: BaseHeightCommandCfg

  def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._height = torch.zeros(self.num_envs, 1, device=self.device)
    # Per-env sampled targets: squat (applied while standing) and walk (applied
    # while moving). Both initialized to the standing height.
    self._squat_target = torch.full(
      (self.num_envs, 1), cfg.height_range[1], device=self.device
    )
    self._walk_target = torch.full(
      (self.num_envs, 1), cfg.height_range[1], device=self.device
    )
    # GUI state (set in create_gui, used in compute).
    self._gui_enabled = None
    self._gui_slider = None
    self._gui_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self._height

  @property
  def _step_count(self) -> int:
    """Global policy-step count, read from the environment.

    See the matching property on ``UpperBodyPoseAction``: ``common_step_counter``
    is checkpointed and restored, a term-local counter is not, so keying the
    squat-depth curriculum off the environment is what makes ``--resume``
    continue the ramp instead of restarting it at the shallow floor.
    """
    return int(self._env.common_step_counter)

  @property
  def current_floor(self) -> float:
    """Squat floor at the current training step (ramps start -> height_range[0])."""
    lo = self.cfg.height_range[0]
    start = self.cfg.floor_curriculum_start
    if start <= lo:
      return lo  # pinned at full depth (evaluation / no-curriculum ablation)
    if self.cfg.floor_stages:
      from .curriculums import piecewise_linear

      return max(lo, piecewise_linear(self._step_count, self.cfg.floor_stages))
    frac = min(1.0, self._step_count / max(self.cfg.floor_curriculum_steps, 1))
    return start + frac * (lo - start)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    hi = self.cfg.height_range[1]
    n = len(env_ids)
    if not self.cfg.enabled:
      self._squat_target[env_ids] = hi
      self._walk_target[env_ids] = hi
      return
    # Standing envs may squat down to the (curriculum) floor.
    self._squat_target[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
      self.current_floor, hi
    )
    # Walking envs use the comfortable bent-knee band [walk_min, top].
    self._walk_target[env_ids] = torch.empty(n, 1, device=self.device).uniform_(
      self.cfg.walk_min_height, hi
    )
    # Mixture slice: some envs stay at the nominal height for the whole
    # command window (both regimes), whatever the curriculum floor is.
    if self.cfg.nominal_env_fraction > 0.0:
      nominal = (
        torch.rand(n, 1, device=self.device) < self.cfg.nominal_env_fraction
      )
      self._squat_target[env_ids] = torch.where(
        nominal, torch.full_like(nominal, hi, dtype=torch.float), self._squat_target[env_ids]
      )
      self._walk_target[env_ids] = torch.where(
        nominal, torch.full_like(nominal, hi, dtype=torch.float), self._walk_target[env_ids]
      )

  def _update_command(self) -> None:
    # Moving -> walk-band height; (near) standing -> squat target (full depth).
    twist = self._env.command_manager.get_command(self.cfg.velocity_command_name)
    if twist is None:
      self._height[:] = self._squat_target
      return
    speed = torch.norm(twist[:, :3], dim=1, keepdim=True)
    moving = speed > self.cfg.velocity_threshold
    self._height = torch.where(moving, self._walk_target, self._squat_target)

  def _update_metrics(self) -> None:
    pass

  # GUI: interactive squat/stand control (mirrors the velocity joystick).

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    lo, hi = self.cfg.height_range
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)
      slider = server.gui.add_slider(
        "height",
        min=lo,
        max=hi,
        step=0.01,
        initial_value=hi,
      )
    self._gui_enabled = enabled
    self._gui_slider = slider
    self._gui_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._gui_enabled is not None and self._gui_enabled.value:
      assert self._gui_get_env_idx is not None and self._gui_slider is not None
      idx = self._gui_get_env_idx()
      self._height[idx, 0] = self._gui_slider.value
