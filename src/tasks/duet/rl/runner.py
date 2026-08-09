"""On-policy runner for the DUET task.

Adds three things to ``MjlabOnPolicyRunner``:

1. **A real entropy schedule** (defect 3). The previous task hardcoded
   ``entropy_coef = 0.004`` -- a fine-tune value -- with a comment telling you
   to hand-edit it for fresh runs. Here it is a function of the training
   iteration, evaluated every update and logged, so a from-scratch run and a
   resumed run are the same command.
2. **Selectable symmetry mode** (HOMIE contribution c), so augmentation and the
   mirror loss can each be ablated without editing source.
3. **Deployment metadata on every exported ONNX**, including a config hash, so
   `scripts/check_deploy_consistency.py` can prove a policy file matches the
   `deploy.yaml` it is about to run under.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import wandb

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.rl.runner import MjlabOnPolicyRunner

from src.tasks.duet.rl.export_metadata import build_deploy_metadata


@dataclass(frozen=True)
class EntropySchedule:
  """Two-stage entropy coefficient: explore, then settle.

  Rationale. A constant bonus keeps pumping noise into a policy that has already
  converged: on the previous run, entropy inflated 3.3 -> 4.5 and action std
  0.33 -> 0.36 between the curriculum freeze and iteration 11.3k, while mean
  reward drifted 76 -> 72 and yaw tracking eroded. But dropping it early starves
  exploration while the curricula are still opening up the task.

  So: hold ``high`` until the curricula finish (``hold_iters``), then decay
  linearly to ``low`` over ``decay_iters`` and stay there. Both endpoints are
  the values the previous task used -- the change is that the transition is
  scheduled instead of hand-edited between runs.
  """

  high: float = 0.01
  low: float = 0.004
  hold_iters: int = 7_000
  decay_iters: int = 2_000

  def __call__(self, iteration: int) -> float:
    if iteration <= self.hold_iters:
      return self.high
    if self.decay_iters <= 0:
      return self.low
    frac = min(1.0, (iteration - self.hold_iters) / self.decay_iters)
    return self.high + frac * (self.low - self.high)


class DuetOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  #: Overridable by a task registration; see rl_cfg.py.
  entropy_schedule: EntropySchedule | None = EntropySchedule()

  def __init__(self, env, train_cfg: dict, log_dir=None, device: str = "cpu"):
    # Deep-copy before mutating: rsl_rl stores the live env in
    # symmetry_cfg["_env"], and the caller's dict is later dumped to agent.yaml
    # -- mutating it in place would make that dump fail on the MjSpec.
    train_cfg = copy.deepcopy(train_cfg)
    algorithm = train_cfg.setdefault("algorithm", {})

    # HOMIE contribution (c). The mode is carried on the algorithm cfg by
    # rl_cfg.py; pop it here because rsl_rl's PPO does not accept the key.
    mode = algorithm.pop("symmetry_mode", "both")
    coeff = algorithm.pop("mirror_loss_coeff", 1.0)
    from src.tasks.common.mdp.symmetry import build_symmetry_cfg

    algorithm["symmetry_cfg"] = build_symmetry_cfg(mode, coeff)
    self._symmetry_mode = mode

    schedule_cfg = algorithm.pop("entropy_schedule", None)
    if schedule_cfg is not None:
      self.entropy_schedule = (
        EntropySchedule(**schedule_cfg)
        if isinstance(schedule_cfg, dict)
        else schedule_cfg
      )
      # Start the algorithm at the schedule's iteration-0 value so the config
      # dump and the first update agree.
      algorithm["entropy_coef"] = self.entropy_schedule(0)

    super().__init__(env, train_cfg, log_dir, device)

    self._install_entropy_schedule()
    self._verify_symmetry()

  # -- Entropy schedule ------------------------------------------------------

  def _install_entropy_schedule(self) -> None:
    """Set ``alg.entropy_coef`` from the schedule before every update.

    rsl_rl's ``learn()`` exposes no per-iteration hook, so the schedule is
    applied by wrapping ``alg.update`` -- the one call made exactly once per
    iteration. The coefficient is added to the returned loss dict, which the
    logger writes to tfevents generically, so the schedule is visible on the
    training curves rather than being an invisible config detail.
    """
    if self.entropy_schedule is None:
      return
    inner_update = self.alg.update
    schedule = self.entropy_schedule

    def update_with_scheduled_entropy(*args, **kwargs):
      coef = schedule(self.current_learning_iteration)
      self.alg.entropy_coef = coef
      loss_dict = inner_update(*args, **kwargs)
      if isinstance(loss_dict, dict):
        loss_dict["entropy_coef"] = coef
      return loss_dict

    self.alg.update = update_with_scheduled_entropy  # type: ignore[method-assign]

  # -- Symmetry --------------------------------------------------------------

  def _verify_symmetry(self) -> None:
    """Assert the mirror map is an involution on real observations.

    A permutation-plus-sign map is only a valid reflection if applying it twice
    is the identity. If a term's rule is wrong the augmented samples are
    silently corrupted and training degrades for no visible reason, so this is
    checked once at startup where it is cheap and loud. Also catches the
    ``height_scan`` case, which used to raise ``KeyError`` on rough terrain.
    """
    import torch

    from src.tasks.common.mdp.symmetry import check_symmetry

    with torch.inference_mode():
      obs = self.env.get_observations().to(self.device)
      actions = torch.zeros(
        obs.batch_size[0], self.env.num_actions, device=self.device
      )
      errs = check_symmetry(self.env, obs, actions)
    worst = max(errs.values())
    if worst > 1e-5:
      raise RuntimeError(
        f"symmetry map is not self-inverse (max |x - m(m(x))| = {worst:.3e}); "
        f"per-group: { {k: f'{v:.2e}' for k, v in errs.items()} }"
      )
    print(f"[INFO] symmetry mode='{self._symmetry_mode}', involution check passed.")

  # -- Export ----------------------------------------------------------------

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    attach_metadata_to_onnx(onnx_path, build_deploy_metadata(self.env.unwrapped, run_name))
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
