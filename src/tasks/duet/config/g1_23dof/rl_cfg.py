"""PPO configuration for the G1-23DOF DUET task.

Two things differ from a stock mjlab PPO config, both of them defects fixed:

- ``entropy_coef`` is not a hand-edited constant. It is produced by an
  :class:`EntropySchedule` that the runner evaluates every update and logs to
  tfevents, so a fresh run and a resumed run use the same command line.
- ``symmetry_mode`` selects HOMIE contribution (c) without editing source.

Everything a variant needs to change is an argument of
:func:`unitree_g1_23dof_duet_ppo_runner_cfg`.
"""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.tasks.duet.config.g1_23dof.env_cfgs import CURRICULUM_END_ITERS

# Budget. Curricula saturate at 7,000 iterations, leaving ~18k iterations of
# training on the full task distribution. Checkpoint every 500 so the 10k-25k
# band can be scanned by duet_eval and the deployable checkpoint SELECTED rather
# than assumed to be the last one.
MAX_ITERATIONS = 25_001
SAVE_INTERVAL = 500


@dataclass
class DuetPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO config carrying the DUET-specific knobs.

  These three fields are popped by ``DuetOnPolicyRunner.__init__`` before the
  dict reaches rsl_rl's ``PPO``, which does not accept them.
  """

  symmetry_mode: str = "both"
  """HOMIE contribution (c): none | augment | loss | both.

  ``none`` still logs ``Loss/symmetry`` (rsl_rl computes it for logging even
  when it is excluded from the objective), so the ablation reports the metric
  it ablates."""

  mirror_loss_coeff: float = 1.0
  """Weight of the symmetry loss when ``symmetry_mode`` includes it.
  HOMIE uses ``symmetry_scale = 1.0``."""

  entropy_schedule: dict | None = field(
    default_factory=lambda: {
      "high": 0.01,
      # Floor raised 0.004 -> 0.006 on 2026-08-07. Across the 11 checkpoints
      # measured after the curricula froze, tracking error CORRELATED POSITIVELY
      # with iteration (yaw r = +0.52, vxy r = +0.43) and standing drift with it
      # (r = +0.25) -- i.e. the policy kept sharpening against a fixed objective
      # and slowly got worse. 0.004 let action std fall to 0.20; 0.006 keeps a
      # little exploration alive so refinement does not become over-fitting.
      "low": 0.006,
      "hold_iters": CURRICULUM_END_ITERS,
      "decay_iters": 2_000,
    }
  )
  """Two-stage entropy schedule; see ``EntropySchedule``. Set to ``None`` for a
  flat coefficient (then ``entropy_coef`` is used as-is)."""


def unitree_g1_23dof_duet_ppo_runner_cfg(
  max_iterations: int = MAX_ITERATIONS,
  symmetry_mode: str = "both",
  entropy_schedule: dict | None = None,
  experiment_name: str = "DUET_G1_23dof",
  seed: int = 42,
) -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration.

  Args:
    max_iterations: Training budget.
    symmetry_mode: none | augment | loss | both.
    entropy_schedule: Override dict for the entropy schedule; ``None`` keeps
      the default two-stage schedule.
    experiment_name: Log directory name.
    seed: Overridden on the command line with ``--agent.seed``.
  """
  algorithm = DuetPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    # Initial value only; the schedule overwrites it every update.
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    symmetry_mode=symmetry_mode,
  )
  if entropy_schedule is not None:
    algorithm.entropy_schedule = entropy_schedule

  return RslRlOnPolicyRunnerCfg(
    seed=seed,
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=algorithm,
    experiment_name=experiment_name,
    save_interval=SAVE_INTERVAL,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )
