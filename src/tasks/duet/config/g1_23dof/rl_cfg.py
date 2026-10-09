"""PPO configuration for the G1-23DOF DUET task.

Two things differ from a stock mjlab PPO config, both of them defects fixed:

- ``entropy_coef`` is not a hand-edited constant. It is produced by an
  :class:`EntropySchedule` that the runner evaluates every update and logs to
  tfevents, so a fresh run and a resumed run use the same command line. The
  schedule decays through the locomotion -> disturbance transition (see the
  measurement note on ``entropy_schedule``).
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
# Rough: the stretched curriculum (env_cfgs.ROUGH_STRETCH) saturates at
# 12,500, so 50k keeps ~37k iterations on the full task distribution.
ROUGH_MAX_ITERATIONS = 50_001
# Rough: per-step reward floor (dt-scaled units). A normal step is ~+0.08 and
# a fall is -4 (is_terminated -200 x 0.02 s), so -5 never touches a normal or
# a falling step; it only caps physics blow-ups, which reached ~-150 per step
# in the 50k run (see SPAWN_PATCHES in duet_env_cfg.py).
ROUGH_REWARD_FLOOR = -5.0


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
      # Schedule inverted on 2026-09-15 (arm-robustness pass): high exploration
      # DURING the stationary locomotion-foundation stage, low once disturbances
      # ramp -- not "hold high until the curricula finish" as before.
      # Measured in the first run with the old schedule (0.01 held to 7000):
      # std bottomed at 0.216 at iteration ~1500, then rose monotonically to
      # 0.72 by 5170 as pushes, arm motion and squat depth ramped; value loss
      # 0.02 -> 0.54; return 101 -> 2. The action-rate penalty tracked the
      # exploration-noise floor 13*2*std^2*0.1 to within ~13%, i.e. the policy
      # was being charged almost entirely for its own noise. Mechanism: the
      # entropy push on a scalar std is constant (coef*13/std) while the
      # policy-gradient counter-force scales with how much of the advantage is
      # explained by the noise; random pushes and unobserved arm goals add
      # advantage variance the critic cannot remove, so the counter-force
      # weakens exactly as the disturbances grow. Hence the coefficient must
      # come DOWN as the task becomes stochastic. 0.01 was demonstrably fine
      # while the task was stationary (std 0.22 at return 101).
      # Previous floor note (2026-08-07): 0.004 let std fall to 0.20 on the
      # old task and tracking slowly over-fit; 0.006 was chosen then. Under
      # the new disturbance levels 0.004 is the floor that data supports.
      "low": 0.004,
      "hold_iters": 1_500,
      "decay_iters": 1_000,
    }
  )
  """Two-stage entropy schedule; see ``EntropySchedule``. Set to ``None`` for a
  flat coefficient (then ``entropy_coef`` is used as-is)."""

  reward_floor: float | None = None
  """Clamp every per-step reward to at least this value before PPO sees it,
  so a single physics blow-up cannot put an outlier into the value targets.
  ``None`` (default) leaves rewards untouched."""


def unitree_g1_23dof_duet_ppo_runner_cfg(
  max_iterations: int = MAX_ITERATIONS,
  symmetry_mode: str = "both",
  entropy_schedule: dict | None = None,
  experiment_name: str = "DUET_G1_23dof",
  seed: int = 42,
  reward_floor: float | None = None,
) -> RslRlOnPolicyRunnerCfg:
  """Create the RL runner configuration.

  Args:
    max_iterations: Training budget.
    symmetry_mode: none | augment | loss | both.
    entropy_schedule: Override dict for the entropy schedule; ``None`` keeps
      the default two-stage schedule.
    experiment_name: Log directory name.
    seed: Overridden on the command line with ``--agent.seed``.
    reward_floor: Per-step reward floor; ``None`` disables it.
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
    reward_floor=reward_floor,
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
