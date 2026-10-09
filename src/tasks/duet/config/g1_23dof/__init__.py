from mjlab.tasks.registry import register_mjlab_task

from src.tasks.duet.rl import DuetOnPolicyRunner

from .env_cfgs import (
  unitree_g1_23dof_duet_flat_env_cfg,
  unitree_g1_23dof_duet_rough_env_cfg,
  unitree_g1_23dof_duet_rough_squatreach_env_cfg,
)
from .rl_cfg import (
  ROUGH_MAX_ITERATIONS,
  ROUGH_REWARD_FLOOR,
  unitree_g1_23dof_duet_ppo_runner_cfg,
)

# Both Flat and Rough have the same blind 71-D actor observation, matching the
# ONNX the C++ controller loads; Rough keeps height_scan in the critic only.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-Duet-Flat",
  env_cfg=unitree_g1_23dof_duet_flat_env_cfg(),
  play_env_cfg=unitree_g1_23dof_duet_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_duet_ppo_runner_cfg(),
  runner_cls=DuetOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Duet-Rough",
  env_cfg=unitree_g1_23dof_duet_rough_env_cfg(),
  play_env_cfg=unitree_g1_23dof_duet_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_duet_ppo_runner_cfg(
    max_iterations=ROUGH_MAX_ITERATIONS, reward_floor=ROUGH_REWARD_FLOOR
  ),
  runner_cls=DuetOnPolicyRunner,
)

# Rough + deep-squat manipulation; fine-tuned from a Rough checkpoint with
# --agent.resume (same experiment directory, same reward floor). Measured with
# model_35000 (512 envs, 30 s): deep squat 8.0% -> 17.0% of env time, both arms
# forward while squatting 24% -> 41%, i.e. deep squat with arms forward ~1.9%
# -> ~7.0% of env time.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-Duet-Rough-SquatReach",
  env_cfg=unitree_g1_23dof_duet_rough_squatreach_env_cfg(),
  play_env_cfg=unitree_g1_23dof_duet_rough_squatreach_env_cfg(play=True),
  rl_cfg=unitree_g1_23dof_duet_ppo_runner_cfg(
    max_iterations=ROUGH_MAX_ITERATIONS, reward_floor=ROUGH_REWARD_FLOOR
  ),
  runner_cls=DuetOnPolicyRunner,
)

# Deployment-grade PD gain table (unitree_rl_gym: hips 100, knee 150, ankles 40)
# with a uniform 0.25 action scale. Checkpoints do NOT transfer between gain
# variants -- the exported deploy.yaml must carry the matching table, which
# scripts/check_deploy_consistency.py enforces.
register_mjlab_task(
  task_id="Unitree-G1-23Dof-Duet-Flat-DeployGains",
  env_cfg=unitree_g1_23dof_duet_flat_env_cfg(deploy_gains=True),
  play_env_cfg=unitree_g1_23dof_duet_flat_env_cfg(play=True, deploy_gains=True),
  rl_cfg=unitree_g1_23dof_duet_ppo_runner_cfg(),
  runner_cls=DuetOnPolicyRunner,
)

# Matched-budget, single-variable ablations (paper).
from . import ablations  # noqa: E402,F401

# Independent marching/litter experiment; existing task configurations above
# remain unchanged and are covered by the before/after YAML snapshot audit.
from .poultry_surface import (  # noqa: E402
  poultry_surface_ppo_runner_cfg,
  unitree_g1_23dof_duet_poultry_surface_env_cfg,
)

register_mjlab_task(
  task_id="Unitree-G1-23Dof-Duet-PoultrySurface",
  env_cfg=unitree_g1_23dof_duet_poultry_surface_env_cfg(),
  play_env_cfg=unitree_g1_23dof_duet_poultry_surface_env_cfg(play=True),
  rl_cfg=poultry_surface_ppo_runner_cfg(),
  runner_cls=DuetOnPolicyRunner,
)
