from mjlab.tasks.registry import register_mjlab_task

from src.tasks.duet.rl import DuetOnPolicyRunner

from .env_cfgs import (
  unitree_g1_23dof_duet_flat_env_cfg,
  unitree_g1_23dof_duet_rough_env_cfg,
)
from .rl_cfg import unitree_g1_23dof_duet_ppo_runner_cfg

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
  rl_cfg=unitree_g1_23dof_duet_ppo_runner_cfg(),
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
