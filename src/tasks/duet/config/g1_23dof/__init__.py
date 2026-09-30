from mjlab.tasks.registry import register_mjlab_task

from src.tasks.duet.rl import DuetOnPolicyRunner

from .env_cfgs import (
  unitree_g1_23dof_duet_flat_env_cfg,
  unitree_g1_23dof_duet_rough_blind_tall_env_cfg,
  unitree_g1_23dof_duet_rough_env_cfg,
)
from .rl_cfg import unitree_g1_23dof_duet_ppo_runner_cfg

# Flat is the variant that gets deployed: its actor observation is exactly 71-D
# (no height_scan), matching the ONNX the C++ controller loads.
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

# Rough-blind-tall pass (documents/duet/rough_blind_task.md): blind actor on
# litter-like terrain with soft/slippery ground and sim-to-real randomisation,
# nominal height 0.79 m. -H5 (primary) gives the actor 5 frames of the 71-D
# observation (355-D input; deploy.yaml needs history_length: 5 on every term).
# The plain id is identical except for 1 frame, a drop-in for the -Flat
# interface. The Eval ids are fixed evaluation conditions of each (play
# settings, same critic layout, so any checkpoint of that task loads).
for _suffix, _history in (("-H5", 5), ("", 1)):
  _task = f"Unitree-G1-23Dof-Duet-RoughBlind-Tall{_suffix}"
  register_mjlab_task(
    task_id=_task,
    env_cfg=unitree_g1_23dof_duet_rough_blind_tall_env_cfg(actor_history=_history),
    play_env_cfg=unitree_g1_23dof_duet_rough_blind_tall_env_cfg(
      play=True, actor_history=_history
    ),
    rl_cfg=unitree_g1_23dof_duet_ppo_runner_cfg(),
    runner_cls=DuetOnPolicyRunner,
  )
  for _eval, _name in (("flat", "EvalFlat"), ("litter", "EvalLitter")):
    _eval_cfg = unitree_g1_23dof_duet_rough_blind_tall_env_cfg(
      actor_history=_history, evaluation=_eval
    )
    register_mjlab_task(
      task_id=f"{_task}-{_name}",
      env_cfg=_eval_cfg,
      play_env_cfg=_eval_cfg,
      rl_cfg=unitree_g1_23dof_duet_ppo_runner_cfg(),
      runner_cls=DuetOnPolicyRunner,
    )

# Matched-budget, single-variable ablations (paper).
from . import ablations  # noqa: E402,F401
