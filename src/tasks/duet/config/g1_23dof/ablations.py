"""Matched-budget ablations of the DUET training recipe.

Each variant differs from the reference in EXACTLY ONE design choice, so the
resulting policies are comparable under a single evaluation protocol
(``duet_eval.py``, 1024 envs, 20 s episodes, arm curriculum pinned at full
strength). All variants train on flat terrain with the same seed, the same PPO
settings and the same budget.

The curriculum is compressed 2x relative to the main run (ramps saturate at
~3,500 instead of ~7,000 iterations) so that within the shortened ablation
budget the reference variant still reaches the full task distribution and then
trains on it for a while.

Variants
--------
Abl-Reference       full recipe (compressed curriculum), the control condition
Abl-UniformArm      HOMIE (a): arm goals 100% uniform over the safe workspace.
                    Since the arm-robustness pass this IS the reference recipe
                    (the default arm_mode is "uniform"); the variant is kept so
                    old result rows keep their name, and now measures nothing
                    the reference does not
Abl-NoArmCurriculum HOMIE (a), removed: arms pinned at the default pose, so the
                    lower body never sees an upper-body disturbance
Abl-NoHeightCmd     HOMIE (b), removed: height pinned at 0.73; the observation
                    slot remains (constant) so the 71-D interface is preserved
Abl-NoSymmetry      HOMIE (c), removed: no augmentation, no mirror loss.
                    Loss/symmetry is still logged, so this variant reports the
                    quantity it ablates
Abl-SymAugmentOnly  HOMIE (c), partial: augmentation but no mirror loss (this
                    is what the previous loco_manip task did)
Abl-NoPayload       no hand or torso payload randomisation
Abl-NoCurriculum    no staged command curriculum: full velocity range, full
                    height range and the final walk/stand mixture from step 0
Abl-FlatEntropy     control for the two-stage entropy schedule: constant 0.01
Abl-LowVelNoise     joint_vel observation noise reduced from +-1.5 to +-0.3
                    (the physical encoder-difference value), testing whether
                    the large noise is load-bearing for sim2real robustness
Abl-NoIdlePrecision drops the idle-precision group (position anchor, idle feet
                    stillness, and the always-on foot-slip gate), i.e. the
                    pre-hardware-feedback reward set. Measures what the standing
                    precision costs in locomotion performance
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import register_mjlab_task

from src.tasks.duet.rl import DuetOnPolicyRunner

from .env_cfgs import apply_duet_curriculum, unitree_g1_23dof_duet_flat_env_cfg
from .rl_cfg import unitree_g1_23dof_duet_ppo_runner_cfg

_ABL_END_ITERS = 3_500  # compressed curriculum saturation
_ABL_ITERS = 6_001
_ABL_EXPERIMENT = "DUET_G1_23dof_Ablation"

# Ablation entropy schedule, compressed to match the compressed curriculum.
_ABL_ENTROPY = {
  "high": 0.01,
  "low": 0.004,
  "hold_iters": _ABL_END_ITERS,
  "decay_iters": 1_000,
}


def _compress_curriculum(cfg: ManagerBasedRlEnvCfg) -> None:
  """Halve every curriculum horizon (7k -> 3.5k iterations).

  Re-installs every staged schedule (command envelope, command mix, squat
  floor, push magnitude, arm ratio / clean fraction) with all knots scaled by
  3.5k / 7k, so the stages keep their relative timing.
  """
  if "command_vel" not in cfg.curriculum:  # play mode clears the curriculum
    return
  apply_duet_curriculum(cfg, end_iters=_ABL_END_ITERS)


# variant -> (env kwargs, ppo kwargs)
_VARIANTS: dict[str, tuple[dict, dict]] = {
  "Abl-Reference": ({}, {}),
  "Abl-UniformArm": ({"arm_mode": "uniform"}, {}),
  "Abl-NoArmCurriculum": ({"arm_mode": "off"}, {}),
  "Abl-NoHeightCmd": ({"height_mode": "fixed"}, {}),
  "Abl-NoSymmetry": ({}, {"symmetry_mode": "none"}),
  "Abl-SymAugmentOnly": ({}, {"symmetry_mode": "augment"}),
  "Abl-NoPayload": ({"payload": False}, {}),
  "Abl-NoCurriculum": ({"command_curriculum": False}, {}),
  "Abl-FlatEntropy": ({}, {"entropy_schedule": None}),
  "Abl-LowVelNoise": ({"joint_vel_noise": 0.3}, {}),
  "Abl-NoIdlePrecision": ({"idle_precision": False}, {}),
}


def ablation_env_cfg(variant: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  if variant not in _VARIANTS:
    raise ValueError(f"Unknown ablation variant: {variant}")
  env_kwargs, _ = _VARIANTS[variant]
  cfg = unitree_g1_23dof_duet_flat_env_cfg(play=play, **env_kwargs)
  # NoCurriculum has already dropped the staged terms; compressing is a no-op
  # for it and would reintroduce them, so skip.
  if variant != "Abl-NoCurriculum":
    _compress_curriculum(cfg)
  return cfg


def ablation_ppo_cfg(variant: str) -> RslRlOnPolicyRunnerCfg:
  _, ppo_kwargs = _VARIANTS[variant]
  cfg = unitree_g1_23dof_duet_ppo_runner_cfg(
    max_iterations=_ABL_ITERS,
    experiment_name=_ABL_EXPERIMENT,
    symmetry_mode=ppo_kwargs.get("symmetry_mode", "both"),
    entropy_schedule=_ABL_ENTROPY,
  )
  if "entropy_schedule" in ppo_kwargs and ppo_kwargs["entropy_schedule"] is None:
    # Abl-FlatEntropy: constant 0.01 for the whole run.
    cfg = replace(cfg, algorithm=replace(cfg.algorithm, entropy_schedule=None))
  return cfg


for _suffix in _VARIANTS:
  register_mjlab_task(
    task_id=f"Unitree-G1-23Dof-Duet-{_suffix}",
    env_cfg=ablation_env_cfg(_suffix),
    play_env_cfg=ablation_env_cfg(_suffix, play=True),
    rl_cfg=ablation_ppo_cfg(_suffix),
    runner_cls=DuetOnPolicyRunner,
  )
