"""Independent poultry-surface task; existing DUET variants keep their configs."""

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from src.tasks.duet.mdp.poultry_surface import (
  PoultrySurfaceActionCfg,
  poultry_clearance,
  poultry_foot_level,
  poultry_metric,
  poultry_plow,
  poultry_swing_height,
)
from .env_cfgs import unitree_g1_23dof_duet_rough_env_cfg
from .rl_cfg import unitree_g1_23dof_duet_ppo_runner_cfg


def unitree_g1_23dof_duet_poultry_surface_env_cfg(play=False, litter_enabled=True):
  cfg = unitree_g1_23dof_duet_rough_env_cfg(play=play)
  # The new task uses identifiable terrain columns even in play. Membership is
  # checked at each foot point's actual xy, including crossings between tiles.
  cfg.scene.terrain.terrain_generator.curriculum = True
  cfg.actions["poultry_surface"] = PoultrySurfaceActionCfg(
    entity_name="robot", litter_enabled=litter_enabled,
  )
  # V1 saved rollouts: sole apex ~1 cm, ankle stays near default, and a level
  # 10 cm swing competes with ~0.23/s posture reward. These tolerances reduce
  # that estimated cost to ~0.04/s while retaining the balance-joint settings.
  cfg.rewards["pose"].params["std_walking"].update({
    r".*hip_pitch.*": 0.8, r".*knee.*": 0.8, r".*ankle_pitch.*": 0.4,
  })
  # Lowest sole above the stance floor; contact-independent travel cost. The
  # old world-z/midfoot foot_clearance is removed here (a separate key keeps the
  # curves from reading as the old term) and untouched for every other task.
  cfg.rewards.pop("foot_clearance")
  cfg.rewards["poultry_clearance"] = RewardTermCfg(func=poultry_clearance, weight=-2.0)
  # Offline V1 rates at d=4 cm: ~0.012-0.015 raw plow on bare flat/soft floor.
  # Weight -6 makes that ~0.075-0.09/s, above the relaxed posture cost and well
  # below individual tracking rewards (~1.4-1.95/s). Layer-relative cost is
  # larger for buried toes. Measured on v1 at 0.35-0.5 m/s: -0.08 to -0.12/s
  # on bare flat/soft floor, -0.18 to -0.25/s on the 4 cm litter row.
  cfg.rewards["poultry_plow"] = RewardTermCfg(func=poultry_plow, weight=-6.0)
  # At 3.33 landings/s, a fully missed target costs about 0.2/s. Unlike just
  # raising an L1 target, this supplies height pressure to an already-low gait.
  cfg.rewards["poultry_swing_height"] = RewardTermCfg(func=poultry_swing_height, weight=-0.06)
  cfg.rewards["poultry_foot_level"] = RewardTermCfg(func=poultry_foot_level, weight=-1.0)
  for key, name in {
    "poultry_swing_apex_m": "apex",  # held last completed apex, episode mean
    "poultry_swing_clearance_m": "clearance",
    "poultry_plow_cost": "plow",
    "poultry_toe_low_travel_mps": "low_travel",
    "poultry_toe_contact_slide_mps": "contact_slide",
    "poultry_liftoff_toe_down_deg": "liftoff_toe_down",
    "poultry_drag_force_n": "drag_force",
    "poultry_layer_height_m": "layer",
    "poultry_falls": "falls",
  }.items():
    cfg.metrics[key] = MetricsTermCfg(func=poultry_metric, params={"name": name})
  return cfg


def poultry_surface_ppo_runner_cfg():
  return unitree_g1_23dof_duet_ppo_runner_cfg(experiment_name="DUET_G1_23dof_poultry")
