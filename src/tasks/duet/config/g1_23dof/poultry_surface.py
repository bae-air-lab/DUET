"""Independent poultry-surface task; existing DUET variants keep their configs."""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

import src.tasks.duet.mdp as mdp
from src.tasks.duet.mdp.poultry_surface import (
  PoultrySurfaceActionCfg,
  backward_speed_deficit,
  poultry_clearance,
  poultry_foot_level,
  poultry_metric,
  poultry_plow,
  poultry_swing_height,
  stop_backward_walkers,
)
from .env_cfgs import (
  HEIGHT_RANGE,
  SQUAT_REACH_POSES,
  _arm_pose,
  unitree_g1_23dof_duet_rough_env_cfg,
)
from .rl_cfg import ROUGH_REWARD_FLOOR, unitree_g1_23dof_duet_ppo_runner_cfg


def unitree_g1_23dof_duet_poultry_surface_env_cfg(play=False, litter_enabled=True):
  cfg = unitree_g1_23dof_duet_rough_env_cfg(play=play)
  # Cap every training stage as well as the unstaged play command range.
  if "command_vel" in cfg.curriculum:
    for stage in cfg.curriculum["command_vel"].params["velocity_stages"]:
      lo, hi = stage["lin_vel_x"]
      stage["lin_vel_x"] = (max(lo, -0.6), hi)
  twist = cfg.commands["twist"]
  lo, hi = twist.ranges.lin_vel_x
  twist.ranges.lin_vel_x = (max(lo, -0.6), hi)
  cfg.rewards["backward_speed_deficit"] = RewardTermCfg(
    func=backward_speed_deficit, weight=-1.5,
  )
  cfg.events["stop_backward_walkers"] = EventTermCfg(
    func=stop_backward_walkers, mode="interval", interval_range_s=(1.0, 3.0),
    params={"prob": 0.3},
  )

  # Restore squat balance in this Rough-derived task, including medium-height
  # squats and a fading centering penalty at the nominal standing height.
  cfg.rewards["squat_com_centering"] = RewardTermCfg(
    func=mdp.squat_com_centering, weight=-2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "height_command_name": "base_height",
      "squat_below": 0.65,
      "full_above": 0.85,
      "foot_half_length": 0.09,
      "command_threshold": 0.1,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )
  cfg.rewards["body_orientation_l2"].params.update(
    relax_below=0.35, full_above=0.60, relax_factor=0.15,
  )
  arm = cfg.actions["upper_body_pose"]
  arm.squat_reach_poses = SQUAT_REACH_POSES + (
    _arm_pose(pitch=0.0, roll=1.2, yaw=0.0, elbow=0.30),
    _arm_pose(pitch=-0.6, roll=0.9, yaw=0.0, elbow=0.50),
  )
  arm.squat_reach_prob = 0.75
  arm.squat_reach_below = 0.65
  cfg.commands["base_height"].deep_squat_fraction = 0.6
  cfg.commands["base_height"].deep_squat_range = (HEIGHT_RANGE[0], 0.65)

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
  return unitree_g1_23dof_duet_ppo_runner_cfg(
    experiment_name="DUET_G1_23dof_poultry",
    reward_floor=ROUGH_REWARD_FLOOR,
    max_iterations=100_001,
  )
