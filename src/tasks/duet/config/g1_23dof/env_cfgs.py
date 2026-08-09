"""DUET environment configuration for the Unitree G1-23DOF.

Sets everything robot-specific on top of ``duet_env_cfg.make_duet_env_cfg()``:
the lower-body/arm joint carve-out, PD gains and action scale (which MUST match
``deploy.yaml``), payload and CoM randomisation, curriculum timing, and the
per-joint posture tolerances.

Three HOMIE contributions, each independently ablatable via the arguments of
:func:`unitree_g1_23dof_duet_rough_env_cfg`:

  a. ``arm_mode``    -- upper-body pose curriculum (amplitude ramp + anchors)
  b. ``height_mode`` -- pelvis height as a first-class command
  c. symmetry        -- set on the runner, see ``rl_cfg.py``

Curriculum state note: every ramp here is keyed on ``env.common_step_counter``,
which ``MjlabOnPolicyRunner`` writes into and restores from the checkpoint.
Resuming therefore continues the curriculum where it stopped. There is no
``RESUME_AT_FULL_DISTRIBUTION`` switch and no source edit is required to resume.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import (
  G1_23DOF_ACTION_SCALE,
  G1_23DOF_DEPLOY_ACTION_SCALE,
  get_g1_23dof_deploy_robot_cfg,
  get_g1_23dof_robot_cfg,
)
import src.tasks.duet.mdp as mdp
from src.tasks.duet.duet_env_cfg import ITER, make_duet_env_cfg

##
# Joint partition for the loco-manipulation carve-out.
# The RL policy controls the lower body (12 legs + waist_yaw = 13 joints); the
# arms (5 joints/side) are driven externally -- by the disturbance generator in
# training, by the VLA on rt/arm_targets at deployment.
##

LOWER_BODY_ACTUATORS = (
  r".*_hip_pitch_joint",
  r".*_hip_roll_joint",
  r".*_hip_yaw_joint",
  r".*_knee_joint",
  r".*_ankle_pitch_joint",
  r".*_ankle_roll_joint",
  r"waist_yaw_joint",
)
ARM_JOINTS = (
  r".*_shoulder_pitch_joint",
  r".*_shoulder_roll_joint",
  r".*_shoulder_yaw_joint",
  r".*_elbow_joint",
  r".*_wrist_roll_joint",
)
_ARM_TOKENS = ("shoulder", "elbow", "wrist")

##
# Curriculum timing, sized for the 25k-iteration budget: every ramp saturates
# at 7,000 iterations, leaving ~18k iterations of training on the full task
# distribution.
##

CURRICULUM_END_ITERS = 7_000
_CURRICULUM_END_STEPS = CURRICULUM_END_ITERS * ITER

##
# Height command (HOMIE contribution b).
##

# Floor raised 0.12 -> 0.18 on 2026-08-06. The old 0.12 was KINEMATICALLY
# IMPOSSIBLE: measured deepest reachable pelvis-above-flat-foot on this robot is
# 0.144 m, and only with hip_pitch pinned at its -2.53 rad limit and ankle_pitch
# at +0.524. Commanding below that produced a permanently unsatisfiable ~4% slice
# of the height distribution, generating a standing gradient that pushed into the
# joint-limit barrier (`joint_pos_limits`, -10.0) for a target the robot cannot
# reach -- which is why the policy learned to STOP chasing deep commands and
# saturated around 0.22-0.25 m. 0.18 sits above the 0.144 limit with margin, so
# the whole commanded range is achievable without pinning joints at their stops.
# deploy.yaml base_height.range MUST match (updated in the same commit).
HEIGHT_RANGE = (0.18, 0.73)
WALK_MIN_HEIGHT_FINAL = 0.60  # walking is restricted to >= 0.60 m
_SQUAT_FLOOR_START = 0.45  # squat depth ramps 0.45 -> 0.18

##
# Payload. See documents/duet/reward_design.md section 2.
##

# Per hand, added to the 0.357 kg sim rubber hand.
#   floor 0.25 = Dex3 (~0.53 kg) + wrist camera - sim hand, the PERMANENT
#     hardware surplus. The policy must never train lighter than the real robot.
#   ceiling 1.75 = 0.25 surplus + 1.5 kg cargo. Still ~6x HOMIE's
#     hand_payload_mass_range [-0.1, 0.3] -- cargo carrying is where we exceed
#     HOMIE deliberately.
HAND_PAYLOAD_KG = (0.25, 1.75)

# Torso. torso_link measures 7.818 kg in the compiled model, so the previous
# (-1.0, +5.0) put +64% of the link's own mass on it -- making an unmodelled
# torso brick the dominant randomisation and diluting the hand-payload CoM
# shift that is the actual task disturbance. +2.0 covers a realistic accessory
# load (compute box + spare battery + camera mounts, +26% of the link); -0.5
# covers URDF mass overestimate. Worst-case total added mass is
# 2*1.75 + 2.0 = 5.5 kg = 17% of the 32.1 kg robot.
TORSO_PAYLOAD_KG = (-0.5, 2.0)


def _arm_pose(
  pitch: float, roll: float, yaw: float, elbow: float, wrist: float = 0.0
) -> dict[str, float]:
  """Symmetric arm pose; the right side mirrors roll/yaw/wrist signs."""
  return {
    "left_shoulder_pitch_joint": pitch,
    "right_shoulder_pitch_joint": pitch,
    "left_shoulder_roll_joint": roll,
    "right_shoulder_roll_joint": -roll,
    "left_shoulder_yaw_joint": yaw,
    "right_shoulder_yaw_joint": -yaw,
    "left_elbow_joint": elbow,
    "right_elbow_joint": elbow,
    "left_wrist_roll_joint": wrist,
    "right_wrist_roll_joint": -wrist,
  }


# Deployment-pose anchors for the arm curriculum. Signs: negative shoulder pitch
# raises the arm forward/up; default hang is pitch 0.35, elbow 0.87, roll +-0.18.
# These are the poses the VLA will actually hold at deployment.
TASK_ARM_POSES = (
  _arm_pose(pitch=-0.10, roll=0.10, yaw=0.0, elbow=1.20),  # box carry at waist
  _arm_pose(pitch=-0.50, roll=0.12, yaw=0.0, elbow=1.00),  # box carry at chest
  _arm_pose(pitch=-0.90, roll=0.10, yaw=0.0, elbow=0.30),  # forward reach (table)
  _arm_pose(pitch=0.60, roll=0.15, yaw=0.0, elbow=0.15),  # low pick (with squat)
  _arm_pose(pitch=-1.60, roll=0.10, yaw=0.0, elbow=0.40),  # high reach (shelf)
  _arm_pose(pitch=0.35, roll=0.18, yaw=0.0, elbow=0.87),  # rest / carry at side
)

# Noise half-widths (rad) around each anchor: enough spread to cover different
# box widths, grasp heights and asymmetric reaches without becoming flailing.
TASK_ARM_POSE_NOISE = {
  r".*_shoulder_pitch_joint": 0.25,
  r".*_shoulder_roll_joint": 0.15,
  r".*_shoulder_yaw_joint": 0.30,
  r".*_elbow_joint": 0.25,
  r".*_wrist_roll_joint": 0.50,
}


def unitree_g1_23dof_duet_rough_env_cfg(
  play: bool = False,
  deploy_gains: bool = False,
  arm_mode: str = "anchored",
  height_mode: str = "command",
  payload: bool = True,
  command_curriculum: bool = True,
  joint_vel_noise: float | None = None,
  idle_precision: bool = True,
) -> ManagerBasedRlEnvCfg:
  """Create the G1-23DOF DUET rough-terrain configuration.

  Args:
    play: Play/visualisation mode (no corruption, no pushes, no curriculum).
    deploy_gains: Train against the unitree_rl_gym deployment gain table
      (hips 100, knee 150, ankles 40) with a uniform 0.25 action scale instead
      of the soft first-principles gains. Checkpoints do NOT transfer between
      the two; the exported deploy.yaml must carry the matching table.
    arm_mode: HOMIE contribution (a) ablation.
      ``anchored`` -- 70% deployment-pose anchors + 30% uniform (default).
      ``uniform``  -- 100% uniform workspace sampling.
      ``off``      -- arms pinned at the default pose; no upper-body disturbance.
    height_mode: HOMIE contribution (b) ablation.
      ``command`` -- pelvis height sampled in [0.12, 0.73] (default).
      ``fixed``   -- height pinned at 0.73; the obs slot stays (constant), so
                     the 71-D interface and deployability are unchanged.
    payload: When False, drop hand and torso payload randomisation entirely.
    command_curriculum: When False, sample the final (hardest) command
      distribution from step 0 instead of staging into it.
    joint_vel_noise: Override the joint-velocity observation noise half-width.
  """
  cfg = make_duet_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {
    "robot": get_g1_23dof_deploy_robot_cfg()
    if deploy_gains
    else get_g1_23dof_robot_cfg()
  }

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "pelvis"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg, self_collision_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  ##
  # Actions: the lower-body carve-out. action_dim = 13.
  ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = LOWER_BODY_ACTUATORS
  if deploy_gains:
    joint_pos_action.scale = G1_23DOF_DEPLOY_ACTION_SCALE
  else:
    # Soft first-principles per-joint scale (hip 0.55, knee 0.35, ankle 0.44),
    # matching deploy.yaml. Arms excluded -- they are not policy actions.
    joint_pos_action.scale = {
      k: v
      for k, v in G1_23DOF_ACTION_SCALE.items()
      if not any(tok in k for tok in _ARM_TOKENS)
    }

  # HOMIE contribution (a): the upper-body pose curriculum. Consumes ZERO
  # policy action dimensions -- it only writes joint targets for the arms.
  if arm_mode not in ("anchored", "uniform", "off"):
    raise ValueError(f"arm_mode must be anchored|uniform|off, got {arm_mode!r}")
  cfg.actions["upper_body_pose"] = mdp.UpperBodyPoseActionCfg(
    entity_name="robot",
    joint_names=ARM_JOINTS,
    enabled=arm_mode != "off",
    sample_range_scale=0.5,
    task_poses=TASK_ARM_POSES,
    task_pose_prob=0.7 if arm_mode == "anchored" else 0.0,
    task_pose_noise=TASK_ARM_POSE_NOISE,
    ratio_curriculum_steps=_CURRICULUM_END_STEPS,
  )

  cfg.viewer.body_name = "torso_link"

  ##
  # Commands.
  ##

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  # HOMIE contribution (b). Range matches deploy.yaml exactly; the squat floor
  # ramps 0.45 -> 0.12 so the policy masters shallow squats before deep ones.
  if height_mode not in ("command", "fixed"):
    raise ValueError(f"height_mode must be command|fixed, got {height_mode!r}")
  base_height_cmd = cfg.commands["base_height"]
  assert isinstance(base_height_cmd, mdp.BaseHeightCommandCfg)
  base_height_cmd.enabled = height_mode == "command"
  base_height_cmd.height_range = HEIGHT_RANGE
  base_height_cmd.walk_min_height = 0.71  # lowered to 0.60 by command_mix
  base_height_cmd.floor_curriculum_start = _SQUAT_FLOOR_START
  base_height_cmd.floor_curriculum_steps = _CURRICULUM_END_STEPS

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  ##
  # Domain randomisation.
  ##

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  # Mainly fore-aft (x): that is the CoM shift a carried box induces.
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.10, 0.10),
    1: (-0.06, 0.06),
    2: (-0.05, 0.05),
  }

  if payload:
    cfg.events["hand_payload"] = EventTermCfg(
      mode="reset",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=(r".*_wrist_roll_rubber_hand",)
        ),
        "operation": "add",
        "ranges": HAND_PAYLOAD_KG,
      },
    )
    # Where the hand mass SITS, not just how much of it there is. The sim
    # rubber hand is a compact blob at the wrist; a real Dex3 plus wrist camera
    # carries its mass further out, so an extended arm applies a much larger
    # moment about the shoulder and about the support polygon. Randomising the
    # hand CoM makes the policy robust to that lever arm rather than to mass
    # alone. This is the leading suspect for why arms-forward drift shows up on
    # hardware but measures ~8x CALMER in sim -- mass is already covered by
    # hand_payload, the moment arm was not.
    cfg.events["hand_com"] = EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=(r".*_wrist_roll_rubber_hand",)
        ),
        "operation": "add",
        # Symmetric on every axis: the wrist frame convention is not verified
        # here, so an asymmetric range would encode an assumption rather than a
        # measurement. +-5 cm spans where a Dex3's mass can plausibly sit.
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    )
    cfg.events["torso_payload"] = EventTermCfg(
      mode="reset",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": TORSO_PAYLOAD_KG,
      },
    )

  if joint_vel_noise is not None:
    for group in ("actor", "critic"):
      term = cfg.observations[group].terms["joint_vel"]
      if term.noise is not None:
        term.noise.n_min = -joint_vel_noise
        term.noise.n_max = joint_vel_noise

  ##
  # Rewards: fill in the per-robot bodies/sites/joints.
  ##

  # Posture regularisation covers ONLY the RL-controlled joints. The arms are
  # driven externally, so penalising their deviation would penalise the policy
  # for its own disturbance curriculum.
  cfg.rewards["pose"].params["asset_cfg"].joint_names = LOWER_BODY_ACTUATORS
  # Loose on the squat joints so a commanded squat is not fought; tight on the
  # lateral/yaw joints, which are what balance depends on. Squat depth is driven
  # by track_base_height, never by posture regularisation.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_pitch.*": 0.8,
    r".*knee.*": 0.8,
    r".*ankle_pitch.*": 0.4,
    r".*hip_roll.*": 0.05,
    r".*hip_yaw.*": 0.05,
    r".*ankle_roll.*": 0.05,
    r".*waist_yaw.*": 0.05,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.15,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.15,
  }
  # Squat regime: the folding joints are effectively unregularised at depth
  # (a deep squat needs ~1.5-2.5 rad of hip_pitch/knee deviation, which the
  # standing std of 0.8 punishes as (1.9/0.8)^2 = 5.6 per joint), while the
  # lateral/yaw joints stay tight because they are what balance depends on.
  cfg.rewards["pose"].params["std_squatting"] = {
    r".*hip_pitch.*": 3.0,
    r".*knee.*": 3.0,
    r".*ankle_pitch.*": 1.5,
    r".*hip_roll.*": 0.05,
    r".*hip_yaw.*": 0.05,
    r".*ankle_roll.*": 0.05,
    r".*waist_yaw.*": 0.05,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.25,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
  for name in (
    "track_base_height",
    "feet_distance",
    "foot_clearance",
    "foot_slip",
    "idle_feet_still",
  ):
    cfg.rewards[name].params["asset_cfg"].site_names = site_names

  # Idle-precision group, added after hardware testing showed the standing base
  # drifting -- markedly worse with the arms extended forward, which shifts the
  # CoM. Ablatable as one unit so its cost to locomotion can be measured.
  if not idle_precision:
    cfg.rewards.pop("idle_position_anchor", None)
    cfg.rewards.pop("idle_feet_still", None)
    cfg.rewards["foot_slip"].params["always_active"] = False
  cfg.rewards["joint_vel_l2"].params["asset_cfg"].joint_names = LOWER_BODY_ACTUATORS

  ##
  # Curriculum. All stages are in POLICY STEPS (iterations * ITER) and are read
  # against env.common_step_counter, which is checkpointed -> resume-safe.
  ##

  if command_curriculum:
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {
            "step": 0,
            "lin_vel_x": (-0.5, 1.0),
            "lin_vel_y": (-0.4, 0.4),
            "ang_vel_z": (-0.8, 0.8),
          },
          {
            # HOMIE's own command range, except yaw widened to +-1.0 to cover
            # deploy.yaml's joystick mapping. The trained range is a strict
            # superset of the commanded range, so deployment never extrapolates.
            "step": 5_000 * ITER,
            "lin_vel_x": (-0.8, 1.2),
            "lin_vel_y": (-0.5, 0.5),
            "ang_vel_z": (-1.0, 1.0),
          },
        ],
      },
    )
    # Curriculum by command DISTRIBUTION: walk-dominant near nominal height
    # first, then standing squats, finally squat-while-walking.
    cfg.curriculum["command_mix"] = CurriculumTermCfg(
      func=mdp.command_mix,
      params={
        "twist_command_name": "twist",
        "height_command_name": "base_height",
        "stages": [
          {"step": 0, "rel_standing_envs": 0.20, "walk_min_height": 0.71},
          {"step": 4_000 * ITER, "rel_standing_envs": 0.30, "walk_min_height": 0.66},
          {
            "step": _CURRICULUM_END_STEPS,
            "rel_standing_envs": 0.30,
            "walk_min_height": WALK_MIN_HEIGHT_FINAL,
          },
        ],
      },
    )
  else:
    # Ablation: no staging. Sample the final distribution from step 0.
    twist_cmd.rel_standing_envs = 0.30
    twist_cmd.ranges.lin_vel_x = (-0.8, 1.2)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
    base_height_cmd.walk_min_height = WALK_MIN_HEIGHT_FINAL
    base_height_cmd.floor_curriculum_start = HEIGHT_RANGE[0]

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={}
    )
    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
      cfg.scene.terrain.terrain_generator.curriculum = False
      cfg.scene.terrain.terrain_generator.num_cols = 5
      cfg.scene.terrain.terrain_generator.num_rows = 5
      cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_g1_23dof_duet_flat_env_cfg(
  play: bool = False, **kwargs
) -> ManagerBasedRlEnvCfg:
  """Flat-terrain DUET config. This is the variant that gets deployed.

  Deleting ``height_scan`` from both observation groups is what makes the actor
  observation exactly 71-D, matching the deployed ONNX input.
  """
  cfg = unitree_g1_23dof_duet_rough_env_cfg(play=play, **kwargs)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
