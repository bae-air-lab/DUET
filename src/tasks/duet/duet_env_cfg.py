"""DUET base environment: robot-agnostic scaffold for the decoupled
loco-manipulation task.

This module owns the **reward set** and the observation layout. Robot-specific
configs (gains, joint partition, payload, curriculum timing) live under
``config/``.

Reward set: 22 terms, every one of which has a stated purpose. The full
justification -- including why four terms present in the previous task were
dropped, and the arithmetic showing the termination penalty does not dominate --
is in ``documents/duet/reward_design.md``. The short version is repeated in the
comments below so the file is readable on its own.

Observation contract (the deployed ONNX interface, do not change without also
changing ``deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml``):

    actor obs = 3 base_ang_vel + 3 projected_gravity + 3 command + 2 phase
              + 23 joint_pos + 23 joint_vel + 13 last_action + 1 height_command
              = 71                                       -> 13 actions

``height_command`` is emitted LAST. The actor is blind on every terrain: the
robot has no height map, so ``height_scan`` is a critic-only (privileged) term
and the rough and flat actors are the same 71-D layout. The flat task also
drops it from the critic, since there is no terrain to scan.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
# Import the twist command from mjlab, NOT from src.tasks.common.mdp, which
# also defines a same-named variant. duet_eval.py isinstance-checks the mjlab
# class, and loco_manip uses it too -- both packages must agree or a DUET policy
# becomes un-evaluable under the manuscript's protocol.
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.duet.mdp as mdp
from src.tasks.common.terrains import SoftFlatTerrainCfg

# One training iteration = num_steps_per_env policy steps. Curriculum "step"
# fields count POLICY steps, so schedules are written as `iterations * _ITER`.
ITER = 24

# joint_vel observation noise. Physical encoder-difference noise on the G1 is
# ~0.1-0.3 rad/s, so +-1.5 is deliberately 5-15x that. It is not modelling the
# sensor: it models the filtered, 1-2 control-period delayed velocity ESTIMATE
# the hardware actually reports, and it stops the policy building a high-gain
# derivative loop it cannot support on the robot. HOMIE, the config we
# benchmark against, uses 2.0 -- larger than this. Kept, and measured against
# the physical value by the `Abl-LowVelNoise` variant rather than assumed.
JOINT_VEL_NOISE = 1.5

# Rough terrain: mjlab's ROUGH_TERRAINS_CFG with both stair types (0.2 + 0.2)
# replaced by one soft-floor type at their combined 0.4, so it takes the same
# 8 of the 20 columns the stairs did. Everything else is unchanged. Stairs are
# not part of the deployment site (a poultry house), whereas soft, slippery
# litter is. See ``src/tasks/common/terrains.py`` for the contact model.
_ROUGH = ROUGH_TERRAINS_CFG.sub_terrains
DUET_TERRAINS_CFG = replace(
  ROUGH_TERRAINS_CFG,
  sub_terrains={
    "flat": _ROUGH["flat"],
    "soft_floor": SoftFlatTerrainCfg(proportion=0.4),
    **{
      name: sub
      for name, sub in _ROUGH.items()
      if name not in ("flat", "pyramid_stairs", "pyramid_stairs_inv")
    },
  },
)


def make_duet_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base DUET task configuration (robot-agnostic)."""

  ##
  # Sensors
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),  # Set per-robot.
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  ##
  # Observations
  ##
  # Order matters: it IS the deployed ONNX input layout. height_command is last.

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-JOINT_VEL_NOISE, n_max=JOINT_VEL_NOISE),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    # HOMIE contribution (b): pelvis height is a first-class command, not a
    # constant. Emitted last so the flat task's obs vector is exactly 71-D.
    "height_command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "base_height"},
    ),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    # Privileged: the critic sees the terrain under the robot, which lowers
    # value-target variance on rough tiles. NOT in the actor -- the robot has
    # no height map, and the actor must stay the 71-D deployed interface.
    "height_scan": ObservationTermCfg(
      func=envs_mdp.height_scan,
      params={"sensor_name": "terrain_scan"},
      scale=1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func=mdp.foot_height,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # Set per-robot.
    ),
    "foot_air_time": ObservationTermCfg(
      func=mdp.foot_air_time,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func=mdp.foot_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
    # Arm trajectory INTENT (commanded arm joint velocity, rad/s). Privileged:
    # the critic can anticipate an arm-induced momentum transient the instant
    # it is commanded, which lowers value-target variance under fast arm
    # motion. Deliberately NOT in the actor group: that would widen the 71-D
    # deployment interface and require the C++ controller to publish dq_ref
    # (see ``mdp.arm_traj_vel``). The actor still sees the arms through
    # joint_pos / joint_vel, exactly as before.
    "arm_traj_vel": ObservationTermCfg(
      func=mdp.arm_traj_vel,
      params={"action_term_name": "upper_body_pose"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }

  # Diagnostics (Episode_Metrics/*): per-step averages over the episode, no
  # weights. These are what say whether robustness is improving -- torso tilt,
  # stationary drift, CoM support margin, slip, tracking -- alongside the arm
  # disturbance magnitude actually applied. Robot-specific bodies/sites are
  # filled in per-robot like the rewards.
  metrics = {
    "mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc),
    "torso_pitch_abs": MetricsTermCfg(
      func=mdp.torso_pitch_abs,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Per-robot.
    ),
    "torso_roll_abs": MetricsTermCfg(
      func=mdp.torso_roll_abs,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Per-robot.
    ),
    "idle_root_drift": MetricsTermCfg(func=mdp.idle_root_drift),
    "com_support_error": MetricsTermCfg(
      func=mdp.com_support_error,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Per-robot.
      },
    ),
    "foot_slip_speed": MetricsTermCfg(
      func=mdp.foot_slip_speed,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Per-robot.
      },
    ),
    "arm_traj_speed": MetricsTermCfg(func=mdp.arm_traj_speed),
    "arm_traj_accel": MetricsTermCfg(func=mdp.arm_traj_accel),
    "lin_vel_track_error": MetricsTermCfg(
      func=mdp.lin_vel_track_error, params={"command_name": "twist"}
    ),
    "yaw_track_error": MetricsTermCfg(
      func=mdp.yaw_track_error, params={"command_name": "twist"}
    ),
    "height_track_error": MetricsTermCfg(
      func=mdp.height_track_error,
      params={
        "command_name": "base_height",
        "ankle_sole_distance": 0.02,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Per-robot.
      },
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),  # Narrowed to the lower body per-robot.
      scale=0.25,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.20,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 1.0),
        lin_vel_y=(-0.4, 0.4),
        ang_vel_z=(-0.8, 0.8),
        heading=(-math.pi, math.pi),
      ),
    ),
    # HOMIE contribution (b). Ranges are robot-specific and set in config/;
    # defined here so the base config is self-consistent (both the
    # `height_command` observation and the `track_base_height` reward read it).
    "base_height": mdp.BaseHeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
    ),
  }

  ##
  # Events
  ##

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.0),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # FINAL push distribution. The per-robot config installs a ``push_magnitude``
    # curriculum that scales these ranges up from ~10% over training, so a
    # newborn policy learns to walk before it learns to be shoved.
    # push_and_relatch_anchor: same impulse as mjlab's push_by_setting_velocity,
    # plus a re-latch of the idle position anchor for the pushed envs, so the
    # anchor is not charging for displacement the push caused (see events.py).
    "push_robot": EventTermCfg(
      func=mdp.push_and_relatch_anchor,
      mode="interval",
      interval_range_s=(3.0, 5.0),
      params={
        "velocity_range": {
          "x": (-0.7, 0.7),
          "y": (-0.7, 0.7),
          "z": (-0.3, 0.3),
          "roll": (-0.4, 0.4),
          "pitch": (-0.4, 0.4),
          "yaw": (-0.6, 0.6),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }

  ##
  # Rewards -- 22 terms. See documents/duet/reward_design.md.
  ##

  rewards = {
    # -- Task: what the policy is paid to do. ------------------------------
    # The primary objective. The embedded 2*v_z^2 replaces HOMIE's separate
    # lin_vel_z penalty. HOMIE splits this 1.5 (x) + 1.0 (y).
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=1.5,
      params={"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    # Weighted ABOVE linear tracking: the arms are externally driven, so the
    # policy cannot counter-rotate the upper body to make yaw and must produce
    # it entirely through foot placement. HOMIE weights it 2.0 for this reason.
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.75,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    # HOMIE contribution (b). Height measured above the LOWER (stance) foot so
    # the reference does not move when the swing foot lifts -- with the higher
    # foot as reference the pelvis bobs once per stride. HOMIE: 2.0.
    "track_base_height": RewardTermCfg(
      func=mdp.track_base_height,
      weight=2.0,
      params={
        "command_name": "base_height",
        "ankle_sole_distance": 0.02,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # -- Postural stability: what keeps it upright while the arms move. -----
    # Measured on torso_link, not pelvis, so a waist_yaw excursion is visible.
    # The single term that most directly buys "does not tip while reaching".
    # Height-gated: full weight while standing tall or walking (uprightness is
    # what keeps the robot alive there), relaxed to 30% for deep squats so the
    # torso can lean forward to reach the floor. The 23-DOF G1 has no waist
    # pitch, so torso pitch IS the whole body rotating about the ankles -- an
    # ungated penalty opposes ground-reaching directly.
    # Weight -1.5 -> -2.5 (arm-robustness pass). At -1.5 a 5 deg backward lean
    # cost 0.011/s, which made "pitch the torso back" the cheapest answer to an
    # arms-forward CoM shift once the pelvis was anchored with no deadband. The
    # anchor now has a deadband and the CoM term rewards the alternative, so
    # this is a moderate nudge, not the fix; the gate is unchanged.
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-2.5,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Per-robot.
        "height_command_name": "base_height",
        "relax_below": 0.30,
        "full_above": 0.55,
        "relax_factor": 0.30,
      },
    ),
    # Torso pitch/roll RATE. Damps the low-frequency torso oscillation that is
    # the dominant hardware failure mode under payload. HOMIE ang_vel_xy -0.025.
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Per-robot.
    ),
    # Speed-scheduled per-joint deviation from default. Balance-critical joints
    # (hip_roll/yaw, ankle_roll, waist) held tight; squat joints (hip_pitch,
    # knee, ankle_pitch) left loose so a commanded squat is not fought. HOMIE
    # uses three separate deviation_{hip,knee,ankle} penalties for the same job.
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),  # Per-robot.
        "command_name": "twist",
        "std_standing": {},  # Set per-robot.
        "std_walking": {},  # Set per-robot.
        "std_running": {},  # Set per-robot.
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
        # Squat regime: blend to much looser tolerances on the folding joints as
        # a low pelvis height is commanded. Without this the regulariser pulls
        # hip_pitch/knee/ankle toward the STANDING default exactly when a deep
        # fold is asked for, opposing track_base_height. Measured: squat depth
        # was pinned at ~0.25 m against a 0.18 m command across 11 consecutive
        # checkpoints and did not respond to more training.
        "std_squatting": {},  # Set per-robot.
        "height_command_name": "base_height",
        "squat_below": 0.35,
        "full_above": 0.60,
      },
    ),
    # Stance width in [0.20, 0.35] m. A narrow stance is precisely what tips a
    # half-squat + arms-forward pose. A HOMIE term (feet_distance_lateral 0.5 +
    # knee_distance_lateral 1.0, identical band); we keep only the feet half,
    # hence the reduced weight.
    "feet_distance": RewardTermCfg(
      func=mdp.feet_lateral_distance,
      weight=-0.75,
      params={
        "min_distance": 0.20,
        "max_distance": 0.35,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Pelvis xy velocity + yaw rate, gated to zero-twist envs. NOT a drift
    # patch: track_linear_velocity is an exponential whose gradient vanishes as
    # error -> 0 (at 0.1 m/s it still pays 0.96), so it gives almost no signal
    # in exactly the regime that matters for stationary manipulation. This
    # quadratic restores the near-zero gradient. Replaces HOMIE's stand_still,
    # which is a joint-deviation penalty and therefore fights commanded squats.
    "idle_base_motion": RewardTermCfg(
      func=mdp.idle_base_motion,
      weight=-1.0,
      params={"command_name": "twist", "command_threshold": 0.1},
    ),
    # -- Idle precision (added after hardware testing). ---------------------
    # Displacement from where the robot last stopped, L2 norm (not squared) so
    # the gradient stays constant down to zero error. idle_base_motion above
    # penalizes VELOCITY quadratically, which is nearly blind to a slow creep:
    # 0.05 m/s costs 0.0025 per step but is half a metre after ten seconds.
    # This is also the term that forces the legs to reject the forward CoM
    # shift when the arms extend, which otherwise shows up purely as drift.
    # Weight raised -3.0 -> -5.0 (2026-08-07). At -3.0 a 15 mm error costs only
    # -0.045/s, which measurably failed to suppress a systematic forward creep
    # (73% of envs drifting the same direction, +15 mm in 11 s, most of it in
    # the first 3 s). The gradient is constant by construction; it just needed
    # to be large enough to matter at the millimetre scale precision work needs.
    # Deadband added (arm-robustness pass): a 2.5 cm xy / ~1 deg yaw band costs
    # nothing. With a zero-width anchor the -5.0 weight made ANY pelvis travel
    # expensive, so the policy absorbed an arms-forward CoM shift by holding the
    # pelvis fixed and leaning the torso back instead of the natural ankle/hip
    # shift. Inside the band the pelvis may move; beyond it drift is penalised
    # exactly as before (constant gradient, clamped).
    "idle_position_anchor": RewardTermCfg(
      func=mdp.idle_position_anchor,
      weight=-5.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "yaw_weight": 0.5,
        "max_error": 1.0,
        "xy_deadband": 0.025,
        "yaw_deadband": 0.02,
      },
    ),
    # Whole-body CoM (arms, hands and payload included -- MuJoCo subtree_com,
    # not the pelvis) kept inside a conservative ellipse between the feet.
    # Zero inside half the region, quadratic toward the edge, linear beyond it.
    # Double-support only and scaled to 25% while a locomotion command is
    # active, so it shapes stationary balance without fighting the single-
    # support phases of walking. This is the term that gives credit for the
    # pelvis/ankle compensation the deadband above makes free.
    "com_support": RewardTermCfg(
      func=mdp.com_support_region,
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "foot_half_length": 0.09,
        "foot_half_width": 0.04,
        "safe_fraction": 0.5,
        "outside_gain": 2.0,
        "moving_scale": 0.25,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Catches the shuffle step that returns a foot to the same place: invisible
    # to the anchor above, but it still breaks the fixed base frame that precise
    # manipulation needs. Acts on foot motion, not joint deviation, so it does
    # not fight a commanded squat.
    # Weight halved -0.5 -> -0.25 (2026-08-07). At -0.5 this forbade the small
    # corrective step the robot used to take when the payload-loaded arms pull
    # the CoM forward, so instead of stepping to catch itself it crept forward.
    # The 7000 policy, which predates this term, drifts BACKWARD (-18 mm); the
    # checkpoints trained with it drift forward. Discouraging a shuffle is
    # wanted; removing the ankle/step correction entirely is not.
    "idle_feet_still": RewardTermCfg(
      func=mdp.idle_feet_still,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # -- Gait shaping: what makes the walk clean enough to deploy. ----------
    # Rewards contact state agreeing with the commanded 0.6 s phase clock. This
    # is what the 2-D `phase` observation exists for, and the strongest single
    # anti-shuffle term.
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      },
    ),
    # Cost = |z_foot - target| * horizontal foot speed, i.e. it penalises
    # TRAVELLING at the wrong height. Target 0.10 m (HOMIE 0.14): the previous
    # task inflated this to 0.15 to force a march, which cost energy and needed
    # a second term (feet_drag) that computes the same product. Both reverted.
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Foot xy speed^2 while in contact. Directly the sim2real term: slip in sim
    # becomes a fall on a real floor whose friction we do not know. HOMIE -0.25.
    # always_active: the default gate switches this OFF at zero command, which
    # left sliding feet unpenalized in exactly the regime where a stationary
    # base matters most. A planted foot that slips is equally bad standing.
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "always_active": True,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    # Single-stance duration near 0.4 s. Prevents the degenerate high-frequency
    # shuffle that satisfies velocity tracking without taking real steps.
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=0.05,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.4,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    # -- Safety and regularisation. ----------------------------------------
    # Rewards are dt-scaled by the manager, so this is a one-time -4.0 against
    # a ~136 full-episode return (~3%). It marks falls as bad without making
    # the policy freeze; the real cost of falling is the forfeited episode.
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    # A 0.12 m squat necessarily folds thigh against torso, so soft self-contact
    # is CORRECT behaviour here. The 50 N threshold flags only hard collisions.
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.25,
      params={"sensor_name": "self_collision", "force_threshold": 50.0},
    ),
    # Barrier term. Hitting a soft limit is a hard failure on hardware (the SDK
    # clamps, and the tracking error becomes a torque spike). Should read ~0 in
    # a converged policy -- if it does not, that is a bug, not a tuning issue.
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    # First-difference of actions. Jitter is the primary sim2real killer at
    # 50 Hz into a soft PD loop. HOMIE splits action_rate -0.01 + smoothness
    # -0.05 (2nd order); one term at -0.1 covers both.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    # Second-order action smoothness (jerk), added 2026-09-16 after the v4
    # policy was judged good but visibly twitchy. action_rate_l2 above is a
    # FIRST difference and over half its cost was exploration noise, which
    # inference never emits; the second difference -- the quantity that reads
    # as jitter -- had no term at all. Sized so its cost (~0.07/s against a
    # measured 3.6 sum-of-squares) is about 40% of action_rate_l2's, i.e.
    # real pressure without dominating the objective. See the docstring on
    # mdp.action_smoothness_l2 for why this instrument and not a bigger
    # action_rate_l2.
    "action_smoothness_l2": RewardTermCfg(
      func=mdp.action_smoothness_l2, weight=-0.02
    ),
    # Damps twitchy leg oscillation. Restricted to the RL-controlled joints
    # per-robot (penalising the externally-driven arms would be meaningless).
    "joint_vel_l2": RewardTermCfg(
      func=envs_mdp.joint_vel_l2,
      weight=-1e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},  # Per-robot.
    ),
    # Torque-rate proxy; protects the gearboxes. HOMIE dof_acc -2.5e-7 exactly.
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # Near-horizontal states are unrecoverable; letting episodes linger there
    # only pollutes the batch. HOMIE terminates at ~53 deg.
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(60.0)},
    ),
  }

  ##
  # Curriculum (stages set per-robot; all read env.common_step_counter, which
  # IS checkpointed, so every ramp resumes correctly with no source edits).
  ##

  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(DUET_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    metrics=metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=1500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
