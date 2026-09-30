"""DUET environment configuration for the Unitree G1-23DOF.

Sets everything robot-specific on top of ``duet_env_cfg.make_duet_env_cfg()``:
the lower-body/arm joint carve-out, PD gains and action scale (which MUST match
``deploy.yaml``), payload and CoM randomisation, curriculum timing, and the
per-joint posture tolerances.

Three HOMIE contributions, each independently ablatable via the arguments of
:func:`unitree_g1_23dof_duet_rough_env_cfg`:

  a. ``arm_mode``    -- upper-body disturbance generator (trapezoidal
                        trajectories over a safe workspace, curriculum-ramped)
  b. ``height_mode`` -- pelvis height as a first-class command
  c. symmetry        -- set on the runner, see ``rl_cfg.py``

Training curriculum (arm-robustness pass). Every schedule below is a list of
(iteration, value) knots installed by :func:`apply_duet_curriculum`; the
stages overlap rather than isolate tasks:

  A  0-1500     locomotion foundation: reduced command envelope, near-nominal
                height, mild arm motion, ~10% pushes
  B  1500-3000  arm disturbances begin: full basic command range, trapezoidal
                arm trajectories opening up, height still near nominal
  C  3000-5000  height variation: squat floor deepens, arm workspace and
                dynamics keep growing, pushes at ~65%
  D  5000-7000  approach the full task: full command range, full arm workspace /
                velocity / acceleration, full pushes. The squat floor reaches
                its final 0.27 m at 6000 and is constant thereafter.
  E  7000+      full distribution, held fixed until the end of training

A mixture runs throughout: a slice of envs (50% -> 10%) keeps mild arm motion
and a slice of height commands (50% -> 10%) stays at the nominal height, so
clean locomotion is never crowded out of the batch.

Curriculum state note: every ramp here is keyed on ``env.common_step_counter``,
which ``MjlabOnPolicyRunner`` writes into and restores from the checkpoint.
Resuming therefore continues the curriculum where it stopped. There is no
``RESUME_AT_FULL_DISTRIBUTION`` switch and no source edit is required to resume.
"""

import dataclasses
import math

import mjlab.terrains as terrain_gen
from mjlab.actuator import DelayedActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from src.assets.robots import (
  G1_23DOF_ACTION_SCALE,
  G1_23DOF_DEPLOY_ACTION_SCALE,
  get_g1_23dof_deploy_robot_cfg,
  get_g1_23dof_robot_cfg,
)
from src.assets.robots.unitree_g1.g1_23dof_constants import (
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
)
from src.tasks.common.terrains import HfScaledRandomUniformTerrainCfg
import src.tasks.duet.mdp as mdp
from src.tasks.duet.duet_env_cfg import ITER, make_duet_env_cfg

##
# Curriculum timing, sized for the 25k-iteration budget: every ramp saturates
# at 7,000 iterations, leaving ~18k iterations of training on the full task
# distribution. All knots below are written against this horizon and scaled
# by ``apply_duet_curriculum`` when a shorter horizon is requested (ablations).
##

CURRICULUM_END_ITERS = 7_000

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
#
# Floor 0.27 -> 0.24 on 2026-09-16 (second pass). Measured by constrained
# optimisation over the leg chain: the deepest pelvis height reachable with the
# foot flat and EVERY joint strictly inside its soft limits is 0.204 m, and that
# pose pins hip_pitch and ankle_pitch exactly at their stops. Allowing 0.10 rad
# of margin gives 0.235 m, 0.20 rad gives 0.264 m. 0.24 keeps ~0.13 rad of
# margin on the binding joints, so a commanded full-depth squat still has room
# to reject a push without driving a joint into its stop (where the SDK clamps
# the target and the motor holds a stall current). Deeper than ~0.22 is not
# free: it trades that margin away. The previous floor is described below.
#
# Floor raised 0.18 -> 0.27 on 2026-09-16, on the operator's call, after the
# arm-robustness run tracked the descending curriculum floor down to 0.27 m with
# height error at or below 0.025 m the whole way -- deep enough for the intended
# manipulation workspace. Height error made the first upward move of that run as
# the ramp continued below 0.27 m, so the remaining depth was being bought at a
# measurable cost in accuracy.
#
# This is the COMMAND range, not merely the curriculum floor, and the two MUST
# agree: `build_deploy_metadata` records it as `height_command_range`, so
# leaving 0.18 here while training only to 0.27 would make the exported ONNX
# claim a trained depth the policy never saw, and would let
# `check_deploy_consistency.py` pass a deploy.yaml permitting an untrained
# command. Expect a consequence: deploy.yaml still carries [0.18, 0.73] for the
# PREVIOUSLY deployed checkpoint, so the checker now fails its
# "base_height range within trained range" test until that file is narrowed to
# [0.27, 0.73] alongside a policy exported from this run. That failure is the
# mechanism working, not a regression.
HEIGHT_RANGE = (0.24, 0.73)
WALK_MIN_HEIGHT_FINAL = 0.60  # walking is restricted to >= 0.60 m

# Squat-floor schedule, (iteration, floor m): near-nominal through the
# locomotion + arm-introduction stages, then deepening to HEIGHT_RANGE[0].
# The final knot sits at 6000, not 7000, because that is where the frozen run
# resumes; `piecewise_linear` holds the last value for every later step, so
# nothing re-deepens the floor afterwards.
SQUAT_FLOOR_STAGES = (
  (0, 0.68), (1_500, 0.68), (3_000, 0.60), (5_000, 0.38), (6_000, HEIGHT_RANGE[0])
)

# Velocity command envelope, staged (iteration -> ranges). Stage A is a
# reduced envelope so the newborn policy learns stand / walk / strafe / turn /
# stop before speed; the final stage is HOMIE's range with yaw widened to +-1.0
# to cover deploy.yaml's joystick mapping.
VELOCITY_STAGES = (
  (0, {"lin_vel_x": (-0.3, 0.6), "lin_vel_y": (-0.2, 0.2), "ang_vel_z": (-0.4, 0.4)}),
  (1_500, {"lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.4, 0.4), "ang_vel_z": (-0.8, 0.8)}),
  (3_500, {"lin_vel_x": (-0.75, 1.0), "lin_vel_y": (-0.45, 0.45), "ang_vel_z": (-0.9, 0.9)}),
  # Final range made SYMMETRIC in x on 2026-09-16 (was -0.8..1.2). Measured on
  # model_19000, backward tracking was short at every speed while forward was
  # near exact: at |0.2| the robot managed 0.285 forward but only 0.104 back,
  # and every one of the nine sweep errors pointed forward. An asymmetric range
  # trains the two directions unequally for no operational gain -- deploy.yaml
  # clamps the joystick to +-0.5, so the old 1.2 ceiling was never commanded on
  # the robot while the -0.8 floor was the binding one.
  (5_000, {"lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)}),
)
VELOCITY_FINAL = VELOCITY_STAGES[-1][1]
# Each stage above is blended in over this many iterations rather than switched
# at once (see ``curriculums.commands_vel``): the one-iteration doubling of the
# envelope at 1500 was where the first run's action-std runaway began.
VELOCITY_RAMP_ITERS = 500

# Command mixture, staged: standing fraction, walk-band floor, and the
# nominal-height slice (mixture curriculum, height half).
COMMAND_MIX_STAGES = (
  (0, {"rel_standing_envs": 0.20, "walk_min_height": 0.71, "nominal_height_fraction": 0.50}),
  (1_500, {"rel_standing_envs": 0.25, "walk_min_height": 0.71, "nominal_height_fraction": 0.40}),
  (3_000, {"rel_standing_envs": 0.30, "walk_min_height": 0.68, "nominal_height_fraction": 0.30}),
  (5_000, {"rel_standing_envs": 0.30, "walk_min_height": 0.64, "nominal_height_fraction": 0.20}),
  (7_000, {"rel_standing_envs": 0.30, "walk_min_height": WALK_MIN_HEIGHT_FINAL, "nominal_height_fraction": 0.10}),
)

# Push magnitude as a fraction of the final ``push_robot`` velocity range.
PUSH_SCALE_STAGES = ((0, 0.10), (1_000, 0.30), (3_000, 0.65), (5_000, 1.0))

##
# Arm disturbance generator (see ``mdp.UpperBodyPoseActionCfg``).
##

# Safe joint workspace (rad) for INDEPENDENT per-joint sampling. A conservative
# box inside the soft limits chosen so that any combination is mechanically
# reasonable: shoulder roll never swings the arm through the torso, yaw and
# elbow are kept short of the range where a bent forearm sweeps the chest.
# Sign conventions: negative shoulder pitch raises the arm forward/up (default
# hang 0.35); positive left / negative right shoulder roll abducts outward
# (defaults +-0.18); elbow default 0.87. Arm-arm contact in rare combinations
# (both arms far forward with inward yaw) is still possible and is tolerated:
# the self-collision term only flags hard (>50 N) contacts.
ARM_WORKSPACE_LIMITS = {
  r".*_shoulder_pitch_joint": (-2.0, 1.0),  # high reach ... hand behind hip
  "left_shoulder_roll_joint": (-0.25, 1.6),
  "right_shoulder_roll_joint": (-1.6, 0.25),
  r".*_shoulder_yaw_joint": (-1.0, 1.0),
  r".*_elbow_joint": (-0.5, 1.6),
  r".*_wrist_roll_joint": (-1.5, 1.5),
}

# Final trajectory dynamics (per joint, per segment, sampled uniformly).
# vmax up to 3.5 rad/s and amax up to 20 rad/s^2 cover a fast VLA / teleop
# motion with margin; the initial ranges are what stage A trains against.
ARM_TRAJ_VMAX_RANGE = (0.3, 3.5)  # rad/s
ARM_TRAJ_VMAX_RANGE_INIT = (0.2, 0.8)
ARM_TRAJ_AMAX_RANGE = (1.0, 20.0)  # rad/s^2
ARM_TRAJ_AMAX_RANGE_INIT = (0.5, 3.0)
ARM_TARGET_HOLD_RANGE = (0.2, 2.5)  # s
ARM_WORKSPACE_SCALE = 1.0  # fraction of ARM_WORKSPACE_LIMITS at full curriculum

# Curriculum ratio (iteration, ratio): scales the workspace and interpolates
# the dynamics ranges from _INIT to final. Stage A keeps it small.
ARM_RATIO_STAGES = ((0, 0.15), (1_500, 0.15), (3_000, 0.50), (5_000, 0.75), (7_000, 1.0))
# Mixture curriculum, arm half: fraction of envs held at the mild "clean" ratio.
ARM_CLEAN_FRACTION_STAGES = ((0, 0.50), (1_500, 0.40), (3_000, 0.30), (5_000, 0.20), (7_000, 0.10))
ARM_CLEAN_ENV_RATIO = 0.10

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

##
# Rough-blind-tall pass (2026-09), documents/duet/rough_blind_task.md. Every
# value below is used only through a keyword argument of
# ``unitree_g1_23dof_duet_rough_env_cfg`` whose default reproduces the tasks
# that existed before this pass.
##

# Nominal height. Measured with scripts/duet_stand_height.py (Mode 1: default
# pose, zero action, rigid plane, pelvis held level with its height free,
# settled): h = 0.7949 m in the track_base_height convention, so height_max =
# 0.79 (rounded down to 0.01). The old top of the command range, 0.73, sat 6 cm
# below the pose the regulariser pulls toward.
RB_TALL_HEIGHT_MAX = 0.79

# Foot-site height above rigid flat ground at rest, measured by the same
# script: the site sits 2.1 mm BELOW the sole (the capsule bottoms are at
# z = -0.035 in the ankle-roll frame, the site at -0.037). Used by the
# terrain-relative foot clearance (mdp.feet_clearance, reference="stance_foot").
FOOT_SITE_Z_REST = -0.0021

# Contact softness of the foot geoms (mdp.geom_solref, solref timeconst, s).
# Measured with scripts/duet_contact_softness_sweep.py on the rigid plane: the
# static sinkage at 0.30 s is 8.6 mm (0.20 s: 4.6 mm, 0.10 s: 1.4 mm), about 5x
# less than the single-contact formula predicts, because the 28 foot contact
# points share the load and each is regularised on its own. The operator's
# litter sinkage is not measured yet, so the brief's 25 mm target applies,
# which no timeconst under the 0.30 s hard cap reaches: tc_max is the cap. A
# 5 cm drop at 0.30 s sinks 45 mm at most, inside the 50 mm tunnelling limit.
FOOT_TC_MIN = 0.02
FOOT_TC_MAX = 0.30
# Ramp of the upper bound, (iteration, tc_max): rigid first, soft from 3000.
FOOT_TC_MAX_STAGES = (
  (0, FOOT_TC_MIN),
  (1_000, FOOT_TC_MIN + 0.3 * (FOOT_TC_MAX - FOOT_TC_MIN)),
  (3_000, FOOT_TC_MAX),
)
FOOT_TC_INTERVAL_S = (1.0, 3.0)  # mid-episode re-draw: patches of litter

# Foot friction for the new tasks (HOMIE 0.1-2.0, DWL 0.2-2.0): 0.2 is a clean
# slip without making walking impossible. The existing tasks keep (0.3, 1.6).
RB_FOOT_FRICTION = (0.2, 1.6)

# Sim-to-real randomisation. PD gains +-10% around nominal (HOMIE's range) on
# the 13 lower-body joints; actuator latency 0-20 ms (4 physics steps of 5 ms)
# on the leg and waist actuators, constant within an episode.
RB_PD_GAIN_RANGE = (0.9, 1.1)
RB_DELAY_MAX_LAG = 4
# DelayBuffer re-samples a lag when (step_count + phase) % update_period == 0,
# and reset() zeroes step_count. With no hold probability, no per-env phase and
# a period longer than any episode (1e9 physics steps = 58 days of sim time),
# the lag is drawn once at the first physics step after each reset and then
# held: constant within an episode, re-sampled at reset, never jittering.
RB_DELAY_UPDATE_PERIOD = 1_000_000_000

# Task-success terrain curriculum thresholds (mdp.terrain_levels_task).
# Relaxed once (brief section 9), 2026-09-30 at iteration 3000 of the H5 run
# 2026-09-30_00-17-59_rough_blind_tall_h5: terrain_level_mean was 0.00 with no
# promotion since the start (episode means lin 0.41 m/s, yaw 0.90 rad/s at
# 2000). Was lin 0.20 / yaw 0.35.
RB_TERRAIN_PROMOTE = {
  "min_moving_fraction": 0.25,
  "max_lin_vel_error": 0.25,
  "max_yaw_error": 0.45,
  "max_height_error": 0.04,
  "max_idle_drift": 0.05,
}
RB_TERRAIN_DEMOTE_LIN_VEL_ERROR = 0.35

# Contact buffer for the litter terrain (per world, pooled across worlds). The
# rough task's 48 cannot even build here: mujoco_warp's put_data checks it
# against one CPU MjData at qpos0 -- straight legs at the world origin -- which
# overlaps whatever tile is there (measured 117 contacts on the training grid,
# up to 143 on the random play/eval grids; capsule-vs-heightfield collisions
# make many contacts). At runtime standing on the heightfields uses 24/world on
# average with step peaks of 43-47, i.e. 90-98% of 48. 256 costs no measurable
# throughput (19.9k vs 19.8k env-steps/s at 4096 envs for 128 vs 256).
LITTER_NCONMAX = 256

# Spawn height offset on the litter terrain (added to reset_base's z). Measured
# on the hardest row with rigid feet: waves tiles put a wave crest up to 4 cm
# above the tile origin inside the +-0.5 m spawn window while the sole starts
# 1.6 cm above the origin, so feet spawned up to 14 mm inside the surface and
# the robot popped off the ground at 0.35 m/s. Every other sub-terrain spawned
# clean. 2.5 cm clears the worst case (2.4 cm).
LITTER_SPAWN_Z = 0.025

FOOT_GEOM_NAMES = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

# The deployed actor interface: the -Flat task's actor terms, in order (see
# the observation contract at the top of src/tasks/duet/duet_env_cfg.py).
DEPLOY_ACTOR_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "command",
  "phase",
  "joint_pos",
  "joint_vel",
  "actions",
  "height_command",
)


def _delayed_lower_body_robot_cfg() -> EntityCfg:
  """A fresh G1-23DOF cfg whose leg and waist actuators carry 0-20 ms latency.

  Built here rather than in g1_23dof_constants.py, whose action-scale loop
  asserts BuiltinPositionActuatorCfg. The wrapped groups are the same objects
  with the same nominal gains, so G1_23DOF_ACTION_SCALE and the exported gains
  are unchanged; the arm group (5020) is not delayed.
  """
  robot = get_g1_23dof_robot_cfg()
  assert robot.articulation is not None
  wrap = (G1_ACTUATOR_7520_14, G1_ACTUATOR_7520_22, G1_ACTUATOR_ANKLE)
  actuators = tuple(
    DelayedActuatorCfg(
      base_cfg=act,
      delay_target="position",
      delay_min_lag=0,
      delay_max_lag=RB_DELAY_MAX_LAG,
      delay_hold_prob=0.0,
      delay_update_period=RB_DELAY_UPDATE_PERIOD,
      delay_per_env_phase=False,
    )
    if any(act is w for w in wrap)
    else act
    for act in robot.articulation.actuators
  )
  assert sum(isinstance(a, DelayedActuatorCfg) for a in actuators) == len(wrap)
  robot.articulation = dataclasses.replace(robot.articulation, actuators=actuators)
  return robot


def litter_terrain_generator_cfg() -> TerrainGeneratorCfg:
  """Poultry-litter-like terrain: small bumps, gentle undulation, caked clumps.

  Litter is loose shavings, hulls or straw, several centimetres (up to ~15 cm)
  deep. Every sub-terrain is scaled by difficulty (row), and every surface
  statistic was measured with ``scripts/duet_terrain_stats.py`` (gate G6).
  Columns are allocated by proportion over 20 columns (3 flat, 6 bumps,
  4 undulation, 2 waves, 2 slope, 1 inverted slope, 2 clumps).
  """
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    curriculum=True,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.15),
      # Fine bumps: 0 at difficulty 0, up to 4 cm in 5 mm steps at 1.
      "fine_bumps": HfScaledRandomUniformTerrainCfg(
        proportion=0.30,
        noise_range=(0.0, 0.04),
        noise_step=0.005,
        border_width=0.25,
      ),
      # Gentle undulation: peak-to-peak up to 6 cm, wavelength ~1.9 m with
      # octaves down to ~0.5 m. The 5 mm floor is not cosmetic: at exactly 0
      # the heightfield has zero height and MuJoCo refuses to compile it.
      # 10 cm cells, not mjlab's 5 cm default: at 5 cm a fallen body overlaps
      # 50+ cells and mujoco_warp drops contacts ("height field collision
      # overflow"); measured, this sub-terrain produced all of them.
      "undulation": terrain_gen.HfPerlinNoiseTerrainCfg(
        proportion=0.20,
        height_range=(0.005, 0.06),
        octaves=3,
        persistence=0.5,
        lacunarity=2.0,
        scale=5.0,
        horizontal_scale=0.1,
        resolution=0.1,
        border_width=0.25,
      ),
      # Waves: amplitude up to 4 cm about the mean (8 cm peak-to-peak).
      "waves": terrain_gen.HfWaveTerrainCfg(
        proportion=0.10,
        amplitude_range=(0.0, 0.04),
        num_waves=4,
        border_width=0.25,
      ),
      # Gentle floor slope up to 0.15 (~8.5 deg), both directions.
      "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.075,
        slope_range=(0.0, 0.15),
        platform_width=2.0,
        border_width=0.25,
      ),
      "slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=0.075,
        slope_range=(0.0, 0.15),
        platform_width=2.0,
        border_width=0.25,
        inverted=True,
      ),
      # Caked clumps: 0.3 m cells, heights in +-2.5 cm at difficulty 1.
      # Equal-height neighbours are MERGED (heights snap to 1.25 cm levels:
      # 0, +-1.25, +-2.5 cm). Measured: as individual boxes the two clump
      # columns are ~12,500 geoms, which pushes mujoco_warp past its 250k
      # candidate-pair limit onto the segmented sweep-and-prune broadphase;
      # there, at 1024 envs, collision detection silently returns NO contacts
      # (robots fall through the floor, no warning), and at 4096 envs CUDA
      # graph creation runs out of memory. Merged, the scene is ~6,200 geoms
      # and keeps the n-squared broadphase (4096 envs, 19.9k env-steps/s).
      "clumps": terrain_gen.BoxRandomGridTerrainCfg(
        proportion=0.10,
        grid_width=0.3,
        grid_height_range=(0.0, 0.025),
        platform_width=1.0,
        merge_similar_heights=True,
        height_merge_threshold=0.0125,
      ),
    },
    add_lights=True,
  )


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


# Deployment-pose anchors. Signs: negative shoulder pitch raises the arm
# forward/up; default hang is pitch 0.35, elbow 0.87, roll +-0.18. Used by
# ``arm_mode="anchored"`` as a MINORITY of goal draws and by the evaluation
# probes (``scripts/duet_probe_idle.py``, ``scripts/duet_arm_scenarios.py``)
# as static test poses. The default task distribution is uniform over
# ``ARM_WORKSPACE_LIMITS`` so the policy is not biased toward these.
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


def apply_duet_curriculum(
  cfg: ManagerBasedRlEnvCfg, end_iters: int = CURRICULUM_END_ITERS
) -> None:
  """Install every staged schedule, scaled so all ramps saturate at ``end_iters``.

  The knots at module level are written against the 7,000-iteration horizon;
  a shorter horizon (ablations) scales every knot's iteration by the same
  factor so the stages keep their relative timing. Curriculum state is read
  from ``env.common_step_counter``, which is checkpointed, so resuming
  continues every ramp where it stopped.
  """
  f = end_iters / CURRICULUM_END_ITERS

  def it(iters: float) -> int:
    return int(round(iters * f)) * ITER

  arm = cfg.actions["upper_body_pose"]
  assert isinstance(arm, mdp.UpperBodyPoseActionCfg)
  arm.ratio_stages = tuple((it(i), r) for i, r in ARM_RATIO_STAGES)
  arm.clean_fraction_stages = tuple((it(i), r) for i, r in ARM_CLEAN_FRACTION_STAGES)

  base_height_cmd = cfg.commands["base_height"]
  assert isinstance(base_height_cmd, mdp.BaseHeightCommandCfg)
  base_height_cmd.floor_curriculum_start = SQUAT_FLOOR_STAGES[0][1]
  base_height_cmd.floor_curriculum_steps = it(CURRICULUM_END_ITERS)
  base_height_cmd.floor_stages = tuple((it(i), h) for i, h in SQUAT_FLOOR_STAGES)
  base_height_cmd.walk_min_height = COMMAND_MIX_STAGES[0][1]["walk_min_height"]
  base_height_cmd.nominal_env_fraction = COMMAND_MIX_STAGES[0][1][
    "nominal_height_fraction"
  ]

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = COMMAND_MIX_STAGES[0][1]["rel_standing_envs"]
  for axis, rng in VELOCITY_STAGES[0][1].items():
    setattr(twist_cmd.ranges, axis, rng)

  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [{"step": it(i), **r} for i, r in VELOCITY_STAGES],
      "ramp_steps": it(VELOCITY_RAMP_ITERS),
    },
  )
  cfg.curriculum["command_mix"] = CurriculumTermCfg(
    func=mdp.command_mix,
    params={
      "twist_command_name": "twist",
      "height_command_name": "base_height",
      "stages": [{"step": it(i), **r} for i, r in COMMAND_MIX_STAGES],
    },
  )
  cfg.curriculum["push_magnitude"] = CurriculumTermCfg(
    func=mdp.push_magnitude,
    params={
      "event_name": "push_robot",
      "base_velocity_range": dict(cfg.events["push_robot"].params["velocity_range"]),
      "scale_stages": [(it(i), s) for i, s in PUSH_SCALE_STAGES],
    },
  )
  # Log-only: surfaces the arm generator's ratio on the training curves.
  cfg.curriculum["arm_curriculum_state"] = CurriculumTermCfg(
    func=mdp.arm_curriculum_state, params={"action_term_name": "upper_body_pose"}
  )
  # The push event keeps its FULL range in the cfg. The curriculum manager runs
  # on every reset, including the initial env.reset(), so the stage-0 scale is
  # applied before the first push can fire (interval >= 3 s); and any
  # evaluation script that clears ``cfg.curriculum`` gets full pushes, as before.


def pin_duet_full_distribution(cfg: ManagerBasedRlEnvCfg) -> None:
  """No staging: sample the final (hardest) task distribution from step 0."""
  arm = cfg.actions["upper_body_pose"]
  assert isinstance(arm, mdp.UpperBodyPoseActionCfg)
  arm.init_ratio = 1.0
  arm.clean_fraction_stages = ((0, ARM_CLEAN_FRACTION_STAGES[-1][1]),)
  base_height_cmd = cfg.commands["base_height"]
  assert isinstance(base_height_cmd, mdp.BaseHeightCommandCfg)
  base_height_cmd.floor_curriculum_start = HEIGHT_RANGE[0]
  base_height_cmd.floor_stages = None
  final_mix = COMMAND_MIX_STAGES[-1][1]
  base_height_cmd.walk_min_height = final_mix["walk_min_height"]
  base_height_cmd.nominal_env_fraction = final_mix["nominal_height_fraction"]
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.rel_standing_envs = final_mix["rel_standing_envs"]
  for axis, rng in VELOCITY_FINAL.items():
    setattr(twist_cmd.ranges, axis, rng)
  for name in ("command_vel", "command_mix", "push_magnitude", "arm_curriculum_state"):
    cfg.curriculum.pop(name, None)


def unitree_g1_23dof_duet_rough_env_cfg(
  play: bool = False,
  deploy_gains: bool = False,
  arm_mode: str = "uniform",
  height_mode: str = "command",
  payload: bool = True,
  command_curriculum: bool = True,
  joint_vel_noise: float | None = None,
  idle_precision: bool = True,
  height_max: float = HEIGHT_RANGE[1],
  actor_height_scan: bool = True,
  actor_history: int = 1,
  litter_terrain: bool = False,
  clearance_reference: str = "world",
  contact_softness: bool = False,
  friction_range: tuple[float, float] | None = None,
  sim2real_dr: bool = False,
  terrain_curriculum: str = "vel",
  privileged_contact_obs: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the G1-23DOF DUET rough-terrain configuration.

  Args:
    play: Play/visualisation mode (no corruption, no pushes, no curriculum).
    deploy_gains: Train against the unitree_rl_gym deployment gain table
      (hips 100, knee 150, ankles 40) with a uniform 0.25 action scale instead
      of the soft first-principles gains. Checkpoints do NOT transfer between
      the two; the exported deploy.yaml must carry the matching table.
    arm_mode: HOMIE contribution (a) ablation.
      ``uniform``  -- goals sampled uniformly and independently per joint over
                      the safe workspace (default; VLA-agnostic).
      ``anchored`` -- as uniform, but 20% of goals are deployment-pose anchors
                      plus independent per-joint noise.
      ``off``      -- arms pinned at the default pose; no upper-body disturbance.
    height_mode: HOMIE contribution (b) ablation.
      ``command`` -- pelvis height sampled in [0.12, 0.73] (default).
      ``fixed``   -- height pinned at 0.73; the obs slot stays (constant), so
                     the 71-D interface and deployability are unchanged.
    payload: When False, drop hand and torso payload randomisation entirely.
    command_curriculum: When False, sample the final (hardest) command
      distribution from step 0 instead of staging into it.
    joint_vel_noise: Override the joint-velocity observation noise half-width.

  Rough-blind-tall pass (documents/duet/rough_blind_task.md). Every default
  below reproduces the configs that existed before the pass:
    height_max: top of the height command range = the nominal height.
    actor_height_scan: False removes ``height_scan`` from the ACTOR only (the
      critic keeps it, privileged); the actor then has exactly the -Flat terms.
    actor_history: frames of actor observation history (critic stays at 1).
    litter_terrain: the litter terrain generator instead of mjlab's rough mix.
    clearance_reference: ``feet_clearance`` height reference, world|stance_foot.
    contact_softness: per-foot ``geom_solref`` timeconst randomisation, re-drawn
      at reset and mid-episode, with its upper bound ramped in over training.
    friction_range: foot friction range (None keeps (0.3, 1.6)).
    sim2real_dr: PD gains +-10% and 0-20 ms actuator latency on the lower body.
    terrain_curriculum: ``vel`` (mjlab distance rule) or ``task`` (task success).
    privileged_contact_obs: critic-only foot friction and foot softness terms.
  """
  if terrain_curriculum not in ("vel", "task"):
    raise ValueError(f"terrain_curriculum must be vel|task, got {terrain_curriculum!r}")
  if sim2real_dr and deploy_gains:
    raise ValueError("sim2real_dr is defined for the first-principles gains only")
  cfg = make_duet_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {
    "robot": get_g1_23dof_deploy_robot_cfg()
    if deploy_gains
    else _delayed_lower_body_robot_cfg()
    if sim2real_dr
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

  if litter_terrain:
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_generator = litter_terrain_generator_cfg()
    cfg.sim.nconmax = LITTER_NCONMAX
    cfg.events["reset_base"].params["pose_range"]["z"] = (LITTER_SPAWN_Z, LITTER_SPAWN_Z)

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

  # HOMIE contribution (a): the upper-body disturbance generator. Consumes ZERO
  # policy action dimensions -- it only writes joint targets for the arms.
  # Per-env asynchronous trapezoidal trajectories between independently
  # sampled left/right goals; NO slow-down while walking (the deployed VLA does
  # not know the legs are walking, so the policy must not train as if it did).
  # Stage timing is installed by ``apply_duet_curriculum`` below.
  if arm_mode not in ("anchored", "uniform", "off"):
    raise ValueError(f"arm_mode must be anchored|uniform|off, got {arm_mode!r}")
  cfg.actions["upper_body_pose"] = mdp.UpperBodyPoseActionCfg(
    entity_name="robot",
    joint_names=ARM_JOINTS,
    enabled=arm_mode != "off",
    workspace_limits=ARM_WORKSPACE_LIMITS,
    sample_range_scale=ARM_WORKSPACE_SCALE,
    traj_vmax_range=ARM_TRAJ_VMAX_RANGE,
    traj_vmax_range_init=ARM_TRAJ_VMAX_RANGE_INIT,
    traj_amax_range=ARM_TRAJ_AMAX_RANGE,
    traj_amax_range_init=ARM_TRAJ_AMAX_RANGE_INIT,
    hold_prob=0.5,
    hold_time_range=ARM_TARGET_HOLD_RANGE,
    stationary_arm_prob=0.2,
    max_segment_time=8.0,
    clean_env_ratio=ARM_CLEAN_ENV_RATIO,
    task_poses=TASK_ARM_POSES,
    task_pose_prob=0.2 if arm_mode == "anchored" else 0.0,
    task_pose_noise=TASK_ARM_POSE_NOISE,
  )

  cfg.viewer.body_name = "torso_link"

  ##
  # Commands.
  ##

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  # HOMIE contribution (b). Range matches deploy.yaml exactly; the squat floor
  # follows SQUAT_FLOOR_STAGES (installed by apply_duet_curriculum) so the
  # policy masters shallow squats before deep ones.
  if height_mode not in ("command", "fixed"):
    raise ValueError(f"height_mode must be command|fixed, got {height_mode!r}")
  base_height_cmd = cfg.commands["base_height"]
  assert isinstance(base_height_cmd, mdp.BaseHeightCommandCfg)
  base_height_cmd.enabled = height_mode == "command"
  # height_range[1] is the nominal height (BaseHeightCommand pins the
  # nominal-height slice and the GUI slider's top to it). The squat floor, walk
  # band and command-mix schedules do not depend on it, so with a raised
  # height_max walking spans [walk_min_height, height_max].
  base_height_cmd.height_range = (HEIGHT_RANGE[0], height_max)

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
  # Sagittal tolerances loosened 2x on 2026-09-16 (0.8/0.8/0.4 -> 1.6/1.6/0.8).
  # Why: the DEFAULT joint pose is 0.786 m tall, which is ABOVE the top of the
  # height command range (0.73). At the top of the range the pose regulariser
  # and track_base_height therefore want different things, and at the old
  # tolerances the regulariser won: measured on model_11500, a 0.73 command
  # settled at 0.786 m (pose 0.969, height 1.477) instead of folding to 0.71
  # (pose 0.761, height 2.000, minus ~0.38 of extra idle-anchor and torso
  # cost that holding a squat incurs). The tall pose won by 0.053/s, and that
  # thin margin also made commands near 0.68 bistable -- a 13 cm jump between
  # a 0.67 and a 0.69 command, with everything in 0.66..0.72 unreachable.
  # At 1.6/1.6/0.8 the same arithmetic favours tracking by 0.117/s.
  # Only STANDING is touched: std_walking and std_running are unchanged, so
  # gait regularisation is unaffected, and the lateral/yaw joints stay tight
  # at 0.05 because they are what balance depends on.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*hip_pitch.*": 1.6,
    r".*knee.*": 1.6,
    r".*ankle_pitch.*": 0.8,
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
    "com_support",
  ):
    cfg.rewards[name].params["asset_cfg"].site_names = site_names
  for name in ("torso_pitch_abs", "torso_roll_abs"):
    cfg.metrics[name].params["asset_cfg"].body_names = ("torso_link",)
  for name in ("com_support_error", "foot_slip_speed", "height_track_error"):
    cfg.metrics[name].params["asset_cfg"].site_names = site_names

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
    apply_duet_curriculum(cfg, CURRICULUM_END_ITERS)
  else:
    # Ablation: no staging. Sample the final distribution from step 0.
    pin_duet_full_distribution(cfg)

  ##
  # Rough-blind-tall pass. Each block is inert at its keyword's default.
  ##

  # Blind actor: the actor loses height_scan, the critic keeps it. The actor
  # must then be the deployed -Flat interface, term for term, in order.
  if not actor_height_scan:
    del cfg.observations["actor"].terms["height_scan"]
    actor_terms = tuple(cfg.observations["actor"].terms)
    assert actor_terms == DEPLOY_ACTOR_TERMS, actor_terms
  # Actor history: each term's frames oldest -> newest, concatenated per term
  # (mjlab CircularBuffer; the C++ ObservationManager with use_gym_history
  # false builds the same layout). The critic stays at one frame.
  cfg.observations["actor"].history_length = actor_history

  if privileged_contact_obs:
    critic = cfg.observations["critic"].terms
    critic["foot_friction_coef"] = ObservationTermCfg(
      func=mdp.foot_friction_coef,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES)},
    )
    critic["foot_softness"] = ObservationTermCfg(
      func=mdp.foot_softness,
      params={
        "left_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES[:7]),
        "right_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOM_NAMES[7:]),
      },
    )

  if friction_range is not None:
    cfg.events["foot_friction"].params["ranges"] = friction_range

  if contact_softness:
    # Per foot, one timeconst shared by its seven capsules (the foot geoms have
    # priority 1, so theirs is the solref MuJoCo uses). Drawn at reset and
    # re-drawn every 1-3 s: the ground under each foot changes within an
    # episode, as it does across patches of litter. The cfg holds the full
    # range; the curriculum below ramps the upper bound in.
    names = []
    for side, geoms in (("left", FOOT_GEOM_NAMES[:7]), ("right", FOOT_GEOM_NAMES[7:])):
      for mode in ("reset", "interval"):
        name = f"foot_softness_{side}" + ("_interval" if mode == "interval" else "")
        cfg.events[name] = EventTermCfg(
          mode=mode,
          interval_range_s=FOOT_TC_INTERVAL_S if mode == "interval" else None,
          func=mdp.geom_solref,
          params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geoms),
            "operation": "abs",
            "ranges": (FOOT_TC_MIN, FOOT_TC_MAX),
            "shared_random": True,
          },
        )
        names.append(name)
    cfg.curriculum["foot_softness"] = CurriculumTermCfg(
      func=mdp.event_range_schedule,
      params={
        "event_names": names,
        "lower": FOOT_TC_MIN,
        "upper_stages": [(i * ITER, v) for i, v in FOOT_TC_MAX_STAGES],
      },
    )

  if sim2real_dr:
    # PD gains +-10% around nominal, lower body only. dr.pd_gains indexes
    # entity.actuators -- the actuator GROUPS in articulation order -- with
    # asset_cfg.actuator_ids, while actuator_names would resolve to per-joint
    # ctrl indices (0..22); so the ids here are group indices, computed from the
    # robot cfg. Verified on a 4-env build: exactly the 13 lower-body
    # actuators' gainprm/biasprm change. Export metadata reads the nominal gains
    # from mj_model, which this does not touch.
    robot_cfg = cfg.scene.entities["robot"]
    assert robot_cfg.articulation is not None
    lower_groups = [
      i
      for i, act in enumerate(robot_cfg.articulation.actuators)
      if not any(tok in expr for expr in act.target_names_expr for tok in _ARM_TOKENS)
    ]
    cfg.events["pd_gains"] = EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=lower_groups),
        "kp_range": RB_PD_GAIN_RANGE,
        "kd_range": RB_PD_GAIN_RANGE,
        "operation": "scale",
      },
    )

  if clearance_reference != "world":
    cfg.rewards["foot_clearance"].params["reference"] = clearance_reference
    cfg.rewards["foot_clearance"].params["z_rest"] = FOOT_SITE_Z_REST

  if terrain_curriculum == "task":
    cfg.metrics["moving_command_fraction"] = MetricsTermCfg(
      func=mdp.moving_command_fraction, params={"command_name": "twist"}
    )
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=mdp.terrain_levels_task,
      params={
        **RB_TERRAIN_PROMOTE,
        "demote_lin_vel_error": RB_TERRAIN_DEMOTE_LIN_VEL_ERROR,
      },
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    # Play: arms static at the default pose (a fresh process starts at step 0;
    # the probes and the squat/idle scripts rely on this). Scenario tests that
    # need arm motion drive the term explicitly, see
    # ``scripts/duet_arm_scenarios.py``.
    arm_cfg = cfg.actions["upper_body_pose"]
    assert isinstance(arm_cfg, mdp.UpperBodyPoseActionCfg)
    arm_cfg.init_ratio = 0.0
    arm_cfg.ratio_stages = ((0, 0.0),)
    arm_cfg.clean_fraction_stages = ((0, 0.0),)
    twist_cmd.ranges.lin_vel_x = VELOCITY_FINAL["lin_vel_x"]
    twist_cmd.ranges.lin_vel_y = VELOCITY_FINAL["lin_vel_y"]
    twist_cmd.ranges.ang_vel_z = VELOCITY_FINAL["ang_vel_z"]
    base_height_cmd.floor_curriculum_start = HEIGHT_RANGE[0]  # full depth
    base_height_cmd.walk_min_height = WALK_MIN_HEIGHT_FINAL
    base_height_cmd.nominal_env_fraction = 0.0
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


# Everything that defines the rough-blind-tall tasks except the actor history.
RB_TALL_KWARGS = dict(
  height_max=RB_TALL_HEIGHT_MAX,
  actor_height_scan=False,
  litter_terrain=True,
  clearance_reference="stance_foot",
  contact_softness=True,
  friction_range=RB_FOOT_FRICTION,
  sim2real_dr=True,
  terrain_curriculum="task",
  privileged_contact_obs=True,
)


def unitree_g1_23dof_duet_rough_blind_tall_env_cfg(
  play: bool = False, actor_history: int = 1, evaluation: str | None = None
) -> ManagerBasedRlEnvCfg:
  """Blind DUET policy for loose litter at the natural standing height.

  Litter-like terrain, per-foot contact softness and a low-friction tail,
  PD-gain and latency randomisation, a task-success terrain curriculum,
  terrain-relative swing height, nominal height 0.79 m; the actor sees exactly
  the -Flat 71-D interface per frame (``actor_history`` frames of it) and the
  critic additionally sees the height scan, foot friction and foot softness.

  Args:
    play: Play settings (as the rough play cfg, with ``randomize_terrain``).
    actor_history: Actor frames: 5 for the -H5 task, 1 for the drop-in task.
    evaluation: ``None`` for training/play, or a fixed evaluation condition.
      Both keep the terrain_scan sensor and the critic layout, so any
      checkpoint of the task loads, and both use the play settings.
      ``flat``   -- plane terrain, contact timeconst fixed at 0.02 (rigid):
                    comparable to v8's scenario results.
      ``litter`` -- litter generator, no curriculum, difficulty 0.8-1.0,
                    timeconst fixed at tc_max, foot friction fixed at 0.3.
  """
  if evaluation not in (None, "flat", "litter"):
    raise ValueError(f"evaluation must be None|flat|litter, got {evaluation!r}")
  cfg = unitree_g1_23dof_duet_rough_env_cfg(
    play=play or evaluation is not None, actor_history=actor_history, **RB_TALL_KWARGS
  )
  if evaluation is None:
    return cfg

  softness = [n for n in cfg.events if n.startswith("foot_softness_")]
  assert len(softness) == 4, softness
  assert cfg.scene.terrain is not None
  if evaluation == "flat":
    tc = FOOT_TC_MIN
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.events.pop("randomize_terrain", None)
  else:
    tc = FOOT_TC_MAX
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None and not gen.curriculum
    gen.difficulty_range = (0.8, 1.0)
    cfg.events["foot_friction"].params["ranges"] = (0.3, 0.3)
  for name in softness:
    cfg.events[name].params["ranges"] = (tc, tc)
  return cfg
