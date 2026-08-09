"""G1-23dof with deployment-grade PD gains.

Sim2real thesis (HomieDeploy / mjlab-homierl): train with the PD gains the
onboard low-level controller actually runs, so the simulated closed loop is
the deployed closed loop.

The first-principles gains in ``g1_23dof_constants`` (armature x natural
frequency, 10 Hz) come out far softer than any proven G1 deployment stack:
hip_pitch/hip_yaw kp 40.2 vs 100-150 everywhere else, ankle 28.5 vs 40, arms
14.3 vs 40+. Deploying that soft loop (deploy.yaml faithfully ships the
training gains) amplifies unmodeled friction/backlash and slows disturbance
rejection on hardware.

Gain table used here = the unitree_rl_gym G1 table this repo's own FixStand
deploy state already runs (deploy config.yaml):

  legs:      kp 100, 100, 100 (hips), 150 (knee), 40, 40 (ankles)
             kd   2,   2,   2,          4,         2,  2
  waist_yaw: kp 200 / kd 5
  arms:      kp  40 / kd 10 (matches the FixStand/IL arm convention)

Motor properties (armature, effort limits) remain physical and come from
``g1_23dof_constants``. The matching action scale is the unitree_rl_gym
uniform 0.25 (see G1_23DOF_DEPLOY_ACTION_SCALE).
"""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from src.assets.robots.unitree_g1.g1_23dof_constants import (
  ACTUATOR_5020,
  ACTUATOR_7520_14,
  ACTUATOR_7520_22,
  ARMATURE_5020,
  ARMATURE_7520_14,
  ARMATURE_7520_22,
  FULL_COLLISION,
  HOME_KEYFRAME,
  get_spec,
)

# (kp [N·m/rad], kd [N·m·s/rad]) per joint pattern — deploy config.yaml table.
G1_23DOF_DEPLOY_PD_GAINS: dict[str, tuple[float, float]] = {
  ".*_hip_pitch_joint": (100.0, 2.0),
  ".*_hip_roll_joint": (100.0, 2.0),
  ".*_hip_yaw_joint": (100.0, 2.0),
  ".*_knee_joint": (150.0, 4.0),
  ".*_ankle_pitch_joint": (40.0, 2.0),
  ".*_ankle_roll_joint": (40.0, 2.0),
  "waist_yaw_joint": (200.0, 5.0),
  ".*_shoulder_pitch_joint": (40.0, 10.0),
  ".*_shoulder_roll_joint": (40.0, 10.0),
  ".*_shoulder_yaw_joint": (40.0, 10.0),
  ".*_elbow_joint": (40.0, 10.0),
  ".*_wrist_roll_joint": (40.0, 10.0),
}

# unitree_rl_gym deployment convention: uniform action scale.
G1_23DOF_DEPLOY_ACTION_SCALE = 0.25


def _actuator(
  patterns: tuple[str, ...],
  armature: float,
  effort_limit: float,
) -> BuiltinPositionActuatorCfg:
  gains = {G1_23DOF_DEPLOY_PD_GAINS[p] for p in patterns}
  if len(gains) != 1:
    raise ValueError(f"Actuator group {patterns} mixes different PD gains: {gains}.")
  kp, kd = gains.pop()
  return BuiltinPositionActuatorCfg(
    target_names_expr=patterns,
    stiffness=kp,
    damping=kd,
    effort_limit=effort_limit,
    armature=armature,
  )


G1_23DOF_DEPLOY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _actuator(
      (".*_hip_pitch_joint", ".*_hip_yaw_joint"),
      armature=ARMATURE_7520_14,
      effort_limit=ACTUATOR_7520_14.effort_limit,
    ),
    _actuator(
      (".*_hip_roll_joint",),
      armature=ARMATURE_7520_22,
      effort_limit=ACTUATOR_7520_22.effort_limit,
    ),
    _actuator(
      (".*_knee_joint",),
      armature=ARMATURE_7520_22,
      effort_limit=ACTUATOR_7520_22.effort_limit,
    ),
    _actuator(
      (".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
      armature=ARMATURE_5020 * 2,
      effort_limit=ACTUATOR_5020.effort_limit * 2,
    ),
    _actuator(
      ("waist_yaw_joint",),
      armature=ARMATURE_7520_14,
      effort_limit=ACTUATOR_7520_14.effort_limit,
    ),
    _actuator(
      (
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
      ),
      armature=ARMATURE_5020,
      effort_limit=ACTUATOR_5020.effort_limit,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_23dof_deploy_robot_cfg() -> EntityCfg:
  """G1-23dof with deployment-grade PD gains (fresh instance per call)."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=G1_23DOF_DEPLOY_ARTICULATION,
  )
