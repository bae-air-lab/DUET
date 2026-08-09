"""DUET: decoupled loco-manipulation lower-body policy for the Unitree G1.

An RL policy owns the 13 lower-body joints (12 legs + waist_yaw); the 10 arm
joints are never actions. They appear only as observations, driven at training
time by a disturbance generator and at deployment by a VLA publishing to
``rt/arm_targets``.

Design rationale, including the justification for every reward weight, is in
``documents/duet/reward_design.md``. Operational guide is in this package's
``README.md``.
"""
