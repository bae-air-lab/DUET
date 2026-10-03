# Arm-robustness training pass

Goal: a lower-body policy that stands, walks (forward, backward, lateral), turns,
stops, changes height and recovers while the ten arm joints follow **arbitrary
bounded trajectories from any source** (VLA, teleop, script), without answering an
arms-forward CoM shift with a visible backward torso lean.

Everything below is configuration in `src/tasks/duet/config/g1_23dof/env_cfgs.py`
and MDP terms in `src/tasks/common/mdp/`. Nothing in `deploy/`, `vla_bridge/` or
the DeployGains task changed; the actor observation is still the 71-D contract.

## 1. Arm disturbance generator (`mdp.UpperBodyPoseAction`)

Per env and per arm joint the term keeps `goal`, `traj_pos`, `traj_vel`, `vmax`,
`amax` (all `[num_envs, 10]` GPU tensors) plus per-env timers (`holding`,
`hold_timer`, `segment_time`, mixture `slot`). Each policy step:

```
h      = a_max*dt/2
v_stop = sqrt(h^2 + 2 a_max |goal-pos|) - h          # discrete-time stopping speed
v_des  = sign(goal-pos) * min(v_max, v_stop, |goal-pos|/dt)
vel   += clamp(v_des - vel, -a_max dt, +a_max dt)
pos   += vel dt ; a step that crosses the goal snaps to it (no overshoot)
```

Long moves are trapezoidal, short ones triangular. Goal changes are per-env:
reach goal -> with p=0.5 hold 0.2-2.5 s -> draw a new goal, new `vmax`, new
`amax`. A segment older than 8 s is redrawn. Each arm independently keeps its
current pose with p=0.2 (one arm moving, one still). There is **no walking
slow-down**: the arm distribution is independent of the locomotion command.

Goals are sampled **independently per joint** inside `ARM_WORKSPACE_LIMITS`, a
conservative box inside the soft limits (shoulder pitch -2.0..1.0, roll never
through the torso, yaw +-1.0, elbow -0.5..1.6, wrist +-1.5). The deployment-pose
anchors are kept only as an optional minority (`arm_mode="anchored"`, 20%); the
default is `uniform`.

## 2. Curriculum (iterations; every knot is linearly interpolated)

| quantity | 0 | 1500 | 3000 | 5000 | 7000+ |
|---|---|---|---|---|---|
| arm ratio (workspace, vmax/amax interpolation) | 0.15 | 0.15 | 0.50 | 0.75 | 1.0 |
| clean-arm env fraction (ratio capped at 0.10) | 0.50 | 0.40 | 0.30 | 0.20 | 0.10 |
| squat floor (m) | 0.68 | 0.68 | 0.60 | 0.38 | 0.27 (final at 6000) |
| nominal-height env fraction | 0.50 | 0.40 | 0.30 | 0.20 | 0.10 |
| walk-band floor (m) | 0.71 | 0.71 | 0.68 | 0.64 | 0.60 |
| standing env fraction | 0.20 | 0.25 | 0.30 | 0.30 | 0.30 |
| push scale (x final range) | 0.10 | 0.30 (at 1000) | 0.65 | 1.0 | 1.0 |
| vx / vy / yaw range (step-wise) | -0.3..0.6 / +-0.2 / +-0.4 | -0.5..1.0 / +-0.4 / +-0.8 | (3500) -0.65..1.1 / +-0.45 / +-0.9 | -0.8..1.2 / +-0.5 / +-1.0 | same |

Arm dynamics at ratio 0: vmax 0.2-0.8 rad/s, amax 0.5-3 rad/s^2; at ratio 1:
vmax 0.3-3.5 rad/s, amax 1-20 rad/s^2. Friction, encoder bias, CoM offset, hand
and torso payload randomisation are unchanged and active from iteration 0.

All schedules read `env.common_step_counter` (checkpointed), so `--resume`
continues where it stopped. `apply_duet_curriculum(cfg, end_iters)` rescales every
knot for a shorter horizon (the ablations use 3,500).

## 3. Rewards touched (and nothing else)

* `idle_position_anchor`: `xy_deadband=0.025 m`, `yaw_deadband=0.02 rad`. Inside
  the band pelvis travel over stationary feet is free; beyond it the constant-
  gradient penalty is unchanged. This was the conflict that produced the lean:
  with a zero-width anchor the pelvis was pinned and the quadratic orientation
  term made a 5 deg lean nearly free.
* `body_orientation_l2`: -1.5 -> -2.5 (height gating unchanged).
* new `com_support` (-1.0): whole-body CoM (`subtree_com`, includes arms, hands
  and payload) vs. a conservative ellipse between the feet (foot half-length
  0.09 m + half stagger, half-width 0.04 m + half separation). Zero inside 50% of
  the region, quadratic to the edge, +2(r-1) outside; double support only; x0.25
  while a locomotion command is active.
* Critic-only observation `arm_traj_vel` (10-D commanded arm velocity). The actor
  is unchanged; feeding dq_ref to the actor would need the C++ controller to
  publish it and is left as a future step.

## 4. Diagnostics (`Episode_Metrics/*`)

`torso_pitch_abs`, `torso_roll_abs` (rad), `idle_root_drift` (m),
`com_support_error` (0 = centre, 1 = edge), `foot_slip_speed` (m/s),
`arm_traj_speed` (rad/s), `arm_traj_accel` (rad/s^2), `lin_vel_track_error`,
`yaw_track_error`, `height_track_error`; plus `Metrics/idle_drift_max_m`,
`Curriculum/push_magnitude`, `Curriculum/arm_curriculum_state`. Falls are
`Episode_Termination/fell_over`.

## 5. Digital-twin acceptance tests

```bash
PYTHONPATH=. python scripts/duet_arm_scenarios.py --checkpoint logs/rsl_rl/DUET_G1_23dof/<run>/model_XXXX.pt
```

runs the eight scenarios (arms stationary through the locomotion set; both arms
forward; asymmetric; moving arms while standing / walking forward / walking
backward then stop / turning / changing height) and prints torso pitch and roll,
idle drift, CoM support ratio, foot slip, unnecessary stepping, tracking errors
and falls per scenario. `--scenario <name>` runs one.

Note on `PYTHONPATH`: the editable `src` package installed in this Python
environment points at `~/unitree_rl_mjlab`, so `python scripts/train.py` run from
this repo without `PYTHONPATH=.` imports the *other* repository's tasks.

## 6. First run: what went wrong and what was changed (2026-09-15)

Run `2026-09-15_20-32-31_arm_robust`, read from its tfevents:

| iter | return | action std | value loss | action_rate | noise floor | anchor | idle drift m | push | arm ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 101.2 | 0.217 | 0.017 | -0.175 | -0.123 | -0.080 | 0.065 | 0.30 | 0.15 |
| 1500 | 101.4 | 0.219 | 0.021 | -0.185 | -0.125 | -0.047 | 0.076 | 0.39 | 0.15 |
| 2000 | 81.3 | 0.336 | 0.049 | -0.387 | -0.293 | -0.220 | 0.177 | 0.48 | 0.27 |
| 3000 | 63.4 | 0.434 | 0.102 | -0.614 | -0.490 | -0.307 | 0.243 | 0.65 | 0.50 |
| 4000 | 36.7 | 0.559 | 0.259 | -0.953 | -0.811 | -0.652 | 0.372 | 0.83 | 0.63 |
| 5000 | 7.6 | 0.701 | 0.517 | -1.465 | -1.276 | -0.785 | 0.520 | 1.00 | 0.75 |

Noise floor = 13 x 2 std^2 x 0.1, the action-rate cost of exploration alone.
Stage A converged cleanly (return 101, std 0.22). The std rise began at
iteration 1588, the stage A -> B switch, and was monotonic thereafter; the
learning rate stayed at 2e-4..6e-4 throughout (not the cause). Falls stayed
near zero; height tracking stayed good. Diagnosis: an action-std runaway. The
entropy push on a scalar std is constant while the policy-gradient counter-force
weakens as unobservable randomness (push timing, arm goal draws) dominates the
advantage; the old schedule held the coefficient at its maximum through exactly
that window. Two amplifiers: the anchor term charged for push-induced
displacement the policy cannot undo, and the velocity envelope doubled in one
iteration at 1500.

Changes:

1. `rl_cfg.py`: entropy 0.01 held to 1500, decayed to 0.004 by 2500 (was held
   to 7000, decayed to 0.006 by 9000).
2. `events.py` / `duet_env_cfg.py`: `push_and_relatch_anchor` re-latches the
   idle anchor for pushed envs.
3. `curriculums.commands_vel`: `ramp_steps` blends each velocity stage in over
   500 iterations (`VELOCITY_RAMP_ITERS`).

Resumed from `model_2000.pt` (std 0.34, return 81, before the anchor and noise
costs dominated). Curricula continue from iteration 2000; the entropy
coefficient there is 0.007 and reaches 0.004 at 2500. What to watch after the
resume: std should turn down within ~300 iterations and be below 0.35 by 3000;
return should stay above 60 through the 3000-5000 ramps; yaw tracking error
(0.26 rad/s in stage A, 1.1 at 5000) should stop growing.

## 7. Squat floor frozen at 0.27 m (2026-09-16)

Run `2026-09-16_08-25-02_arm_robust_v3` tracked the descending floor with height
error at or below 0.025 m from 0.60 m down to 0.27 m, then showed the first
upward move in height error (0.0198 -> 0.0245 over iterations 5785-6101) as the
ramp continued below 0.27. On the operator's call the floor was frozen there.

`HEIGHT_RANGE` moved 0.18 -> 0.27 in the same edit, because it is the command
range the ONNX metadata reports as `height_command_range`. Leaving it at 0.18
would have exported a policy claiming a trained depth it never saw.

**Deployment consequence.** `deploy.yaml` still declares
`base_height.range: [0.18, 0.73]` for the previously deployed checkpoint, so
`scripts/check_deploy_consistency.py` will fail its "base_height range within
trained range" check for any policy exported from this run until that field is
narrowed to `[0.27, 0.73]`. Narrow it only together with deploying a policy from
this run; the old checkpoint genuinely was trained to 0.18 m.
