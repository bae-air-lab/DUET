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

## 8. Rough-blind-tall pass (2026-09)

Brief: `documents/duet/rough_blind_task.md`. Two new training tasks, identical
except for the actor history, plus two fixed evaluation variants of each:

| task id | actor input | role |
|---|---|---|
| `Unitree-G1-23Dof-Duet-RoughBlind-Tall-H5` | 5 x 71 = 355 | primary litter candidate |
| `Unitree-G1-23Dof-Duet-RoughBlind-Tall` | 71, `-Flat` layout | drop-in fallback, and the history ablation |
| `...-EvalFlat` (for each) | same | plane, contacts rigid (timeconst 0.02), play settings |
| `...-EvalLitter` (for each) | same | litter, difficulty 0.8-1.0, timeconst tc_max, friction 0.3 |

Everything is behind keyword arguments of `unitree_g1_23dof_duet_rough_env_cfg`
whose defaults rebuild every pre-existing task bit-identically (verified by
dumping all 25 existing tasks' train/play env cfg, rl cfg and runner before and
after: empty diff). New scripts: `duet_stand_height.py`,
`duet_contact_softness_sweep.py`, `duet_terrain_stats.py`.

### 8.1 Nominal height 0.79 m (`height_max`)

`scripts/duet_stand_height.py` (Mode 1): the default pose under PD with zero
action on the rigid plane settles at **h = 0.7949 m** (`root_z - min(foot_site_z)
+ 0.02`, identical at 10 s and 20 s), so `height_max = 0.79` and the command
range is `(0.24, 0.79)`. Forward kinematics of the same pose without PD sag gives
0.806 m, so the sag is ~11 mm.

Measurement detail: with zero action the default pose is *not* statically
stable on the first-principles gains (ankle kp 28.5 Nm/rad per side against a
toppling stiffness of roughly m g l ~ 200 Nm/rad); it pitches forward and trips
`fell_over` ~1.2 s after every reset. The script therefore holds the pelvis
level and fixed in x/y after every physics substep while leaving its height
free -- the same "pelvis level, feet flat" assumption as Mode 2 -- which applies
no vertical force at rest. It also measures **z_rest = -0.0021 m**: the foot
site sits 2.1 mm *below* the sole (capsule bottoms at z = -0.035 in the
ankle-roll frame, site at -0.037).

Deploy convention (read-only check of
`deploy/include/isaaclab/envs/mdp/observations/observations.h`): the
controller's `[base_height] achieved` printout is the vertical drop from pelvis
to the lower foot site, **without** the 0.02 sole offset, i.e. `h - 0.02`. A
perfectly tracked command prints `err = -0.020`. Add 0.02 to `achieved` before
comparing it with the command (hardware test 3).

### 8.2 Blind actor, privileged critic

`actor_height_scan=False` deletes `height_scan` from the actor only; the builder
asserts the actor terms equal the deployed `-Flat` terms in order. With
`privileged_contact_obs=True` the critic also sees `foot_friction_coef` (1-D,
live `geom_friction[...,0]` of the foot geoms) and `foot_softness` (2-D, live
left/right `geom_solref` timeconst). Critic: 286-D. Mirror rules: friction
`([0],[1])`, softness `([1,0],[1,1])`. Reason: under heavy randomisation the
value target is noisy; a critic that knows the ground removes much of the
unexplained advantage variance behind the section-6 entropy runaway.

### 8.3 Actor history (`actor_history`)

`history_length = 5` on the actor group only. Layout (mjlab `CircularBuffer`
and the C++ `ObservationManager` with `use_gym_history: false` agree): per term,
its 5 frames oldest to newest, then the next term; both backfill with the first
frame after a reset. `symmetry.py` tiles each per-frame mirror rule over the
frames (H read from the group config). Export metadata now records
`history_length`, per-frame `observation_dims`, and `obs_dim` = full input width;
`history_length` is deliberately not in the config hash (it is implied by
`obs_dim` and the per-frame dims, and leaving it out keeps every 1-frame
export's hash unchanged). `check_deploy_consistency.py` reads H from the
metadata (default 1), requires every deploy.yaml term's `history_length` to
equal it, compares per-frame widths, requires `sum(widths) * H` = ONNX input,
and fails any `use_gym_history: true`; `--expect-obs-dim` stays per frame.
Its output for v8 against the v0 deploy.yaml is byte-identical to before.
`export_duet_onnx.py` now passes the per-frame width to the checker.

### 8.4 Litter terrain (`litter_terrain`)

`litter_terrain_generator_cfg()`: 8 x 8 m tiles, 10 rows x 20 columns, border 20,
curriculum; `max_init_terrain_level = 5` unchanged. Columns: 3 flat, 6 fine
bumps, 4 undulation, 2 waves, 2 slope, 1 inverted slope, 2 clumps (20 columns
cannot split the two slope types 1.5/1.5). New
`src/tasks/common/terrains.py:HfScaledRandomUniformTerrainCfg` scales the noise
bound with difficulty (mjlab's ignores difficulty); it also guards the parent's
`int()` truncation of float bounds (0.015/0.005 = 2.999...).

Measured with `scripts/duet_terrain_stats.py` (ray casts on a 2.5 cm grid;
peak-to-peak at difficulty 0 / 0.5 / 1, and the largest slope inside the tile at
difficulty 1):

| sub-terrain | settings | p2p (m) | slope inside tile |
|---|---|---|---|
| flat | box | 0 / 0 / 0 | 0 |
| fine bumps | noise 0-0.04, step 0.005 | 0 / 0.020 / 0.040 | 0.56 |
| undulation | Perlin, height (0.005, 0.06), octaves 3, scale 5, 0.1 m cells | 0.005 / 0.032 / 0.060 | 0.11 |
| waves | amplitude 0-0.04 (+-0.04 about the mean), 4 waves | 0 / 0.040 / 0.080 | 0.14 |
| slope, slope_inv | slope 0-0.15, platform 2 m | 0 / 0.155 / 0.31 | 0.21 |
| clumps | 0.3 m cells, +-0.025, merged at 1.25 cm levels | 0 / 0.025 / 0.050 | steps up to 0.05 m |

Undulation wavelength (mean-crossing estimate) 1.92 m, with octaves at ~1.9,
0.96 and 0.48 m. It uses 10 cm cells, not mjlab's 5 cm Perlin default: at 5 cm a
fallen body overlaps 50+ cells and mujoco_warp drops contacts ("height field
collision overflow, number of collisions >= 50"); measured over 300 steps x 256
envs this sub-terrain produced every such warning (100 when alone, 0 for every
other sub-terrain), and 10 cm cells produced none with the same statistics.
The pyramid's product shape gives a local slope of 0.21 against
the nominal 0.15. mjlab's heightfields do not taper into their flat 0.25 m tile
border, so undulation and waves have a step of up to 6 / 4 cm at that edge.
Two mjlab classes cannot build at exactly difficulty 0 (Perlin: zero-height
heightfield; box grid: NaN colour ramp); the 5 mm undulation floor avoids the
first, and the generator never draws exactly 0 for the second (row 0 draws from
U[0, 0.1)).

Three measured problems and what was done:

* **Clumps merged.** As individual 0.3 m boxes the two clump columns are
  ~12,500 geoms (12,842 in the scene). That is over mujoco_warp's 250k
  candidate-pair limit for its n-squared broadphase, so it switches to the
  segmented sweep-and-prune broadphase, which at 1024 envs silently returns *no
  contacts at all* (robots fall through the floor; no warning) and at 4096 envs
  fails CUDA graph creation with out-of-memory; <= 512 envs works. With
  `merge_similar_heights=True, height_merge_threshold=0.0125` the heights snap
  to 0 / +-1.25 / +-2.5 cm, the scene is ~6,200 geoms and keeps the n-squared
  broadphase. The same limit applies to any future terrain with many boxes.
* **`nconmax` 48 -> 256** for the litter tasks. mujoco_warp's `put_data` checks
  `nconmax` against one CPU `MjData` at `qpos0` (straight legs at the world
  origin), which overlaps whatever tile is there: 117 contacts on the training
  grid, up to 143 on the random play/eval grids. At runtime, standing on the
  heightfields uses 24 contacts/world on average with step peaks of 43-47, i.e.
  90-98% of a 48 pool. 256 costs no measurable throughput. (Pre-existing: the
  `-Rough` training grid happens to give exactly 48 at `qpos0`, and the `-Rough`
  play config fails to build in about 5 of 8 attempts for the same reason.)
* **Spawn z +0.025 m** (`reset_base`, litter tasks only). On the hardest row
  with rigid feet, waves tiles put a crest up to 4 cm above the tile origin
  inside the +-0.5 m spawn window; feet spawned up to 14 mm inside it and the
  robot popped off at 0.35 m/s. After the offset: no penetration on any
  sub-terrain, all robots settle downward.

### 8.5 Terrain-relative foot clearance (`clearance_reference`)

`mdp.feet_clearance(reference="stance_foot", z_rest=-0.0021)` measures the foot
above the lower foot and targets `0.10 - z_rest`; the weight is unchanged and
every existing task keeps `"world"`. On flat rigid ground (v8's gait, 0.5 m/s,
15.5k swing samples) the two references agree to 0.49 mm on average (foot-speed
weighted 0.34 mm; mean cost 0.05694 vs 0.05666), but 4.1% of swing samples
differ by more than 2 mm (max 4.7 mm): the difference is exactly how far the
stance foot's site is above its flat-rest height, which grows when the stance
foot rolls at heel-off/toe-off. In mid-swing (swing foot > 3 cm up) p99 is
1.7 mm. Accepted: the cost the policy sees differs by 0.5%.

### 8.6 Soft and slippery ground (`contact_softness`, `friction_range`)

New event `mdp.geom_solref` (modelled on `geom_friction`; imports mjlab's
private `_randomize_model_field`, mjlab being pinned at 1.2.0). Per foot, the
seven capsules share one timeconst (`shared_random`), drawn at reset and
re-drawn every 1-3 s (a reset-mode and an interval-mode term per foot). Foot
geoms have priority 1, so their `solref` is the one MuJoCo uses.

`scripts/duet_contact_softness_sweep.py` (rigid plane, 16 envs, zero action,
pelvis held level):

| timeconst (s) | 0.02 | 0.05 | 0.08 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|---|---|
| static sinkage (mm) | 0 | 0.36 | 0.91 | 1.38 | 2.87 | 4.64 | 6.56 | 8.61 |
| deepest in a 5 cm drop (mm) | 5.7 | 16.4 | 26.1 | 32.9 | 41.4 | 43.9 | 45.0 | 45.1 |

Contacts <= 32/world, constraint rows <= 122, no oscillation or NaN, throughput
flat across timeconsts. The sinkage is about 5x below the single-contact
formula in the brief because the 28 foot contact points share the load and each
is regularised on its own. **tc_max = 0.30**, the hard cap: no operator
sinkage measurement exists yet, and the 25 mm fallback target is out of reach
under the cap (0.30 s gives 8.6 mm static). If the litter bed sinks more than
~1 cm under the stock firmware, soft contacts cannot represent it and the
granular-contact follow-up in the brief's section 12 applies. The upper bound
ramps in via `mdp.event_range_schedule`: (0, 0.02), (1000, 0.104), (3000, 0.30),
logged as `Curriculum/foot_softness`. Foot friction for the new tasks:
(0.2, 1.6), startup.

Verified: a mid-episode interval write reaches the physics with CUDA graphs on
and no graph re-creation (the left foot sank while the right did not), and the
critic's `foot_softness` reads the live value.

### 8.7 Sim-to-real randomisation (`sim2real_dr`)

* PD gains x U(0.9, 1.1) (kp and kd separately), startup, lower body only.
  `dr.pd_gains` indexes `entity.actuators` (the four actuator *groups*) with
  `asset_cfg.actuator_ids`, whereas `actuator_names=...` resolves to per-joint
  ctrl indices, so the brief's `actuator_names=LOWER_BODY_ACTUATORS` form raises
  `IndexError`. The config passes the group indices (computed from the robot
  cfg: [1, 2, 3]). Verified on 4 envs: exactly the 13 lower-body actuators
  change (ratios 0.90-1.10), the 10 arm actuators do not, and `mj_model`
  (the export source) keeps the nominal gains.
* Actuator latency 0-4 physics steps (0-20 ms): `DelayedActuatorCfg` around the
  7520-14, 7520-22 and ankle groups, built in the env cfg (the constants file's
  action-scale loop is untouched). `delay_hold_prob=0`, `delay_per_env_phase=False`,
  `delay_update_period=1e9`: the lag is drawn once at the first physics step
  after each reset and held. Verified: no change over 800 physics steps without
  a reset, re-drawn on reset, uniform over 0-4. The three groups draw
  independently.

### 8.8 Task-success terrain curriculum (`terrain_curriculum="task"`)

`mdp.terrain_levels_task` plus the metric `moving_command_fraction`. Per
finished episode: promote if it timed out without falling, moved for >= 25% of
it, and the episode means are lin vel error <= 0.20 m/s, yaw error <= 0.35
rad/s, height error <= 0.04 m, idle drift <= 0.05 m; demote if it fell, or moved
>= 25% with lin vel error >= 0.35 m/s; otherwise stay. It reads the metrics
manager's `_episode_sums / _step_count` (the curriculum runs before the metrics
reset in `_reset_idx`; asserted). Logged as
`Curriculum/terrain_levels/{terrain_level_mean,promote_frac,demote_frac}`.

### 8.9 Environment and operations

* `requirements-train.txt` could not build its own environment: `numpy==1.26.4`
  has no Python 3.13 build (and onnx 1.21 needs numpy >= 2.1 there);
  rsl-rl-lib's torchvision dependency, resolved from PyPI, replaces torch 2.7.1
  with 2.14.0 (CUDA 13); mjlab's terrains import scipy without declaring it; and
  rsl-rl-lib 5.0.1's wandb writer is rejected by wandb 0.30. The file now pins
  the environment these runs used (numpy 2.5.2, scipy 1.17.1, torchvision
  0.22.1 from the cu128 index, wandb 0.28.0).
* mujoco_warp prints `Warning: opt.ccd_iterations, currently set to 500, needs
  to be increased.` for some capsule-vs-heightfield pairs whose GJK/EPA does not
  converge: ~5 lines per env-step on this terrain (the undulation the most;
  `-Rough` does it too, ~8x less). Raising `ccd_iterations` to 10,000 does not
  change the count; contacts are still produced (robots stand normally on the
  bumps). The launch commands filter that one line out of the console stream so
  it cannot fill the disk.
* `scripts/select_duet_checkpoint.py` cannot run on the training PC: its
  evaluator `/home/rbbist.lab/paper_rl/duet_bench/duet_eval.py` is not there.
  Use `scripts/duet_rank_checkpoints.py` on `-EvalFlat` and `-EvalLitter`.
* Deploy files for the new policies: `exported/rough_blind_tall_h5/deploy.yaml`
  (every term `history_length: 5`) and `exported/rough_blind_tall_h1/deploy.yaml`
  (v0's file with only `range: [0.24, 0.79]` and `standstill: 0.79` changed).
  Copy the matching one next to each exported policy.

### 8.10 Tooling

Every script listed in the brief ran once against a 20-iteration checkpoint of
the new tasks (H5 unless noted): `play.py` (viser, `-H5` and `-EvalLitter` of
H1), `export_duet_onnx.py` (both), `duet_arm_scenarios.py` (`-H5-EvalFlat`),
`duet_probe_locomotion.py`, `duet_probe_idle.py`, `duet_rank_checkpoints.py`
(`-EvalFlat` and `-EvalLitter` of both). Two fixes:

* `export_duet_onnx.py` passed the full `obs_dim` to the checker's
  `--expect-obs-dim`, which is per frame; it now passes the per-frame width.
* `duet_arm_scenarios.py` runs each scenario in its own process when several
  are requested. On the new tasks (which carry the terrain-scan raycast sensor)
  building the sixth env + runner in one process fails CUDA graph capture of
  the sensor graph (Warp CUDA error 901); `-Flat` runs all eight in one process,
  any single scenario runs, and eight env builds with 1000-step rollouts in one
  process do not fail -- so it is a repeated-construction effect, not something
  a single training process can hit. Output is unchanged.

`select_duet_checkpoint.py` cannot run on this machine (its evaluator is
missing, see 8.9).

### 8.11 Gates, throughput and launch

Gate outputs are kept in `logs/rough_blind_tall_gates/` on the training PC.
G0-G6, G8, G9 pass; G7 passes on the mean and misses the 2 mm bound on a 4%
tail (8.5). Smoke runs (20 iterations, 256 envs): no NaN, no buffer-overflow
warning, 1.77 s (H5) / 1.73 s (H1) per iteration.

G10 at 4096 envs: H5 alone 3.71 s/iteration (20k iterations in 20.6 h, the
25,001 budget in 25.7 h), H1 alone 3.59 s/iteration (19.9 h / 24.9 h). Both at
once is impossible: one run peaks at 76.9 GB of the card's 96 GB, and the
parallel attempt fails CUDA graph creation with out-of-memory. Per the brief's
fallback the runs are sequential: H5 (A) first, H1 (B) after A has finished.

```bash
cd ~/Desktop/DUET && mkdir -p logs
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
# A: now
tmux new -d -s duet_rbt_h5 "bash -lc 'source $CONDA_SH && conda activate duet && cd ~/Desktop/DUET && \
  PYTHONPATH=. python scripts/train.py Unitree-G1-23Dof-Duet-RoughBlind-Tall-H5 \
  --agent.seed 1 --env.scene.num-envs 4096 --agent.run-name rough_blind_tall_h5 \
  2>&1 | grep --line-buffered -v ccd_iterations | tee logs/rough_blind_tall_h5.console.log'"
# B: waits until an H5 run has written model_25000.pt AND the duet_rbt_h5
# session has ended (so a stop/resume of A does not start B early)
tmux new -d -s duet_rbt_h1 "bash -lc 'source $CONDA_SH && conda activate duet && cd ~/Desktop/DUET && \
  until ls logs/rsl_rl/DUET_G1_23dof/*_rough_blind_tall_h5*/model_25000.pt >/dev/null 2>&1 && \
  ! tmux has-session -t duet_rbt_h5 2>/dev/null; do sleep 300; done; \
  PYTHONPATH=. python scripts/train.py Unitree-G1-23Dof-Duet-RoughBlind-Tall \
  --agent.seed 1 --env.scene.num-envs 4096 --agent.run-name rough_blind_tall_h1 \
  2>&1 | grep --line-buffered -v ccd_iterations | tee logs/rough_blind_tall_h1.console.log'"
```

The only difference from the brief's commands, apart from B waiting, is the
`grep -v ccd_iterations` filter (8.9). TensorBoard:
`conda activate duet && tensorboard --logdir ~/Desktop/DUET/logs/rsl_rl/DUET_G1_23dof --port 6006 --bind_all`.
Runs also sync to wandb (project `mjlab`), the configured default.

### 8.12 Terrain promote thresholds relaxed once (2026-09-30, iteration 3000)

Section-9 rule: `terrain_level_mean` below 1.0 at iteration 3000 -> relax the
promote thresholds once to lin 0.25 / yaw 0.45, resume. H5 run
`2026-09-30_00-17-59_rough_blind_tall_h5` at iteration 3010: terrain level
0.00, no env promoted since the start (promote_frac 0.000 throughout; demote
0.65 of resetting envs), with otherwise healthy training (action std 0.257,
falls 8% of episodes over the last 500 iterations, height error 0.028 m,
reward 55.5). At iteration 2000 the episode means were lin 0.41 m/s, yaw 0.90
rad/s, height 0.024 m, idle drift 0.059 m against promote limits 0.20 / 0.35 /
0.04 / 0.05.

Change: `RB_TERRAIN_PROMOTE` lin 0.20 -> 0.25, yaw 0.35 -> 0.45 (height, drift,
moving fraction and the demote rule unchanged). The run was stopped with Ctrl-C
right after `model_3000.pt` was written and resumed from it
(`--agent.resume True --agent.load-run 2026-09-30_00-17-59_rough_blind_tall_h5
--agent.load-checkpoint model_3000.pt --agent.max-iterations 22001`, ending at
25000) into a new run directory with the same run name; the console log is
appended. Terrain levels are not checkpointed and restart at random rows 0-5
on resume. Run B (H1) had not started, so it trains with the relaxed thresholds
from iteration 0: A and B now also differ in this for A's first 3000 iterations.
