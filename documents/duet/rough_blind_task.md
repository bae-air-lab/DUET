# Task: a stable, blind DUET policy for poultry litter at the robot's natural standing height

Brief for the agent that implements, verifies and launches the next training
runs on the training PC (RTX PRO 6000 Blackwell, 96 GB). Written 2026-09-29.
Read it all before editing anything. Every number marked **measure** must be
measured and reported, not assumed.

---

## 1. Goal and priorities

One lower-body policy (13 joints: 12 legs + waist yaw) that walks and stands on
loose poultry litter as reliably as the stock Unitree firmware, while the ten
arm joints are driven by someone else (VLA, teleop, script) through
`rt/arm_targets`.

In priority order:

1. **Stability.** On litter: no falls, no drift while standing, no visible
   wobble or oscillation, no stumbling at the concrete/litter transition.
2. **Height change.** Nominal stance at the robot's natural standing height,
   commandable down to a squat at deployment time.
3. **Arm independence.** The legs keep balance while the arms move
   arbitrarily. The existing arm-disturbance generator and curriculum already
   train this and are **unchanged**.

"Done" means the sim gates in section 7 pass, and then the hardware tests
in section 11 pass on a representative litter bed.

## 2. Why the deployed policy fails on litter, and what the evidence says

The deployed policy `exported/arm_robust_v8_21000` (task
`Unitree-G1-23Dof-Duet-Flat`, commit `a992697`) walks on flat concrete but
struggles on poultry litter. The stock firmware is stable on the same litter
and stands visibly taller. v8 was trained:

- only on a rigid, flat plane with no contact compliance;
- blind with **one frame** of proprioception (71-D), so the policy has no way
  to notice that the ground is soft or slipping and adapt to it;
- 5-6 cm crouched: the top of the height command is 0.73 m while the default
  joint pose is ~0.786 m, so the pose regulariser and `track_base_height`
  pull against each other at the nominal height (see the `std_standing`
  comment in `env_cfgs.py`);
- with no actuator-latency or PD-gain randomisation.

We start **from scratch**: v8 is judged unstable, and the new primary policy
has a different input width, so v8 weights cannot be loaded into it anyway.

### What others do for this problem

| approach | source | what we take from it |
|---|---|---|
| Actor sees a short **history** of proprioception (Unitree G1: 5 frames) | Unitree `unitree_rl_lab` G1 velocity task ([cfg](https://github.com/unitreerobotics/unitree_rl_lab)); HOMIE uses 6 frames ([arXiv 2502.13013](https://arxiv.org/abs/2502.13013)) | 5-frame actor history. Our C++ deployer (inherited from unitree_rl_lab) already supports per-term `history_length` in `deploy.yaml`. |
| Nominal pelvis height 0.78 m on G1 | Unitree `unitree_rl_lab` (`base_height` target 0.78) | Nominal = the measured height of our default pose (~0.78). |
| Blind humanoid on snow, sand, gravel via a history encoder and wide randomisation (friction 0.2-2.0, PD gains 80-120%, motor strength 90-110%, delay 0-10 ms) | DWL, Gu et al. 2024 ([arXiv 2408.14472](https://arxiv.org/abs/2408.14472)) | PD-gain and latency randomisation; asymmetric actor-critic with privileged critic information. |
| Compliant ground in MuJoCo by randomising the foot `solref` time constant in (0.02, 0.4), re-sampled mid-episode (~0.5 s), plus <= 4 cm heightfield bumps; tested on mattress, foam and grass | HRP-5P, [arXiv 2504.13619](https://arxiv.org/abs/2504.13619) | Per-foot softness randomisation that changes during the episode, over a range **calibrated** to measured sinkage. |
| G1 on granular media: a rigid-contact policy got 0% success on deep sand vs 85% with a physics-based granular model; the good policy raised swing clearance on soft ground (9 -> 16 cm) and was penalised for foot orientation, joint power and impact | GM-Loco, [arXiv 2609.10286](https://arxiv.org/abs/2609.10286); Choi et al., Sci. Robotics 2023 ([doi](https://www.science.org/doi/10.1126/scirobotics.ade2256)) | The limit of what soft contacts can represent. If hardware tests show deep sinkage or foot entrapment, the next step is a granular contact model (section 12), not more randomisation. |
| MuJoCo contact priority and softness | [MuJoCo modeling docs](https://mujoco.readthedocs.io/en/stable/modeling.html) | With different geom priorities, the higher-priority geom's `solref`/`solimp`/friction are used outright. Penetration at rest is `r = a_u (1-d) timeconst^2 dampratio^2`. |

### Corrections to the previous version of this brief

These were checked against this repository and the installed mjlab 1.2.0:

1. **"Taller is more stable" is not the reason to raise the height.** A taller
   stance raises the CoM. The real benefit is that the height command and the
   pose regulariser stop fighting, and the knees get their normal range back.
   The nominal height is therefore **measured** from the default pose, not
   picked.
2. **Soft MuJoCo contacts are not litter.** They are an elastic
   penetration-spring. Litter is loose, plastic (does not spring back) and
   shears sideways. Soft contacts are used here as a partial proxy only, and
   the real test is the litter bed in section 11.
3. **The old softness range did almost nothing.** Foot collision geoms have
   `priority=1`, so the foot's `solref` is used outright; there is no
   averaging with the terrain. By MuJoCo's formula, timeconst 0.02-0.05 s
   gives about 0.2-1.2 mm of static sinkage, which is effectively rigid.
4. **A survival-only terrain curriculum rewards standing still.** Promotion
   here requires tracking the commanded motion (section 6.8).
5. **mjlab's `HfRandomUniformTerrainCfg` ignores difficulty** (`del
   difficulty` in `mjlab/terrains/heightfield_terrains.py`), so "bumps" would
   be equally rough on every curriculum row. A difficulty-scaled subclass is
   needed (section 6.4).
6. **`feet_clearance` measures the foot's world z** against a fixed 0.10 m.
   On any terrain tile whose surface is not at z=0 the swing-height target is
   wrong. The existing `-Rough` task has this latent bug (section 6.5).
7. **Deploy auto-selects the newest policy folder.** `policy_dir:
   config/policy/velocity` loads the last-sorted subfolder that contains
   `exported/` (`deploy/include/param.h`, `parser_policy_dir`). Creating any
   new `deploy/.../velocity/<name>/exported/` would silently change what the
   robot runs on the next boot. Nothing may be written under `deploy/`.

## 3. Plan: two runs, same seed, in parallel

| run | task id | actor obs | purpose |
|---|---|---|---|
| **A (primary)** | `Unitree-G1-23Dof-Duet-RoughBlind-Tall-H5` | 5 x 71 = 355-D | The candidate for litter. Needs a `deploy.yaml` with `history_length: 5` on each term; no C++ change. |
| **B (fallback and ablation)** | `Unitree-G1-23Dof-Duet-RoughBlind-Tall` | 71-D, identical layout to `-Flat` | Drop-in replacement for v8 (only `range`/`standstill` change in `deploy.yaml`). Also measures what history adds. |

A and B are identical except `actor_history`. The previous brief's flat-tall
control run is dropped: attributing the gain to terrain vs height does not help
the robot work, while B is both a deployable fallback and a clean measure of
the history change.

## 4. Environment setup

Work in the existing checkout on this machine
(`~/Desktop/DUET`, branch `arm-robustness-pass`). Clone
`https://github.com/bae-air-lab/DUET.git` only if it is absent.

```bash
conda create -n duet python=3.13 -y && conda activate duet
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
```

- Do **not** reuse the existing `unitree_rl_mjlab` conda env: it has
  mujoco-warp 3.5.0, while v8 was trained with 3.6.0.
- Do **not** `pip install` (editable or otherwise) this repo or
  `unitree_rl_mjlab`. Both ship a top-level `src` package. Always run from the
  repo root with `PYTHONPATH=.`. Check once:
  `PYTHONPATH=. python -c "import src, mjlab; print(src.__file__, mjlab.__version__)"`
  must print this repo's `src` and `1.2.0`.

## 5. Hard constraints

- Do not modify `deploy/` or `vla_bridge/`. Deployment files for the new
  policies go under `exported/<name>/` (section 10).
- Do not change nominal actuator gains, action scale, default joint pose,
  the arm disturbance generator, the existing reward terms' weights, or the
  PPO/entropy schedule in `rl_cfg.py`. Randomising gains *around* nominal
  (section 6.7) is allowed. Export metadata must still report the nominal gains.
- Existing tasks (`-Flat`, `-Rough`, `-Flat-DeployGains`, all `-Abl-*`) must
  build **bit-identical** configs to before. Put every change behind a new
  keyword argument or a new function parameter whose default reproduces
  current behaviour, and register new task ids. Gate G1 proves this.
- Per frame, the actor observation keeps exactly the `-Flat` term order and
  widths (the docstring at the top of `src/tasks/duet/duet_env_cfg.py`):
  `base_ang_vel(3), projected_gravity(3), command(3), phase(2),
  joint_pos(23), joint_vel(23), actions(13), height_command(1)` = 71.
- New reward terms: none, except the reference-frame fix in 6.5, which keeps
  the existing weight.
- Scope discipline: implement exactly sections 6.1-6.10. Put ideas beyond that
  in the report, not the code.
- Document every change and the reason for it in a new section
  "8. Rough-blind-tall pass (2026-09)" of `documents/duet/arm_robustness.md`,
  and add the new task ids to `src/tasks/duet/README.md`.

## 6. Changes

Files: `src/tasks/duet/config/g1_23dof/env_cfgs.py`, `__init__.py`,
`src/tasks/common/mdp/` (new MDP functions exported through its
`__init__.py`), `src/tasks/common/mdp/symmetry.py`,
`src/tasks/duet/rl/export_metadata.py`, `scripts/check_deploy_consistency.py`,
and new scripts under `scripts/`. Suggested new kwargs on
`unitree_g1_23dof_duet_rough_env_cfg` (all defaulting to current behaviour):
`height_max`, `actor_height_scan`, `actor_history`, `litter_terrain`,
`clearance_reference`, `contact_softness`, `friction_range`,
`sim2real_dr`, `terrain_curriculum`, `privileged_contact_obs`.

### 6.1 Nominal height, measured (`height_max`, default 0.73)

`HEIGHT_RANGE = (0.24, 0.73)` sets the command range at
`base_height_cmd.height_range = HEIGHT_RANGE`. Add
`height_max: float = HEIGHT_RANGE[1]` and use `(HEIGHT_RANGE[0], height_max)`
there. `BaseHeightCommand` treats `height_range[1]` as nominal, so the
nominal-height slice and the GUI slider follow. Squat floor, walk band and
command-mix schedules are unchanged, so walking spans
`[walk_min_height, height_max]`.

**Measure** the value. Write `scripts/duet_stand_height.py`:

- Mode 1 (sim): build `-Flat` with 1 env, hold the default pose under PD with
  zero action on the rigid plane for 3 s, and print
  `h = root_z - min(foot_site_z) + 0.02`. This is exactly the convention of
  `track_base_height` and `height_track_error` (`ankle_sole_distance=0.02`).
- Mode 2 (hardware pose): `--joints <12 leg angles>` computes the same `h` by
  forward kinematics with the pelvis level and the feet flat. The operator uses
  it with joint angles recorded from `rt/lowstate` while the **stock firmware**
  stands (section 11, step 0).

Set `height_max` to Mode 1's value rounded down to 0.01 (expected 0.78). If
Mode 1 gives a value outside 0.775-0.800, stop and report: the convention is
off somewhere. Read the achieved-height printout in
`deploy/include/isaaclab/envs/mdp/observations/observations.h` (read-only) and
state in the doc whether its convention matches `h`.

Consequences: `build_deploy_metadata` will export
`height_command_range = [0.24, height_max]`. Do not edit
`deploy/.../v0/params/deploy.yaml`; the new deploy files are written under
`exported/` (section 10).

### 6.2 Blind actor, privileged critic (`actor_height_scan`, `privileged_contact_obs`)

- `actor_height_scan=False` deletes `height_scan` from
  `cfg.observations["actor"]` only. The critic keeps it (privileged). Assert
  that the actor's term names equal the `-Flat` task's, in order.
- `privileged_contact_obs=True` adds two critic-only terms:
  `foot_friction_coef` (1-D: the current per-env foot friction, `geom_friction`
  axis 0 of the foot geoms) and `foot_softness` (2-D: current left/right foot
  `solref` timeconst from 6.6). Read them from `env.sim.model`. Add their
  mirror rules to `_TERM_RULES` in `symmetry.py` (friction `([0],[1.0])`,
  softness `([1,0],[1.0,1.0])`), or the symmetry code raises `KeyError`.
  Why: under heavy randomisation, value targets become noisy. The previous
  entropy runaway (see `arm_robustness.md` section 6) was driven by exactly
  that kind of unexplained advantage variance, and a critic that knows the
  ground removes much of it.

### 6.3 Actor observation history (`actor_history`, default 1)

- Set `cfg.observations["actor"].history_length = actor_history`, leaving
  `flatten_history_dim=True`. Leave the critic at 1.
- Layout, verified in the code: mjlab concatenates **per term**, each term's
  frames **oldest to newest** (`CircularBuffer.buffer`). The C++
  `ObservationManager` does the same when `use_gym_history` is false (the
  default; `ObservationTermCfg::get()`). Both backfill the history with the
  first frame after a reset. So the ONNX input is
  `[base_ang_vel x5, projected_gravity x5, ..., height_command x5]`.
  `use_gym_history` must stay false in any deploy file.
- **Symmetry.** `_group_perm_sign` reads `group_obs_term_dim`, which becomes
  `D*H` with history, and currently raises on the length mismatch. Tile each
  per-frame rule over the frames: `perm = [h*D + p for h in range(H) for p in
  rule]`, signs repeated H times. Read H from the group config.
- **Export metadata** (`export_metadata.py`): add `history_length`, report
  per-frame widths in `observation_dims`, and set `obs_dim` to the full input
  width.
- **Checker** (`check_deploy_consistency.py`): read `history_length` from the
  metadata (default 1 when absent), require each `deploy.yaml` term's
  `history_length` to equal it, compare per-frame widths, and require
  total width = `sum(widths) * H` = ONNX input width. `--expect-obs-dim`
  keeps meaning the per-frame width (default 71). Running the
  checker on `exported/arm_robust_v8_21000/policy.onnx` with the current
  `deploy.yaml` must give **the same result as before the change**.

### 6.4 Litter-like terrain (`litter_terrain`)

Real litter: loose shavings, hulls or straw, often several centimetres and up
to ~15 cm deep ([poultry litter](https://en.wikipedia.org/wiki/Poultry_litter),
[litter management](https://www.thepoultrysite.com/articles/poultry-litter-management)).
The surface has small bumps, gentle undulations, caked clumps and a gentle
floor slope. Build a `TerrainGeneratorCfg` with size 8x8, `num_rows=10`,
`num_cols=20`, `border_width=20`, `curriculum=True`, and keep the rough task's
collision settings (`ccd_iterations=500`, `nconmax=48`,
`contact_sensor_maxmatch=500`). Starting mix:

| sub-terrain | class | proportion | at difficulty 1 |
|---|---|---|---|
| flat | `BoxFlatTerrainCfg` | 0.15 | - |
| fine bumps | **new** `HfScaledRandomUniformTerrainCfg` | 0.30 | noise 0-0.04 m, step 0.005 |
| undulation | `HfPerlinNoiseTerrainCfg` | 0.20 | peak-to-peak <= 0.06 m, wavelength ~0.5-2 m |
| waves | `HfWaveTerrainCfg` | 0.10 | amplitude 0-0.04 m |
| slope up/down | `HfPyramidSlopedTerrainCfg` (+ `inverted=True`) | 0.075 + 0.075 | slope 0-0.15 (~8.5 deg) |
| clumps | `BoxRandomGridTerrainCfg` | 0.10 | `grid_height_range` (0, 0.025), `grid_width` 0.3 |

- `HfScaledRandomUniformTerrainCfg`: subclass `HfRandomUniformTerrainCfg` so
  the upper noise bound is interpolated by difficulty
  (`noise_range[0] + difficulty*(noise_range[1]-noise_range[0])`) and the
  noise is zero at difficulty 0. Put it in `src/tasks/common/`, not in mjlab.
- Perlin parameters (`scale`, `octaves`, `horizontal_scale`) are not
  specified here: tune them until the **measured** heightfield statistics meet
  the table (gate G6).
- Spawn check: the robot spawns at the tile origin with the keyframe height.
  If the feet start inside a bump and the robot pops on reset, add a small
  positive z to `reset_base` for this task only, and report the value.
- `max_init_terrain_level=5` stays.

### 6.5 Terrain-relative foot clearance (`clearance_reference`, default "world")

Add `reference: str = "world"` to `mdp.feet_clearance`. With
`"stance_foot"`, the foot height is `foot_z - min(foot_z)` per env, and the
target becomes `target_height - z_rest`, where `z_rest` is the foot site's
height above rigid flat ground at rest (**measure** it with the stand-height
script). On flat rigid ground the two references then give the same cost for
the same motion (gate G7). The weight is unchanged. The new tasks use
`"stance_foot"`; every existing task keeps `"world"`.

### 6.6 Soft and slippery ground (`contact_softness`, `friction_range`)

- New event `geom_solref` in `src/tasks/common/mdp/events.py`, modelled on
  `mjlab.envs.mdp.dr.geom.geom_friction`, decorated
  `@requires_model_fields("geom_solref")`, calling
  `_randomize_model_field(..., "geom_solref", entity_type="geom",
  default_axes=[0], valid_axes=[0, 1])`. `_randomize_model_field` lives in
  `mjlab.envs.mdp.dr._core`, a private module; import it anyway (mjlab is
  pinned at 1.2.0) and say so in a comment.
- Apply it **per foot**: two event terms, one on the seven
  `left_foot{1..7}_collision` geoms and one on the seven right, each with
  `shared_random=True`, `operation="abs"`, axis 0 (timeconst) only,
  dampratio left at 1. Fire each in `reset` mode and in `interval` mode with
  `interval_range_s=(1.0, 3.0)`, so the ground under each foot changes during
  an episode, as it does across patches of litter.
- **Calibrate** the range. Write `scripts/duet_contact_softness_sweep.py`:
  on the rigid plane, zero action, default pose, 16 envs, for timeconst in
  {0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30}, report the settled static
  sinkage (foot-site z relative to the 0.02 baseline), the max contacts per
  world, the sim throughput and any instability. Rough expectation from
  MuJoCo's formula with default `solimp` (d ~ 0.95): 0.10 -> ~5 mm,
  0.20 -> ~2 cm, 0.30 -> ~4 cm. That is a sanity check only; use the measured
  numbers.
- Choose `tc_max` as the timeconst whose static sinkage matches the operator's
  measured litter sinkage x1.25 (section 11, step 0). If no measurement exists
  yet, target 25 mm and flag it in the report. Hard caps: `tc_max <= 0.30`
  and no case where a foot passes more than 5 cm below the terrain surface
  (tunnelling). If real sinkage is above ~40 mm, soft contacts are outside the
  regime they can represent: cap at 40 mm and flag the granular-model follow-up
  (section 12).
- Sample uniformly in `[0.02, tc_max]`; the low end is rigid ground, so hard
  floors stay in the mix. Ramp `tc_max` in with a schedule rather than
  starting soft: knots `(0, 0.02)`, `(1000, 0.02 + 0.3*(tc_max-0.02))`,
  `(3000, tc_max)`. Implement it as a small curriculum term modelled on
  `push_magnitude` that rewrites the two events' `ranges`.
- The `foot_friction` range for the new tasks is `(0.2, 1.6)` (was
  `(0.3, 1.6)`). HOMIE uses 0.1-2.0 and DWL 0.2-2.0; 0.2 is a clean slip
  without making walking impossible. The mode stays `startup`.
- Verify that a mid-episode write to an expanded model field does not require
  a CUDA graph rebuild (it should not: only field expansion does), and that the
  privileged critic term reads the current value.

### 6.7 Sim-to-real randomisation (`sim2real_dr`)

Neither of these exists in the current config, and both are standard for
G1-class humanoids. They are insurance against the wobble seen on hardware.

- **PD gains** +-10% around nominal: `mjlab.envs.mdp.dr.pd_gains`, `startup`,
  `kp_range=(0.9, 1.1)`, `kd_range=(0.9, 1.1)`, `operation="scale"`, on the
  lower-body actuators only (`SceneEntityCfg("robot",
  actuator_names=LOWER_BODY_ACTUATORS)`). The same +-10% as HOMIE. Check on a
  4-env build that exactly the 13 lower-body actuators' `actuator_gainprm` /
  `actuator_biasprm` change and the 10 arm actuators' do not (`pd_gains`
  indexes `asset.actuators`, so confirm how actuator ids resolve).
- **Actuator latency** 0-20 ms: wrap the leg and waist actuator groups
  (`G1_ACTUATOR_7520_14`, `G1_ACTUATOR_7520_22`, `G1_ACTUATOR_ANKLE`) in
  `mjlab.actuator.DelayedActuatorCfg(base_cfg=..., delay_target="position",
  delay_min_lag=0, delay_max_lag=4)`. The physics step is 5 ms, so 4 lags is
  20 ms. Build a fresh robot cfg inside the env-cfg function; do not modify
  `g1_23dof_constants.py`, whose action-scale loop asserts
  `BuiltinPositionActuatorCfg`. Read `DelayBuffer` and choose
  `delay_update_period`/`delay_hold_prob` so the lag is **constant within an
  episode and re-sampled at reset**, not jittering every physics step.
- Export check: an export of a new task must report exactly the same
  `joint_stiffness`, `joint_damping`, `action_scale`, `action_offset` and
  `default_joint_pos` as a `-Flat` export (they are read from `mj_model`,
  which randomisation does not touch). Gate G8 diffs them.

### 6.8 Task-success terrain curriculum (`terrain_curriculum="task"`)

`terrain_levels_vel` promotes an env only after it walks 4 m and looks only at
the last command, which in DUET (30% standing, commands resampled every
3-8 s) almost never happens. A survival rule has the opposite flaw: it
promotes robots that stand still. Add `terrain_levels_task` to
`src/tasks/common/mdp/curriculums.py`, and a metric `moving_command_fraction`
(1 when `|v_xy_cmd| + |yaw_cmd| > 0.1`) to the metric set of the new tasks.

The curriculum manager runs **before** the metrics manager resets
(`ManagerBasedRlEnv._reset_idx`), so this episode's sums are still available:
`env.metrics_manager._episode_sums[name][env_ids] /
clamp(env.metrics_manager._step_count[env_ids], 1)`. These are private
attributes: assert they exist. Falls come from
`env.termination_manager.get_term("fell_over")[env_ids]` and time-outs from
`env.termination_manager.time_outs[env_ids]`.

Rule per finished episode:

- **promote** if it timed out without falling, `moving_command_fraction >= 0.25`,
  mean `lin_vel_track_error <= 0.20` m/s, mean `yaw_track_error <= 0.35` rad/s,
  mean `height_track_error <= 0.04` m, and mean `idle_root_drift <= 0.05` m;
- **demote** if it fell, or if `moving_command_fraction >= 0.25` and mean
  `lin_vel_track_error >= 0.35` m/s;
- otherwise **stay**. An episode spent mostly standing is not evidence of
  walking competence either way.

Return a dict so the curve shows `terrain_level_mean`, `promote_frac` and
`demote_frac`. Keep mjlab's behaviour of sending envs past the top row to a
random row.

### 6.9 Task registration (`__init__.py`)

Register, all with `unitree_g1_23dof_duet_ppo_runner_cfg()` and
`DuetOnPolicyRunner`:

- `Unitree-G1-23Dof-Duet-RoughBlind-Tall-H5`: all of 6.1-6.8 with
  `actor_history=5`.
- `Unitree-G1-23Dof-Duet-RoughBlind-Tall`: the same with `actor_history=1`.
- For each, two **evaluation** ids (use the eval cfg as both `env_cfg` and
  `play_env_cfg`). Each keeps the `terrain_scan` sensor and the critic
  layout, so checkpoints load:
  - `...-EvalFlat`: plane terrain, timeconst fixed at 0.02, the play
    settings. Its numbers are comparable to v8's scenario results.
  - `...-EvalLitter`: litter generator with `curriculum=False` and
    `difficulty_range=(0.8, 1.0)`, timeconst fixed at `tc_max`, friction
    fixed at 0.3, the play settings.
- The regular play cfg uses `randomize_terrain`, as the current rough play
  block does.

### 6.10 Tooling

`play.py`, `export_duet_onnx.py`, `duet_arm_scenarios.py`,
`duet_probe_locomotion.py`, `duet_rank_checkpoints.py`,
`select_duet_checkpoint.py` and `duet_probe_idle.py` already take `--task`.
Run each once against a new-task checkpoint (the smoke run's is fine) and fix
anything that assumes 71-D or the flat plane.

## 7. Gates before launch (all must pass; paste the outputs in the report)

- **G0 imports:** the check in section 4.
- **G1 no regression:** dump every pre-existing task's train and play env cfg
  and rl cfg to text before your first edit (on `arm-robustness-pass`) and
  again after; `diff` must be empty.
- **G2 observation layout:** print actor and critic term names and dims for
  `-Flat`, `-RoughBlind-Tall` (actor 71, same order as `-Flat`) and
  `-RoughBlind-Tall-H5` (actor 355 = 5 x 71, same term order, each width x5).
- **G3 symmetry:** for both new tasks, on a random batch, mirroring twice is
  the identity for actor obs, critic obs and actions, and `check_symmetry`
  runs.
- **G4 height:** `duet_stand_height.py` output, the chosen `height_max`, and
  the new tasks' built `base_height.height_range` = `(0.24, height_max)`.
- **G5 softness:** the sweep table, the chosen `tc_max` and the reason, and
  the tunnelling and contact-count checks.
- **G6 terrain:** a table per sub-terrain at difficulty 0, 0.5 and 1: min,
  max, peak-to-peak and max local slope of the generated surface. Fine bumps
  must be flat at 0 and grow with difficulty. Also save one screenshot or
  render via `scripts/visualize_terrain.py`.
- **G7 clearance:** on flat rigid ground, `"world"` and `"stance_foot"` give
  the same swing-foot height to within 2 mm.
- **G8 export:** export a smoke-run checkpoint of each new task and run the
  checker against a matching deploy file written to `exported/<name>/`
  (H5: every term `history_length: 5`, range `[0.24, height_max]`,
  `standstill: height_max`; H1: the v0 file with only `range` and
  `standstill` changed). Both pass. The gains/scale/offset/default-pose diff
  against a `-Flat` export is empty. The v8 check is unchanged.
- **G9 smoke run:** 20 iterations, 256 envs, each new task: no NaN, no
  contact/constraint buffer overflow warnings, and report iterations/s. Then
  delete those run directories.
- **G10 throughput:** about 50 iterations at 4096 envs for each task alone,
  then both together. Report the ETA to 20,000 iterations. If running both in
  parallel is over 4 days, launch A now, report, and start B after A.

## 8. Launch

1. Create branch `rough-blind-tall` from `arm-robustness-pass`. Commit this
   brief (it is uncommitted in the working tree) and the implementation, then
   `git push -u origin rough-blind-tall`. Do not commit `logs/` or `*.pt`
   (both gitignored).
2. Launch, one tmux session per run:

```bash
cd ~/Desktop/DUET && mkdir -p logs
CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
tmux new -d -s duet_rbt_h5 "bash -lc 'source $CONDA_SH && conda activate duet && cd ~/Desktop/DUET && \
  PYTHONPATH=. python scripts/train.py Unitree-G1-23Dof-Duet-RoughBlind-Tall-H5 \
  --agent.seed 1 --env.scene.num-envs 4096 --agent.run-name rough_blind_tall_h5 \
  2>&1 | tee logs/rough_blind_tall_h5.console.log'"
tmux new -d -s duet_rbt_h1 "bash -lc 'source $CONDA_SH && conda activate duet && cd ~/Desktop/DUET && \
  PYTHONPATH=. python scripts/train.py Unitree-G1-23Dof-Duet-RoughBlind-Tall \
  --agent.seed 1 --env.scene.num-envs 4096 --agent.run-name rough_blind_tall_h1 \
  2>&1 | tee logs/rough_blind_tall_h1.console.log'"
```

3. Keep 4096 envs: the curriculum and PPO settings were tuned at that batch
   size. Curricula saturate at iteration 7,000; train to at least 20,000
   (budget 25,001, checkpoint every 500). Keep whole run directories,
   tfevents included (v8's were lost).
4. TensorBoard:
   `conda activate duet && tensorboard --logdir ~/Desktop/DUET/logs/rsl_rl/DUET_G1_23dof --port 6006 --bind_all`

## 9. Monitoring and pre-agreed decisions

| signal | healthy | action if not |
|---|---|---|
| `Policy/mean_std` | below ~0.35 by iteration 3000 | If it climbs while `fell_over` stays ~0 and `action_rate_l2` tracks `-2.6*std^2`, it is the entropy runaway in `arm_robustness.md` section 6: stop and report. Do not change the schedule on your own. |
| `terrain_level_mean` | rises after ~1000-2000 | Below 1.0 at iteration 3000: stop, relax the promote thresholds once to lin 0.25 / yaw 0.45, resume, and document it. (Levels are not checkpointed and restart at random on resume; that is fine.) Above 8 by 2000: report that the terrain is too easy; change nothing. |
| `promote_frac` / `demote_frac` | both non-zero, promote rising | All zero: log bug. Report. |
| `Episode_Termination/fell_over` | < 1% of episodes after 7000 | Report. |
| `height_track_error` | <= ~0.03 m (v8 flat: 0.016-0.025) | Report. |
| torso pitch/roll, `idle_root_drift`, `foot_slip_speed` | flat or falling after 7000 | Compare A vs B. |
| it/s | stable | A drop over 30% suggests contact-buffer overflow: check the console. |

The only mid-run change you may make is the one threshold relaxation above.
Anything else: stop that run and report.

## 10. Checkpoint selection and export

- From iteration 10,000, every 500 checkpoints, rank with
  `duet_rank_checkpoints.py` on **both** `-EvalFlat` and `-EvalLitter`
  for each run. Choose zero falls on both first, then litter tracking and
  drift, then flat tracking. Also run `duet_arm_scenarios.py` on `-EvalFlat`.
- Export the chosen checkpoints to `exported/rough_blind_tall_h5_<iter>/` and
  `exported/rough_blind_tall_h1_<iter>/`, each with its matching
  `deploy.yaml` (section 7, G8), `check_deploy_consistency.py` output, and
  `deploy_contract.json`. Nothing goes under `deploy/`.

## 11. Hardware acceptance on representative litter (operator)

**Step 0: measure before trusting any sim number.**

- Litter depth where the robot will work (ruler, 5 spots).
- Sinkage under the **stock firmware** standing still: mark the litter surface
  beside a foot and measure how far the sole sits below it; repeat with the
  robot shifting weight onto one foot if possible. This number sets `tc_max`
  in 6.6. If it arrives after launch, compare it with the chosen `tc_max`; a
  large mismatch means a re-run.
- The stock firmware's standing height: record the leg joint angles from
  `rt/lowstate` while it stands and run `scripts/duet_stand_height.py
  --joints ...`.

**Test bed.** At least 3 m x 2 m of the same litter material at the working
depth, with one loose/fluffed zone, one compacted or caked zone, and a
transition onto concrete. Harness or gantry with slack, e-stop, spotter.

**Protocol.** Run each test at least 5 times on the stock firmware
(reference), v8, run B and run A. Record video and `rt/lowstate`.

| # | test | pass (candidate must also match or beat the stock firmware) |
|---|---|---|
| 1 | Stand 60 s at nominal height, arms at rest | no fall, no step, pelvis drift <= 3 cm, no visible oscillation |
| 2 | Stand 60 s while scripted arm trajectories run on `rt/arm_targets` (both arms forward, asymmetric reach) | no fall, drift <= 5 cm, no corrective stepping |
| 3 | Height sweep standing: nominal -> 0.50 -> nominal; walking 0.2 m/s at 0.65 m | no fall, achieved height within +-3 cm (controller printout) |
| 4 | Walk 2 m forward and back at 0.2 and 0.4 m/s; 1 m sideways at 0.2 m/s; 360 deg turn in place | no fall, heading drift <= 10 deg, path deviation <= 20 cm |
| 5 | Concrete -> litter -> concrete at 0.3 m/s | no stumble, no fall |

**Deploying a candidate.** Copy the policy into a new folder under
`deploy/robots/g1_23dof/config/policy/velocity/`. Because the controller loads
the last-sorted folder with `exported/`, the new folder becomes the default;
move it out to return to v8. The H5 policy needs its own `deploy.yaml` with
`history_length: 5` on every term; the H1 policy needs only the new `range` and
`standstill`.

## 12. Second pass (not in this task)

In order of likely value, if the hardware tests fall short:

1. **Sinkage > 3-4 cm or feet trapped:** a granular contact model (3D resistive
   force theory as in GM-Loco) applied at the feet, or a teacher-student
   setup with a terrain latent.
2. **Stamping or rolling on the ankle:** foot-orientation (HOMIE "feet
   parallel") and landing-impact penalties; allow higher swing clearance on
   soft ground.
3. **Residual wobble:** Lipschitz-constrained policy gradient penalty
   ([arXiv 2410.11825](https://arxiv.org/abs/2410.11825)).
4. **Heavy carried loads:** an end-effector force curriculum as in FALCON
   ([arXiv 2505.06776](https://arxiv.org/abs/2505.06776)).

## 13. Report back

- Outputs of gates G0-G10, with `height_max`, `tc_max`, `z_rest` and the
  terrain statistics.
- tmux session names, full run directory paths, the TensorBoard command, and
  the ETA.
- At the end of training, per run: final return, fall rate, height, velocity
  and yaw tracking error, terrain level, torso pitch/roll, idle drift and foot
  slip; the ranked checkpoints on both eval tasks; the chosen exports and
  checker results; the A-vs-B comparison.
- Anything that deviated from this brief, and why.
