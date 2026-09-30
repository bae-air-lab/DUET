# Task: blind rough-terrain DUET policy at nominal standing height

Brief for the agent setting up and launching the next training run on the
training PC (RTX PRO 6000, 96 GB). Read it fully before editing anything.

## Why

The deployed policy `exported/arm_robust_v8_21000` (task
`Unitree-G1-23Dof-Duet-Flat`, run `2026-09-16_19-38-13_arm_robust_v8_turn`,
commit `a992697`) walks on the real G1 but is fragile on soft poultry litter:
compliant, slightly uneven, slippery ground the feet sink into. The stock
Unitree firmware is stable on the same surface. Two causes to address:

1. It was trained only on a rigid, flat plane.
2. It stands 5-6 cm crouched. The top of the height command range is 0.73 m,
   but the default joint pose is 0.786 m tall, so at `standstill: 0.73` the
   robot always holds a squat, and the pose regulariser and `track_base_height`
   pull against each other there (see the `std_standing` comment in
   `env_cfgs.py`).

Train two new policies FROM SCRATCH: a blind rough-terrain policy at a 0.78 m
nominal height, plus a flat control at 0.78 m, so a hardware improvement can
be attributed to terrain or to height.

## Setup

```bash
git clone -b arm-robustness-pass https://github.com/bae-air-lab/DUET.git
cd DUET
conda create -n duet python=3.13 -y && conda activate duet
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
```

`requirements-train.txt` holds the exact versions that trained v8. mjlab is
stock 1.2.0 (no local patches).

**Gotcha:** do NOT `pip install -e` the sibling repo `unitree_rl_mjlab`; both
repos ship a top-level package called `src`. Always run from the repo root
with `PYTHONPATH=.`, e.g. `PYTHONPATH=. python scripts/train.py ...`, or
`import src.tasks` may load the wrong repo and the Duet task ids won't exist.

## Hard constraints

- The actor observation stays EXACTLY 71-D, in the current order (see the
  docstring at the top of `src/tasks/duet/duet_env_cfg.py`). The C++
  controller and `deploy.yaml` must run the new ONNX with no code changes.
- Do not change actuator gains, action scale, `deploy/`, `vla_bridge/`, the
  arm disturbance generator, the reward weights, or the PPO/entropy schedule
  in `rl_cfg.py`.
- Existing tasks (`-Flat`, `-Rough`, `-Flat-DeployGains`, ablations) must
  build bit-identical configs to before: put every change behind new keyword
  arguments whose defaults reproduce current behaviour, and register NEW task
  ids for the new variants.
- Keep reward and config edits minimal. Document each change and the reason
  for it in a new section of `documents/duet/arm_robustness.md`.

## Changes

All in `src/tasks/duet/config/g1_23dof/env_cfgs.py` and `__init__.py`, plus
`src/tasks/common/mdp/` for new MDP functions (exported through
`src/tasks/common/mdp/__init__.py`).

### 1. Height at nominal: `height_max` kwarg (default 0.73)

`HEIGHT_RANGE = (0.24, 0.73)` is read in `unitree_g1_23dof_duet_rough_env_cfg`,
`apply_duet_curriculum`, `pin_duet_full_distribution` and the play block.
Add a `height_max: float = HEIGHT_RANGE[1]` kwarg and use
`(HEIGHT_RANGE[0], height_max)` wherever the command range is set.
`BaseHeightCommand` already treats `height_range[1]` as the nominal height, so
the nominal-height envs follow automatically. The new tasks use
`height_max=0.78`.

- Keep the squat floor, walk band and command-mix schedules as they are. The
  walk band then spans [walk_min_height, 0.78].
- `build_deploy_metadata` reads the command range, so the exported
  `height_command_range` becomes [0.24, 0.78]. The existing `deploy.yaml`
  range [0.24, 0.73] remains a subset and passes
  `scripts/check_deploy_consistency.py`. Do NOT edit `deploy.yaml`; the
  operator sets `standstill: 0.78` when deploying.

### 2. Blind rough variant: height scan critic-only

`unitree_g1_23dof_duet_rough_env_cfg` currently leaves `height_scan` in the
ACTOR group (a 17x11 = 187-ray grid, so the actor would be 258-D and not
deployable). For the new variant, delete `height_scan` from
`cfg.observations["actor"]` only and keep it in the critic as privileged
information. Assert that the actor term order equals the flat task's.

### 3. Litter-like terrain instead of `ROUGH_TERRAINS_CFG`

The base config uses mjlab `ROUGH_TERRAINS_CFG` (stairs up to 10 cm, slopes
up to 1.0, waves up to 20 cm). That's too harsh for a blind policy and not
what litter looks like. Build a new `TerrainGeneratorCfg` (keep size 8x8,
num_rows 10, num_cols 20, border 20, curriculum=True; difficulty scales each
range by row). Starting mix:

| sub-terrain | class | proportion | range at max difficulty |
|---|---|---|---|
| flat | `BoxFlatTerrainCfg` | 0.20 | - |
| bumps | `HfRandomUniformTerrainCfg` | 0.35 | noise 0.01-0.04 m, step 0.005 |
| waves | `HfWaveTerrainCfg` | 0.15 | amplitude 0-0.05 m |
| slope up/down | `HfPyramidSlopedTerrainCfg` (+ inverted) | 0.075 + 0.075 | slope 0-0.2 (~11 deg) |
| small blocks | `BoxRandomGridTerrainCfg` | 0.15 | grid height 0-0.04 m, grid width 0.4 |

Check the field names against the installed `mjlab/terrains/` before using
them. Keep the rough task's collision settings
(`ccd_iterations=500`, `nconmax=48`, `contact_sensor_maxmatch=500`).

### 4. Soft, slippery ground: contact-softness randomisation

Litter is compliant. mjlab 1.2 has no solref randomiser, but `geom_solref` is
a per-world (batched) field in mujoco-warp. Add an event in
`src/tasks/common/mdp/events.py` modelled on `mjlab.envs.mdp.dr.geom.geom_friction`:

```python
@requires_model_fields("geom_solref")
def geom_solref(env, env_ids, ranges, asset_cfg=..., distribution="uniform",
                operation="abs", axes=None, shared_random=False):
  _randomize_model_field(env, env_ids, "geom_solref", entity_type="geom",
                         ranges=ranges, distribution=distribution,
                         operation=operation, asset_cfg=asset_cfg, axes=axes,
                         shared_random=shared_random,
                         default_axes=[0], valid_axes=[0, 1])
```

Apply it at `startup` to the 14 foot collision geoms (the same `geom_names`
used by `foot_friction`) with timeconst (axis 0) in 0.02-0.05 s and
`shared_random=True`. MuJoCo mixes the two geoms' solref, so the foot value is
effectively averaged with the terrain's 0.02. timeconst must stay >= 2x the
0.005 s timestep. Verify in play that standing sinkage is a few mm to ~1-2 cm
and nothing tunnels. If it looks wrong, widen or narrow the range and
document the numbers.

Also widen the low end of `foot_friction` from (0.3, 1.6) to (0.2, 1.6), for
the new variant only.

### 5. Terrain curriculum that suits a mostly standing policy

`mdp.terrain_levels_vel` promotes only when the robot has walked more than
4 m (half the tile) in an episode, and uses only the LAST command. In DUET,
about 30% of envs stand under an idle anchor and many squat, so envs would
almost never be promoted. Add `terrain_levels_survival` to
`src/tasks/common/mdp/curriculums.py`:

- promote an env that reached the episode time-out without a `fell_over`
  termination
- demote an env that fell
- read terminations from `env.termination_manager`

Use it for the new variant in place of `terrain_levels_vel`, and log the mean
level (the curriculum return value already is). `max_init_terrain_level=5`
stays.

### 6. Register the new tasks (`__init__.py`)

- `Unitree-G1-23Dof-Duet-RoughBlind`: rough cfg with `height_max=0.78`, blind
  actor, litter terrain, solref and friction changes, survival terrain
  curriculum. The play variant uses `randomize_terrain` as the current rough
  play block does.
- `Unitree-G1-23Dof-Duet-Flat-Tall`: flat cfg with `height_max=0.78` and
  nothing else changed (the control).

Both use `unitree_g1_23dof_duet_ppo_runner_cfg()` and `DuetOnPolicyRunner`.

### 7. Tooling

`scripts/export_duet_onnx.py` takes `--task`. Check that
`duet_arm_scenarios.py`, `duet_probe_locomotion.py`, `duet_rank_checkpoints.py`
and `play.py` accept a task argument. A RoughBlind checkpoint does not load
into the Flat task (its critic input includes the height scan), so add a
`--task` flag wherever one is hard-coded. For the scenario tests, a flat
terrain option in the RoughBlind play config keeps their numbers comparable
with v8.

## Before launching

1. Build both new env configs and print the actor/critic observation dims and
   term order. The actor must be 71 with the same order as `-Flat`.
2. Run each task for about 20 iterations with 256 envs to catch shape and
   solver errors, then delete those run directories.
3. Commit the changes on a new branch (`rough-blind-tall`) and push it.

## Launch (two runs in parallel, one GPU, in tmux)

```bash
PYTHONPATH=. python scripts/train.py Unitree-G1-23Dof-Duet-RoughBlind \
    --agent.seed 1 --env.scene.num-envs 4096 --agent.run-name rough_blind_tall
PYTHONPATH=. python scripts/train.py Unitree-G1-23Dof-Duet-Flat-Tall \
    --agent.seed 1 --env.scene.num-envs 4096 --agent.run-name flat_tall
```

Keep 4096 envs: the curriculum and PPO settings were tuned at that batch
size. The curricula saturate at iteration 7000. Train to 20000; checkpoints
land every 500 in `logs/rsl_rl/DUET_G1_23dof/<run>/`. KEEP the whole run
directories, tfevents included (v8's were lost). `logs/` and `*.pt` are
gitignored, so copy them back with rsync, not git.

## What to watch (TensorBoard)

- **Entropy runaway, the failure mode of the first arm-robustness run.** If
  `Policy/mean_std` climbs while `Episode_Termination/fell_over` stays about
  0, compare `Episode_Reward/action_rate_l2` with `-2.6 * std^2`. If they
  match, the policy is paying for its own exploration noise; it is not a task
  failure. Healthy: std below about 0.35 by iteration 3000.
- **Terrain level:** the mean should rise steadily. If it stays flat, the
  promotion rule is too strict. If it saturates by about iteration 2000, the
  terrain is too easy.
- **`height_track_error`:** should be at or below about 0.025 m, as in v8.
  Also check that a 0.78 command actually stands at 0.78.
- **Torso pitch and roll, `idle_root_drift`, `foot_slip_speed`:** compare with
  v8 and between the two runs.

## Report back

Final return, falls, height error and terrain level per run, the best
checkpoints by `scripts/duet_rank_checkpoints.py`, and ONNX exports of the
chosen checkpoints under `exported/`, with `check_deploy_consistency.py`
results.
