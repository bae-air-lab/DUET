# DUET — G1-23DOF decoupled loco-manipulation lower-body policy

An RL policy owns the **13 lower-body joints** (12 legs + `waist_yaw`). The 10
arm joints are **never actions** — they appear only as observations, driven by a
disturbance generator during training and by the VLA (`rt/arm_targets`) at
deployment. The exported policy is drop-in compatible with the existing C++
controller: **obs `[1,71]` → actions `[1,13]`**.

Design rationale, including the justification for every reward weight and the
payload numbers, is in [`documents/duet/reward_design.md`](../../../documents/duet/reward_design.md).

## Task IDs

| id | use |
|---|---|
| `Unitree-G1-23Dof-Duet-Flat` | **the one that gets deployed** (71-D obs) |
| `Unitree-G1-23Dof-Duet-Rough` | rough terrain (adds `height_scan`, 258-D obs — not deployable) |
| `Unitree-G1-23Dof-Duet-Flat-DeployGains` | A/B against the unitree_rl_gym gain table |
| `Unitree-G1-23Dof-Duet-Abl-*` | 10 matched-budget ablations, one variable each |

## Train

```bash
python scripts/train.py Unitree-G1-23Dof-Duet-Flat \
    --agent.seed 1 --env.scene.num-envs 4096
```

Budget 25,001 iterations, checkpoint every 500, curricula saturate at 7,000.
The entropy coefficient is a **schedule** (0.01 → 0.004 over iterations
7,000–9,000), logged to tfevents as `entropy_coef`; there is no constant to
hand-edit between fresh and resumed runs.

Multiple seeds:

```bash
scripts/train_duet_seeds.sh "1 2 3"          # sequential, GPU 0
scripts/train_duet_seeds.sh "1 2 3" "0 1 2"  # one seed per GPU
```

## Resume

```bash
python scripts/train.py Unitree-G1-23Dof-Duet-Flat \
    --agent.resume True --agent.load-run <run-dir-name>
```

**No source edit is required.** Every curriculum — arm-motion ramp, squat-depth
floor, velocity stages, command mix — reads `env.common_step_counter`, which
`MjlabOnPolicyRunner` writes into the checkpoint and restores on load. The
predecessor task's `RESUME_AT_FULL_DISTRIBUTION` global does not exist here;
resuming continues on the distribution it left off.

## Evaluate

Fixed protocol (1024 envs, 20 s episodes, arm curriculum pinned at full
strength) — the same one the manuscript's 36 conditions use:

```bash
python /home/rbbist.lab/paper_rl/duet_bench/duet_eval.py \
    --task Unitree-G1-23Dof-Duet-Flat \
    --checkpoint logs/rsl_rl/DUET_G1_23dof/<run>/model_12500.pt \
    --tag duet_12500 --num-envs 1024
```

`--task` defaults to the old `Unitree-G1-23Dof-LocoManip-Flat`, so every
previously recorded result is reproduced unchanged by its original command line.

**Select the deployable checkpoint by measurement, not by recency** — from
iteration 10,000, every 2,500:

```bash
python scripts/select_duet_checkpoint.py --run-dir logs/rsl_rl/DUET_G1_23dof/<run>
```

Ranks by fall rate first, then velocity/height tracking error, and prints the
export command for the winner.

### Reported metrics (HOMIE's set)

| metric | where |
|---|---|
| linear velocity error (m/s) | `mean_vxy_err_mps` (duet_eval) |
| angular velocity error (rad/s) | `mean_vyaw_err_radps` (duet_eval) |
| height error (m) | `mean_height_err_m` (duet_eval) |
| living time (s) | `mean_living_time_s` (duet_eval) |
| symmetry loss | `Loss/symmetry` (tfevents) — logged in **every** symmetry mode, including `none` |

## Export and deploy

```bash
python scripts/export_duet_onnx.py \
    --task Unitree-G1-23Dof-Duet-Flat \
    --checkpoint logs/rsl_rl/DUET_G1_23dof/<run>/model_12500.pt \
    --out-dir exported/duet_12500
```

This writes `policy.onnx` with the full deployment contract in its metadata
(joint order, PD gains, action scale/offset, observation layout, trained command
ranges, and a `deploy_config_hash`), plus a readable `deploy_contract.json`, and
then **runs the consistency checker automatically**. Non-zero exit = do not
deploy.

To check an existing policy against a config at any time:

```bash
python scripts/check_deploy_consistency.py \
    --onnx   deploy/robots/g1_23dof/config/policy/velocity/v0/exported/policy.onnx \
    --deploy-yaml deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml
```

It verifies the ONNX graph is `[1,71] → [1,13]`, that joint order, gains, action
scale/offset and observation order all match, and that deploy-side command
ranges lie inside the trained ranges. Policies exported before this existed
report `SKIP` for the fields they lack — surfaced, never silently passed.

To deploy: copy `policy.onnx` over
`deploy/robots/g1_23dof/config/policy/velocity/v0/exported/policy.onnx`.
`deploy.yaml` needs **no change** — the gains (hip 40.2, knee 99.1, ankle 28.5),
action scale (0.55/0.35/0.44), joint order and height range `[0.12, 0.73]` are
all unchanged from the current deployment.

## Ablations

Each variant changes exactly one thing against a matched budget (6,001
iterations, curriculum compressed 2×):

| variant | ablates |
|---|---|
| `Abl-Reference` | — (control) |
| `Abl-UniformArm` | HOMIE (a) weakened: 100% uniform arm goals, no task-pose anchors |
| `Abl-NoArmCurriculum` | HOMIE (a) removed: arms pinned at default |
| `Abl-NoHeightCmd` | HOMIE (b) removed: height pinned at 0.73 |
| `Abl-NoSymmetry` | HOMIE (c) removed (still logs `Loss/symmetry`) |
| `Abl-SymAugmentOnly` | HOMIE (c) partial: augmentation, no mirror loss |
| `Abl-NoPayload` | payload randomisation |
| `Abl-NoCurriculum` | staged command curriculum |
| `Abl-FlatEntropy` | the two-stage entropy schedule (constant 0.01) |
| `Abl-LowVelNoise` | `joint_vel` obs noise ±1.5 → ±0.3 (physical value) |

## Layout

```
src/tasks/common/mdp/       shared term classes (also re-exported by loco_manip,
                            so duet_eval's isinstance checks pass for both)
src/tasks/duet/
  duet_env_cfg.py           reward set + observation layout (the 71-D contract)
  config/g1_23dof/
    env_cfgs.py             gains, joint carve-out, payload, curriculum timing
    rl_cfg.py               PPO + entropy schedule + symmetry mode
    ablations.py            10 matched-budget variants
  rl/runner.py              entropy schedule, symmetry modes, export hook
  rl/export_metadata.py     the deployment contract written into the ONNX
```
