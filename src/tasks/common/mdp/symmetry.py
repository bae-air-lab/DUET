"""Left-right mirror symmetry for the G1-23DOF lower-body policy.

Used as rsl_rl PPO's ``data_augmentation_func``: every (obs, action) sample can
be augmented with its sagittal-plane (left<->right) mirror, and/or the actor can
be penalised for disagreeing with its own mirror (the symmetry loss). This is
HOMIE's third contribution (``use_flip`` + ``symmetry_scale`` in its PPO config).

The mirror is a per-term index permutation + sign flip, built automatically from
the live observation layout and the robot's joint names, so it stays correct if
the obs order changes. It is self-inverse; :func:`check_symmetry` verifies
``mirror(mirror(x)) == x`` on real observations.

Sagittal mirror = reflect y -> -y:
  - polar vector (lin vel, gravity): y flips           -> sign [+, -, +]
  - axial vector (ang vel, pseudovector): x,z flip     -> sign [-, +, -]
  - twist command [vx, vy, yaw_rate]: vy, yaw flip     -> sign [+, -, -]
  - gait phase [sin, cos]: legs swap == phase + pi     -> both negate
  - per-foot scalars: swap left<->right
  - per-foot 3-vectors (contact force): swap feet + flip each y
  - joints: swap left<->right; flip roll/yaw DOFs (lateral), keep pitch/knee/elbow
  - height scan: reflect the ray grid about its own y axis (see below)
  - arm_traj_vel (critic): joint rule over the arm joints only
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

__all__ = [
  "mirror_g1_obs_actions",
  "check_symmetry",
  "build_symmetry_cfg",
  "SYMMETRY_MODES",
]

# Per-term mirror rule for the fixed-width, non-joint observation terms.
# Terms whose width depends on the scene (height_scan) are built at runtime.
_TERM_RULES: dict[str, tuple[list[int], list[float]]] = {
  "base_ang_vel": ([0, 1, 2], [-1.0, 1.0, -1.0]),
  "projected_gravity": ([0, 1, 2], [1.0, -1.0, 1.0]),
  "command": ([0, 1, 2], [1.0, -1.0, -1.0]),
  "phase": ([0, 1], [-1.0, -1.0]),
  "base_lin_vel": ([0, 1, 2], [1.0, -1.0, 1.0]),
  "foot_height": ([1, 0], [1.0, 1.0]),
  "foot_air_time": ([1, 0], [1.0, 1.0]),
  "foot_contact": ([1, 0], [1.0, 1.0]),
  "foot_contact_forces": ([3, 4, 5, 0, 1, 2], [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]),
  "height_command": ([0], [1.0]),
}

# Caches keyed by env id: action perm/sign (no obs needed) and per-group perm/sign.
_act_cache: dict = {}
_grp_cache: dict = {}


def _joint_perm_sign(joint_names: list[str]) -> tuple[list[int], list[float]]:
  """Mirror map over the robot's joints: swap left<->right, flip roll/yaw DOFs."""
  idx = {n: i for i, n in enumerate(joint_names)}
  perm: list[int] = []
  sign: list[float] = []
  for n in joint_names:
    if n.startswith("left_"):
      mate = "right_" + n[len("left_") :]
    elif n.startswith("right_"):
      mate = "left_" + n[len("right_") :]
    else:
      mate = n  # central joint (e.g. waist_yaw)
    perm.append(idx[mate])
    sign.append(-1.0 if ("roll" in n or "yaw" in n) else 1.0)
  return perm, sign


def _height_scan_perm(env, group: str, dim: int) -> tuple[list[int], list[float]]:
  """Mirror map for a raycast height scan: reflect the ray grid about y = 0.

  Built by regenerating the sensor's own ray offsets and, for each ray at
  ``(x, y)``, finding the ray at ``(x, -y)``. Doing it by lookup rather than by
  index arithmetic keeps this correct for any pattern layout or flattening
  order, instead of silently producing a wrong permutation if either changes.

  Without this, ``_group_perm_sign`` raises ``KeyError`` on the rough-terrain
  task, whose actor group contains ``height_scan`` (the flat task deletes it) --
  which is why symmetry had only ever run on the flat task.
  """
  om = env.unwrapped.observation_manager
  term_cfg = om.get_term_cfg(group, "height_scan")
  sensor = env.unwrapped.scene[term_cfg.params["sensor_name"]]
  offsets, _ = sensor.cfg.pattern.generate_rays(None, "cpu")  # [N, 3]
  if offsets.shape[0] != dim:
    raise ValueError(
      f"symmetry: height_scan has {dim} values but the pattern generates "
      f"{offsets.shape[0]} rays; cannot build a mirror map."
    )
  xy = offsets[:, :2]
  # Quantise to avoid float-equality misses on the regenerated grid.
  key = {
    (round(float(x), 4), round(float(y), 4)): i for i, (x, y) in enumerate(xy)
  }
  perm: list[int] = []
  for x, y in xy:
    mate = key.get((round(float(x), 4), round(-float(y), 4)))
    if mate is None:
      raise ValueError(
        "symmetry: height_scan ray grid is not symmetric about y = 0; "
        "no partner ray for (%.4f, %.4f)." % (float(x), float(y))
      )
    perm.append(mate)
  return perm, [1.0] * dim  # terrain height is a scalar, unchanged by reflection


def _group_perm_sign(env, group: str, jp: list[int], js: list[float]):
  om = env.unwrapped.observation_manager
  names = om.active_terms[group]
  dims = [d[0] for d in om.group_obs_term_dim[group]]
  flat_perm: list[int] = []
  flat_sign: list[float] = []
  off = 0
  for name, dim in zip(names, dims, strict=True):
    if name in ("joint_pos", "joint_vel"):
      p, s = jp, js
    elif name == "actions":
      p, s = jp[:dim], js[:dim]  # the lower-body action joints (first `dim`)
    elif name == "height_scan":
      p, s = _height_scan_perm(env, group, dim)
    elif name == "arm_traj_vel":
      # Per-arm-joint vector in the arm action term's joint order: same rule
      # as the joint terms (swap left<->right, flip roll/yaw), restricted to
      # the driven joints.
      arm = env.unwrapped.action_manager.get_term("upper_body_pose")
      p, s = _joint_perm_sign(list(arm.joint_names))
    elif name in _TERM_RULES:
      p, s = _TERM_RULES[name]
    else:
      raise KeyError(f"symmetry: no mirror rule for obs term '{name}'")
    if len(p) != dim:
      raise ValueError(f"symmetry: term '{name}' rule len {len(p)} != obs dim {dim}")
    flat_perm += [off + pi for pi in p]
    flat_sign += list(s)
    off += dim
  dev = env.unwrapped.device
  return (
    torch.tensor(flat_perm, dtype=torch.long, device=dev),
    torch.tensor(flat_sign, dtype=torch.float, device=dev),
  )


def _action_ps(env, n_act: int):
  key = id(env)
  if key not in _act_cache:
    jp, js = _joint_perm_sign(env.unwrapped.scene["robot"].joint_names)
    dev = env.unwrapped.device
    # inference_mode(False): these tensors are cached for the lifetime of the
    # env and reused inside PPO's update, where they are indexed into graphs
    # that get backpropagated. If the first caller happens to be under
    # torch.inference_mode() -- the startup involution check, or any eval
    # script -- the cache would hold inference tensors and every later training
    # step would die with "Inference tensors cannot be saved for backward".
    with torch.inference_mode(False):
      _act_cache[key] = (
        torch.tensor(jp[:n_act], dtype=torch.long, device=dev),
        torch.tensor(js[:n_act], dtype=torch.float, device=dev),
      )
  return _act_cache[key]


def _group_ps(env, obs: TensorDict):
  key = id(env)
  if key not in _grp_cache:
    jp, js = _joint_perm_sign(env.unwrapped.scene["robot"].joint_names)
    with torch.inference_mode(False):  # see _action_ps
      _grp_cache[key] = {g: _group_perm_sign(env, g, jp, js) for g in obs.keys()}
  return _grp_cache[key]


def mirror_g1_obs_actions(env, obs=None, actions=None):
  """rsl_rl data_augmentation_func: returns ([orig; mirror], [orig; mirror]).

  rsl_rl calls this in three ways: with both obs+actions (data augmentation),
  with ``actions=None`` (augment obs only), and with ``obs=None`` (mirror the
  action mean for the symmetry loss). Handle each.
  """
  aug_act = None
  if actions is not None:
    ap, asg = _action_ps(env, actions.shape[1])
    aug_act = torch.cat([actions, actions[:, ap] * asg], dim=0)

  aug_obs = None
  if obs is not None:
    groups = _group_ps(env, obs)
    n = obs.batch_size[0]
    out = {}
    for g in obs.keys():
      perm, sign = groups[g]
      out[g] = torch.cat([obs[g], obs[g][:, perm] * sign], dim=0)
    aug_obs = TensorDict(out, batch_size=[n * 2])

  return aug_obs, aug_act


def check_symmetry(env, obs: TensorDict, actions: torch.Tensor) -> dict[str, float]:
  """Verify the mirror is an involution on real data. Returns max |x - m(m(x))|.

  A permutation-plus-sign map is only a valid reflection if applying it twice is
  the identity. Anything else means a term's rule is wrong (a bad partner index
  or an unpaired sign), which would inject systematically corrupted samples into
  training rather than fail loudly. Called by the smoke test and by the runner
  once at startup.
  """
  mo, ma = mirror_g1_obs_actions(env, obs=obs, actions=actions)
  assert mo is not None and ma is not None
  n = obs.batch_size[0]
  mirrored_obs = TensorDict({g: mo[g][n:] for g in obs.keys()}, batch_size=[n])
  ro, ra = mirror_g1_obs_actions(env, obs=mirrored_obs, actions=ma[n:])
  assert ro is not None and ra is not None
  errs = {g: float((ro[g][n:] - obs[g]).abs().max()) for g in obs.keys()}
  errs["actions"] = float((ra[n:] - actions).abs().max())
  return errs


SYMMETRY_MODES = ("none", "augment", "loss", "both")

_FUNC_PATH = "src.tasks.common.mdp.symmetry:mirror_g1_obs_actions"


def build_symmetry_cfg(mode: str, mirror_loss_coeff: float = 1.0) -> dict:
  """Build the rsl_rl ``symmetry_cfg`` dict for an ablation mode.

  ``none`` still passes the mirror function to rsl_rl with both switches off.
  rsl_rl then computes the symmetry loss without adding it to the objective
  ("Symmetry not used for learning. We will use it for logging instead."), so
  ``Loss/symmetry`` is reported for every variant -- including the one that does
  not use symmetry, which is the variant the number is most interesting for.
  """
  if mode not in SYMMETRY_MODES:
    raise ValueError(f"symmetry mode must be one of {SYMMETRY_MODES}, got {mode!r}")
  return {
    "use_data_augmentation": mode in ("augment", "both"),
    "use_mirror_loss": mode in ("loss", "both"),
    "mirror_loss_coeff": mirror_loss_coeff,
    "data_augmentation_func": _FUNC_PATH,
  }
