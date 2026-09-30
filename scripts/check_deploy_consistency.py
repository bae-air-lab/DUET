#!/usr/bin/env python3
"""Verify an exported policy matches the deploy.yaml it will run under.

This exists because a stale policy was once deployed against a mismatched
config and the robot misbehaved. The ONNX file on its own is opaque: nothing in
it records the PD gains, action scale or joint order it was trained against, so
the mismatch was undetectable until hardware.

``DuetOnPolicyRunner`` writes that contract into the ONNX metadata at export
(see ``src/tasks/duet/rl/export_metadata.py``). This script reads it back and
compares it, field by field, against the YAML the C++ controller loads.

Usage:
  python scripts/check_deploy_consistency.py \\
      --onnx  deploy/robots/g1_23dof/config/policy/velocity/v0/exported/policy.onnx \\
      --deploy-yaml deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml

Exit code 0 = safe to deploy, 1 = mismatch (details printed), 2 = cannot check.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import onnx
import yaml

# deploy.yaml observation key -> training observation term name. The two sides
# use different names for the same quantity; the ORDER is what must agree, since
# that is the concatenation order of the network input.
OBS_NAME_MAP = {
  "base_ang_vel": "base_ang_vel",
  "projected_gravity": "projected_gravity",
  "velocity_commands": "command",
  "gait_phase": "phase",
  "joint_pos_rel": "joint_pos",
  "joint_vel_rel": "joint_vel",
  "last_action": "actions",
  "base_height_command": "height_command",
}

# Absolute tolerance for gains/scales. Gains are written to 1 decimal place in
# deploy.yaml (e.g. 40.2) while training carries full precision, so an exact
# comparison would false-positive on rounding alone.
ATOL_GAIN = 0.05
ATOL_SCALE = 5e-3
ATOL_POS = 1e-3


class Report:
  def __init__(self) -> None:
    self.failures: list[str] = []
    self.warnings: list[str] = []
    self.checks = 0

  def check(self, ok: bool, name: str, detail: str = "") -> None:
    self.checks += 1
    if ok:
      print(f"  \033[32mPASS\033[0m  {name}")
    else:
      print(f"  \033[31mFAIL\033[0m  {name}{(': ' + detail) if detail else ''}")
      self.failures.append(name)

  def warn(self, name: str, detail: str) -> None:
    print(f"  \033[33mWARN\033[0m  {name}: {detail}")
    self.warnings.append(name)

  def skip(self, name: str, missing: str) -> None:
    """Record a check that could not run because the export predates the field.

    Skips are surfaced, never silent: an unchecked gain table is exactly the
    situation this script exists to prevent, so it must not look like a pass.
    """
    print(f"  \033[33mSKIP\033[0m  {name} (metadata lacks '{missing}')")
    self.warnings.append(f"unchecked: {name}")


def _close(a, b, atol: float) -> tuple[bool, str]:
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)
  if a.shape != b.shape:
    return False, f"length {a.shape} vs {b.shape}"
  d = np.abs(a - b)
  if np.all(d <= atol):
    return True, ""
  i = int(np.argmax(d))
  return False, f"max diff {d[i]:.4g} at index {i} ({a[i]:.4g} vs {b[i]:.4g})"


# Keys that are free-form text and must never be coerced to numbers, even if
# they happen to look numeric (a config hash of all digits is legal).
_STRING_KEYS = frozenset({"run_path", "task", "deploy_config_hash"})


def _parse_metadata_value(key: str, raw: str):
  """Decode one ONNX metadata value.

  ``attach_metadata_to_onnx`` writes lists as comma-separated text
  (``list_to_csv_str``, 3 decimals) rather than JSON, so parsing has to
  round-trip that format -- including for policies exported before this script
  existed, which is exactly the case it most needs to handle.
  """
  if key in _STRING_KEYS:
    return raw
  parts = raw.split(",")
  try:
    values = [float(p) for p in parts]
    return values if len(values) > 1 else values[0]
  except ValueError:
    return parts if len(parts) > 1 else raw


def load_onnx_metadata(path: str) -> dict:
  model = onnx.load(path)
  meta: dict = {}
  for p in model.metadata_props:
    try:
      meta[p.key] = json.loads(p.value)
      if not isinstance(meta[p.key], (list, dict)):
        raise ValueError  # a bare scalar may still be CSV; fall through
    except (json.JSONDecodeError, TypeError, ValueError):
      meta[p.key] = _parse_metadata_value(p.key, p.value)
  # Graph IO is the ground truth for the interface, independent of metadata.
  g = model.graph
  meta["_onnx_input_shape"] = [d.dim_value for d in g.input[0].type.tensor_type.shape.dim]
  meta["_onnx_output_shape"] = [
    d.dim_value for d in g.output[0].type.tensor_type.shape.dim
  ]
  return meta


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--onnx", required=True)
  ap.add_argument("--deploy-yaml", required=True)
  ap.add_argument(
    "--expect-obs-dim",
    type=int,
    default=71,
    help="Deployed actor observation width PER FRAME. With history_length H in "
    "the policy metadata the ONNX input is this times H.",
  )
  ap.add_argument(
    "--expect-action-dim", type=int, default=13, help="Deployed action width."
  )
  args = ap.parse_args()

  meta = load_onnx_metadata(args.onnx)
  with open(args.deploy_yaml) as f:
    dep = yaml.safe_load(f)

  print(f"\npolicy : {args.onnx}")
  print(f"config : {args.deploy_yaml}")
  print(f"run    : {meta.get('run_path', '<unknown>')}")
  print(f"hash   : {meta.get('deploy_config_hash', '<absent>')}\n")

  r = Report()

  # Frames of actor history per term (1 for every policy exported before
  # history existed, whose metadata has no such key).
  hist = int(meta.get("history_length", 1))
  input_dim = args.expect_obs_dim * hist

  # -- 1. ONNX interface. Checked from the graph, not from metadata, so it
  #       holds even for a policy exported before this script existed.
  print("interface")
  r.check(
    meta["_onnx_input_shape"] == [1, input_dim],
    f"ONNX input is [1, {input_dim}]",
    str(meta["_onnx_input_shape"]),
  )
  r.check(
    meta["_onnx_output_shape"] == [1, args.expect_action_dim],
    f"ONNX output is [1, {args.expect_action_dim}]",
    str(meta["_onnx_output_shape"]),
  )

  if "joint_names" not in meta:
    print(
      "\n\033[33mThis policy carries no training metadata\033[0m (exported before "
      "check_deploy_consistency existed). Only the interface could be verified; "
      "gains, action scale and joint order are UNCHECKED.\n"
    )
    return 2 if not r.failures else 1

  # -- 2. Joint order. Every array below is positional, so a mismatch here
  #       silently permutes gains and targets across the whole robot.
  print("\njoint order")
  n_joints = len(meta["joint_names"])
  r.check(
    len(dep["default_joint_pos"]) == n_joints,
    f"deploy.yaml has {n_joints} joints",
    f"{len(dep['default_joint_pos'])}",
  )
  ok, detail = _close(meta["default_joint_pos"], dep["default_joint_pos"], ATOL_POS)
  r.check(
    ok,
    "default_joint_pos agrees (implies identical joint ORDER)",
    detail + "  -- if lengths match but values do not, the two sides order joints "
    "differently",
  )

  # -- 3. PD gains. The closed loop the policy was trained against.
  print("\nPD gains")
  ok, detail = _close(meta["joint_stiffness"], dep["stiffness"], ATOL_GAIN)
  r.check(ok, "stiffness", detail)
  ok, detail = _close(meta["joint_damping"], dep["damping"], ATOL_GAIN)
  r.check(ok, "damping", detail)

  # -- 4. Action contract.
  print("\nactions")
  act = dep["actions"]["JointPositionAction"]
  n_act = meta.get("action_dim", len(meta.get("action_scale", [])))
  r.check(
    n_act == len(act["scale"]) == args.expect_action_dim,
    f"action_dim == {args.expect_action_dim}",
    f"onnx meta {n_act}, deploy {len(act['scale'])}",
  )
  ok, detail = _close(meta["action_scale"], act["scale"], ATOL_SCALE)
  r.check(ok, "action scale", detail)
  if "action_offset" in meta:
    ok, detail = _close(meta["action_offset"], act["offset"], ATOL_POS)
    r.check(ok, "action offset", detail)
  else:
    r.skip("action offset", "action_offset")
  # The 13 controlled joints must be the FIRST 13 entries of joint_ids_map,
  # which is what makes deploy.yaml's joint_ids [0..12] correct.
  if "action_joint_names" in meta:
    trained_action_joints = meta["action_joint_names"]
    expected_prefix = meta["joint_names"][: len(trained_action_joints)]
    r.check(
      trained_action_joints == expected_prefix,
      "controlled joints are the first 13 in joint order",
      f"{trained_action_joints} vs {expected_prefix}",
    )
  else:
    r.skip("controlled joint identity", "action_joint_names")
  r.check(
    act["joint_ids"] == list(range(args.expect_action_dim)),
    "deploy joint_ids == [0..12]",
    str(act["joint_ids"]),
  )

  # -- 5. Observation order. The concatenation order IS the network input.
  print("\nobservations")
  # `use_gym_history` is a controller switch that lives in the same mapping as
  # the terms (C++ ObservationManager). It must stay false: true interleaves
  # the frames across terms, which is not the layout the policy was trained on.
  dep_terms = {k: v for k, v in dep["observations"].items() if k != "use_gym_history"}
  if "use_gym_history" in dep["observations"]:
    r.check(
      dep["observations"]["use_gym_history"] is False,
      "use_gym_history is false",
      str(dep["observations"]["use_gym_history"]),
    )
  deploy_obs = [OBS_NAME_MAP.get(k, k) for k in dep_terms]
  trained_obs = meta["observation_names"]
  r.check(
    deploy_obs == trained_obs,
    "observation term order",
    f"\n         deploy : {deploy_obs}\n         trained: {trained_obs}",
  )
  # Per-frame widths: deploy.yaml's scale lists are per frame, and the
  # metadata's observation_dims are per frame too.
  deploy_widths = [len(v["scale"]) for v in dep_terms.values()]
  deploy_hist = [int(v.get("history_length", 1)) for v in dep_terms.values()]
  if hist != 1 or any(h != 1 for h in deploy_hist):
    r.check(
      all(h == hist for h in deploy_hist),
      f"every deploy.yaml term has history_length {hist}",
      f"deploy {dict(zip(dep_terms, deploy_hist, strict=True))}",
    )
  if "observation_dims" in meta:
    ok, detail = _close(meta["observation_dims"], deploy_widths, 0)
    r.check(ok, "observation term widths", detail)
    total = sum(deploy_widths) * hist
    r.check(
      meta["obs_dim"] == input_dim == total == meta["_onnx_input_shape"][-1],
      f"total observation width == {args.expect_obs_dim}"
      + (f" x {hist} frames = {input_dim}" if hist != 1 else ""),
      f"trained {meta['obs_dim']}, deploy {sum(deploy_widths)} x {hist} = {total}, "
      f"ONNX input {meta['_onnx_input_shape'][-1]}",
    )
  else:
    r.skip("observation term widths", "observation_dims")
    r.check(
      sum(deploy_widths) == args.expect_obs_dim,
      f"deploy observation width == {args.expect_obs_dim}",
      str(sum(deploy_widths)),
    )

  # -- 6. Control rate.
  print("\ntiming")
  if "step_dt" in meta:
    r.check(
      abs(float(meta["step_dt"]) - float(dep["step_dt"])) < 1e-9,
      "step_dt",
      f"trained {meta['step_dt']}, deploy {dep['step_dt']}",
    )
  else:
    r.skip("step_dt", "step_dt")

  # -- 7. Command ranges. Deployment must be a SUBSET of what was trained;
  #       commanding outside the trained range is extrapolation, not a config
  #       error, so this is a warning unless it is wildly out.
  print("\ncommand ranges (deploy must be within trained)")
  if "height_command_range" in meta and "base_height" in dep.get("commands", {}):
    tr = meta["height_command_range"]
    dr_ = dep["commands"]["base_height"]["range"]
    r.check(
      dr_[0] >= tr[0] - 1e-6 and dr_[1] <= tr[1] + 1e-6,
      "base_height range within trained range",
      f"deploy {dr_} not inside trained {tr}",
    )
  bv = dep.get("commands", {}).get("base_velocity", {}).get("ranges", {})
  for key, meta_key in (
    ("lin_vel_x", "twist_range_lin_vel_x"),
    ("lin_vel_y", "twist_range_lin_vel_y"),
    ("ang_vel_z", "twist_range_ang_vel_z"),
  ):
    if key in bv and meta_key in meta:
      d0, d1 = bv[key]
      t0, t1 = meta[meta_key]
      if d0 < t0 - 1e-6 or d1 > t1 + 1e-6:
        r.warn(
          f"{key} range",
          f"deploy {[d0, d1]} exceeds trained {[t0, t1]} -- the policy will be "
          "extrapolating at the edges",
        )
      else:
        r.check(True, f"{key} within trained range")

  print(f"\n{'-' * 64}")
  if r.failures:
    print(f"\033[31mMISMATCH\033[0m  {len(r.failures)}/{r.checks} checks failed:")
    for f in r.failures:
      print(f"    - {f}")
    print("\nDO NOT DEPLOY this policy against this config.\n")
    return 1
  msg = f"\033[32mOK\033[0m  all {r.checks} checks passed"
  if r.warnings:
    msg += f" ({len(r.warnings)} warning(s))"
  print(msg + f"\nconfig hash: {meta.get('deploy_config_hash', '<absent>')}\n")
  return 0


if __name__ == "__main__":
  sys.exit(main())
