"""Export a DUET checkpoint to policy.onnx with the full deployment contract.

Unlike ``export_checkpoint_onnx.py`` this writes the training-side gains, action
scale, joint order, observation layout and a config hash into the ONNX metadata,
then (unless told not to) runs ``check_deploy_consistency.py`` against the
deploy.yaml the policy is destined for. Nothing leaves this script unverified.

Usage:
  python scripts/export_duet_onnx.py \\
      --task Unitree-G1-23Dof-Duet-Flat \\
      --checkpoint logs/DUET_G1_23dof/<run>/model_12500.pt \\
      --out-dir exported/duet_12500 \\
      --deploy-yaml deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (populate registry)
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.duet.rl.export_metadata import build_deploy_metadata

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DEPLOY_YAML = os.path.join(
  _REPO, "deploy/robots/g1_23dof/config/policy/velocity/v0/params/deploy.yaml"
)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", default="Unitree-G1-23Dof-Duet-Flat")
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--filename", default="policy.onnx")
  ap.add_argument(
    "--deploy-yaml",
    default=_DEFAULT_DEPLOY_YAML,
    help="deploy.yaml to verify against. Pass 'none' to skip verification.",
  )
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # Export from the TRAINING config, not the play config. The ONNX weights come
  # from the checkpoint either way, but the play config clears the curriculum,
  # and the metadata needs those staged ranges to report what the policy was
  # actually trained on by the end of the run rather than at iteration 0.
  # The env is only built and read, never stepped, so pushes and corruption in
  # this config are irrelevant.
  env_cfg = load_env_cfg(args.task, play=False)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=device)

  os.makedirs(args.out_dir, exist_ok=True)
  runner.export_policy_to_onnx(args.out_dir, filename=args.filename)
  onnx_path = os.path.join(args.out_dir, args.filename)

  metadata = build_deploy_metadata(env.unwrapped, run_path=args.checkpoint)
  attach_metadata_to_onnx(onnx_path, metadata)
  env.close()

  print(f"\n[OK] exported {args.checkpoint} -> {onnx_path}")
  print(f"     obs_dim={metadata['obs_dim']} action_dim={metadata['action_dim']} "
        f"hash={metadata['deploy_config_hash']}")

  # Write the contract alongside the ONNX in readable form, so a mismatch can
  # be diffed by eye without an ONNX reader.
  side_car = os.path.join(args.out_dir, "deploy_contract.json")
  with open(side_car, "w") as f:
    json.dump(metadata, f, indent=2)
  print(f"     contract -> {side_car}")

  if args.deploy_yaml.lower() == "none":
    print("\n[WARN] deploy.yaml verification SKIPPED by request.")
    return 0

  print("\n--- verifying against deploy.yaml ---")
  return subprocess.call(
    [
      sys.executable,
      os.path.join(_REPO, "scripts", "check_deploy_consistency.py"),
      "--onnx", onnx_path,
      "--deploy-yaml", args.deploy_yaml,
      "--expect-obs-dim", str(metadata["obs_dim"]),
      "--expect-action-dim", str(metadata["action_dim"]),
    ]
  )


if __name__ == "__main__":
  sys.exit(main())
