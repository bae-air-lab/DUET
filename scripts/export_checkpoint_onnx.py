"""Export a specific rsl_rl checkpoint (.pt) to policy.onnx.

Usage: export_checkpoint_onnx.py <task_id> <checkpoint.pt> <out_dir> [filename]
"""

import sys
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (populate registry)
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main():
    task_id = sys.argv[1]
    ckpt = sys.argv[2]
    out_dir = sys.argv[3]
    filename = sys.argv[4] if len(sys.argv) > 4 else "policy.onnx"

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=device)

    runner.export_policy_to_onnx(out_dir, filename=filename)
    env.close()
    print(f"[OK] exported {ckpt} -> {out_dir}/{filename}")


if __name__ == "__main__":
    main()
