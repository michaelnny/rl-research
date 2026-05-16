"""Smoke test that verifies the install:
- PyTorch sees CUDA + the right device.
- A few Gymnasium envs (classic control, Box2D, MuJoCo, Atari) step without error.
- dm-control loads a control suite task.
- TensorBoard's SummaryWriter writes an event.

Run with: `uv run python tests/smoke_test.py`
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def check_torch() -> None:
    import torch

    print(f"torch:        {torch.__version__}")
    print(f"  cuda avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device:     {torch.cuda.get_device_name(0)}")
        print(f"  capability: {torch.cuda.get_device_capability(0)}")
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x.T).sum().item()
        print(f"  matmul ok:  sum={y:.2f}")
    else:
        raise SystemExit("CUDA not available — check driver / wheel selection")


def check_gymnasium() -> None:
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)

    cases = [
        ("CartPole-v1", "classic-control"),
        ("LunarLander-v3", "box2d"),
        ("HalfCheetah-v5", "mujoco"),
        ("ALE/Pong-v5", "atari"),
    ]
    for env_id, label in cases:
        env = gym.make(env_id)
        obs, _ = env.reset(seed=0)
        action = env.action_space.sample()
        env.step(action)
        env.close()
        print(f"gymnasium:    {env_id:<22} ok ({label}, obs shape={getattr(obs, 'shape', None)})")


def check_dm_control() -> None:
    from dm_control import suite

    env = suite.load(domain_name="cartpole", task_name="swingup")
    ts = env.reset()
    action_spec = env.action_spec()
    import numpy as np

    a = np.zeros(action_spec.shape, dtype=action_spec.dtype)
    env.step(a)
    print(f"dm_control:   cartpole.swingup ok (obs keys={list(ts.observation.keys())})")


def check_tensorboard() -> None:
    from torch.utils.tensorboard import SummaryWriter

    with tempfile.TemporaryDirectory() as d:
        w = SummaryWriter(log_dir=d)
        w.add_scalar("smoke/value", 1.23, 0)
        w.close()
        files = list(Path(d).glob("events.out.tfevents.*"))
        assert files, "no tfevents file written"
        print(f"tensorboard:  wrote {files[0].name}")


if __name__ == "__main__":
    check_torch()
    check_gymnasium()
    check_dm_control()
    check_tensorboard()
    print("\nAll smoke checks passed.")
