import sys
import random

import numpy as np

# robot_env parses process arguments at import time.
_original_argv = sys.argv
sys.argv = [_original_argv[0]]
import robust_gymnasium as gym
from robust_gymnasium.envs import robot_env
sys.argv = _original_argv


def _step_with_noise(monkeypatch, noise_factor, noise_type="shift", noise_shift=0.0, noise_sigma=0.0):
    monkeypatch.setattr(robot_env.args, "noise_factor", noise_factor)
    monkeypatch.setattr(robot_env.args, "noise_type", noise_type)
    monkeypatch.setattr(robot_env.args, "noise_shift", noise_shift)
    monkeypatch.setattr(robot_env.args, "noise_sigma", noise_sigma)

    env = gym.make("FetchReach-v3")
    try:
        env.reset(seed=123)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        robust_input = {"action": action, "robust_config": robot_env.args}
        observation, _, _, _, _ = env.step(robust_input)
        unmodified_observation = env.unwrapped._get_obs()["observation"]
        return observation, unmodified_observation
    finally:
        env.close()


def test_state_shift_changes_returned_observation(monkeypatch):
    clean, _ = _step_with_noise(monkeypatch, "none")
    shifted, _ = _step_with_noise(monkeypatch, "state", noise_shift=1.25)

    assert not np.array_equal(shifted, clean)
    np.testing.assert_allclose(shifted, clean + 1.25)


def test_state_gaussian_noise_changes_returned_observation(monkeypatch):
    clean, _ = _step_with_noise(monkeypatch, "none")
    random.seed(23)
    noisy, _ = _step_with_noise(monkeypatch, "state", noise_type="gauss", noise_sigma=100.0)

    assert not np.array_equal(noisy, clean)


def test_state_noise_disabled_returns_clean_observation(monkeypatch):
    disabled, unmodified = _step_with_noise(monkeypatch, "none", noise_shift=1000.0)

    np.testing.assert_array_equal(disabled, unmodified)
