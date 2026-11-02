"""Fetch pick-and-place with controllable task variations.

A *variation* changes one factor the policy did not see during training, so we can
measure how a policy generalises: workspace layout, object physics, sensor noise
and control latency.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict

import gymnasium as gym
import numpy as np

import contextlib  # noqa: E402
import io  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import gymnasium_robotics  # noqa: E402  (silences an unrelated Adroit notice)

gym.register_envs(gymnasium_robotics)

MAX_STEPS = 80  # longer than the default 50 so cycle time has headroom


@dataclass(frozen=True)
class Variation:
    name: str
    obj_range: float = 0.15       # half-width of the object's spawn square (m)
    target_range: float = 0.15    # half-width of the goal's spawn cube (m)
    mass_scale: float = 1.0       # object mass multiplier
    friction_scale: float = 1.0   # object sliding-friction multiplier
    obs_noise: float = 0.0        # std of Gaussian noise added to observations (m)
    latency: int = 0              # actions are applied this many steps late


VARIATIONS: dict[str, Variation] = {
    # what demonstrations are collected on
    "nominal": Variation("nominal", obj_range=0.10, target_range=0.10),
    # held-out variations, never seen in training
    "wide_workspace": Variation("wide_workspace", obj_range=0.15, target_range=0.15),
    "heavy_slippery": Variation("heavy_slippery", obj_range=0.10, target_range=0.10,
                                mass_scale=8.0, friction_scale=0.25),
    "noisy_sensor": Variation("noisy_sensor", obj_range=0.10, target_range=0.10, obs_noise=0.005),
    "latency": Variation("latency", obj_range=0.10, target_range=0.10, latency=2),
}


class VariationWrapper(gym.Wrapper):
    """Applies a Variation to FetchPickAndPlace and flattens observations.

    `info["state"]` always carries the *true* (noise-free) observation, which the
    scripted expert and the failure classifier use.
    """

    def __init__(self, env: gym.Env, var: Variation):
        super().__init__(env)
        self.var = var
        u = env.unwrapped
        u.obj_range = var.obj_range
        u.target_range = var.target_range
        m = u.model
        bid = m.body("object0").id
        gid = m.geom("object0").id
        m.body_mass[bid] *= var.mass_scale
        m.geom_friction[gid, 0] *= var.friction_scale
        self._queue: deque = deque()
        self._rng = np.random.default_rng(0)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)

    def _flat(self, o: dict, noisy: bool) -> np.ndarray:
        x = flatten(o)
        if noisy and self.var.obs_noise > 0:
            x = x.copy()
            x[:6] += self._rng.normal(0, self.var.obs_noise, 6)  # gripper + object positions
            x[6:9] = x[3:6] - x[0:3]                            # keep relative pos consistent
        return x

    def reset(self, *, seed=None, options=None):
        o, info = self.env.reset(seed=seed, options=options)
        self._rng = np.random.default_rng(seed)
        self._queue = deque([np.zeros(4, np.float32)] * self.var.latency)
        info["state"] = flatten(o)
        return self._flat(o, True), info

    def step(self, action):
        action = np.asarray(action, np.float32)
        if self.var.latency:
            self._queue.append(action)
            action = self._queue.popleft()
        o, r, term, trunc, info = self.env.step(action)
        info["state"] = flatten(o)
        return self._flat(o, True), r, term, trunc, info


# Flattened observation layout (28 dims)
#  0:3 gripper pos | 3:6 object pos | 6:9 object - gripper | 9:11 finger widths
# 11:14 object rot | 14:17 object lin vel | 17:20 object ang vel | 20:23 gripper vel
# 23:25 finger vel | 25:28 goal pos
OBS_DIM = 28
ACT_DIM = 4


def flatten(o: dict) -> np.ndarray:
    return np.concatenate([o["observation"], o["desired_goal"]]).astype(np.float32)


def make_env(variation: str | Variation = "nominal") -> VariationWrapper:
    var = VARIATIONS[variation] if isinstance(variation, str) else variation
    env = gym.make("FetchPickAndPlace-v4", max_episode_steps=MAX_STEPS)
    return VariationWrapper(env, var)


def variation_dict(name: str) -> dict:
    return asdict(VARIATIONS[name])
