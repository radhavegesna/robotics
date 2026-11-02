"""Demonstration collection and windowed datasets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .envs import MAX_STEPS, make_env
from .expert import expert_action


@dataclass
class Episodes:
    """A list of trajectories. obs are what the policy sees; acts are expert labels;
    execs (optional) are the actions actually commanded, which differ from the
    labels when noise was injected."""
    obs: list[np.ndarray]
    acts: list[np.ndarray]
    execs: list[np.ndarray] | None = None

    def __len__(self):
        return len(self.obs)

    def n_steps(self) -> int:
        return int(sum(len(o) for o in self.obs))

    def extend(self, other: "Episodes") -> "Episodes":
        ex = None
        if self.execs is not None and other.execs is not None:
            ex = self.execs + other.execs
        return Episodes(self.obs + other.obs, self.acts + other.acts, ex)

    def subset(self, n: int) -> "Episodes":
        return Episodes(self.obs[:n], self.acts[:n], None if self.execs is None else self.execs[:n])

    def save(self, path: str):
        np.savez_compressed(path, n=len(self), **{f"o{i}": o for i, o in enumerate(self.obs)},
                            **{f"a{i}": a for i, a in enumerate(self.acts)})

    @staticmethod
    def load(path: str) -> "Episodes":
        z = np.load(path)
        n = int(z["n"])
        return Episodes([z[f"o{i}"] for i in range(n)], [z[f"a{i}"] for i in range(n)])


def collect_demos(n: int, seed: int = 0, variation: str = "nominal",
                  dart_noise: float = 0.3, noisy_frac: float = 0.5) -> Episodes:
    """Roll out the expert. On a fraction of episodes we execute expert + noise
    (DART-style) but store the *clean* expert label, so the data covers small
    recoveries from off-nominal states."""
    env = make_env(variation)
    rng = np.random.default_rng(seed)
    out = Episodes([], [], [])
    for ep in range(n):
        o, info = env.reset(seed=int(seed * 100_000 + ep))
        noisy = rng.random() < noisy_frac
        O, A, E = [], [], []
        for _ in range(MAX_STEPS):
            a = expert_action(info["state"])
            O.append(o)
            A.append(a)
            exec_a = a.copy()
            if noisy:
                exec_a[:3] += rng.normal(0, dart_noise, 3)
            exec_a = np.clip(exec_a, -1, 1)
            E.append(exec_a)
            o, _, term, trunc, info = env.step(exec_a)
            if term or trunc:
                break
        out.obs.append(np.asarray(O, np.float32))
        out.acts.append(np.asarray(A, np.float32))
        out.execs.append(np.asarray(E, np.float32))
    env.close()
    return out


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    @staticmethod
    def fit(eps: Episodes) -> "Normalizer":
        x = np.concatenate(eps.obs)
        return Normalizer(x.mean(0), x.std(0) + 1e-3)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}


def with_action_history(obs: np.ndarray, execs: np.ndarray, k: int) -> np.ndarray:
    """Append the previous k commanded actions to each observation (zeros before t=0).
    Under control latency the state alone is not Markov: the robot also needs to
    know which commands are still in flight."""
    if k == 0:
        return obs
    T = len(obs)
    prev = np.concatenate([np.zeros((k, execs.shape[1]), np.float32), execs[:-1] if T else execs])
    hist = np.stack([prev[k - j - 1: k - j - 1 + T] for j in range(k)], axis=1)  # (T, k, A), most recent first
    return np.concatenate([obs, hist.reshape(T, -1)], axis=1).astype(np.float32)


def augment(eps: Episodes, k: int) -> Episodes:
    if k == 0:
        return eps
    assert eps.execs is not None, "action history needs the commanded actions"
    return Episodes([with_action_history(o, e, k) for o, e in zip(eps.obs, eps.execs)], eps.acts, eps.execs)


def windows(eps: Episodes, history: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (B, history, obs_dim) inputs ending at each step, left-padded by
    repeating the first observation, and (B, act_dim) targets."""
    X, Y = [], []
    for o, a in zip(eps.obs, eps.acts):
        pad = np.repeat(o[:1], history - 1, axis=0)
        op = np.concatenate([pad, o]) if history > 1 else o
        idx = np.arange(len(o))[:, None] + np.arange(history)[None, :]
        X.append(op[idx])
        Y.append(a)
    return torch.as_tensor(np.concatenate(X)), torch.as_tensor(np.concatenate(Y))
