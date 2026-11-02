"""Offline -> online RL: Implicit Q-Learning (IQL, Kostrikov et al. 2021).

IQL never queries Q at actions outside the dataset during offline training: it
fits V as an upper expectile of Q, then extracts a policy by advantage-weighted
regression. That makes it a natural bridge from demonstrations to online
fine-tuning, which here keeps the offline data in the replay buffer and samples
it 50/50 with fresh experience (as in RLPD).

Rewards are computed from the true simulator state, so the same transitions can
be relabelled with either reward:

* sparse: 0 when the object is within 5 cm of the goal, otherwise -1 (Fetch default)
* shaped: sparse + potential-based shaping  gamma*Phi(s') - Phi(s),  Phi = -SCALE*progress
  (Ng et al. 1999: potential-based shaping does not change the optimal policy)
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .envs import ACT_DIM, MAX_STEPS, OBS_DIM, make_env
from .evaluate import potential

GAMMA = 0.98
SHAPING_SCALE = 10.0


def sparse_reward(s_next: np.ndarray) -> float:
    return 0.0 if np.linalg.norm(s_next[3:6] - s_next[25:28]) < 0.05 else -1.0


def shaped_reward(s: np.ndarray, s_next: np.ndarray) -> float:
    return sparse_reward(s_next) + SHAPING_SCALE * (potential(s) - GAMMA * potential(s_next))


# --------------------------------------------------------------------------- buffer
@dataclass
class Transitions:
    obs: np.ndarray       # policy inputs (observation + action history)
    act: np.ndarray       # commanded action
    state: np.ndarray     # true state before the step (for reward relabelling)
    next_state: np.ndarray
    next_obs: np.ndarray

    def __len__(self):
        return len(self.obs)

    @staticmethod
    def empty(obs_dim: int) -> "Transitions":
        z = lambda d: np.zeros((0, d), np.float32)  # noqa: E731
        return Transitions(z(obs_dim), z(ACT_DIM), z(OBS_DIM), z(OBS_DIM), z(obs_dim))

    def cat(self, o: "Transitions") -> "Transitions":
        return Transitions(*[np.concatenate([getattr(self, f), getattr(o, f)]) for f in
                             ("obs", "act", "state", "next_state", "next_obs")])

    def rewards(self, kind: str) -> np.ndarray:
        if kind == "sparse":
            return np.array([sparse_reward(s2) for s2 in self.next_state], np.float32)
        return np.array([shaped_reward(s, s2) for s, s2 in zip(self.state, self.next_state)], np.float32)


class Replay:
    """Fixed-size ring buffer of tensors with precomputed rewards."""

    def __init__(self, data: Transitions, reward: str, capacity: int | None = None):
        self.reward = reward
        n = capacity or len(data)
        d = data.obs.shape[1]
        self.o = torch.zeros(n, d)
        self.a = torch.zeros(n, ACT_DIM)
        self.r = torch.zeros(n)
        self.o2 = torch.zeros(n, d)
        self.size = self.ptr = 0
        self.cap = n
        self.add(data)

    def add(self, t: Transitions):
        if len(t) == 0:
            return
        r = t.rewards(self.reward)
        for i in range(len(t)):
            j = self.ptr
            self.o[j] = torch.as_tensor(t.obs[i])
            self.a[j] = torch.as_tensor(t.act[i])
            self.r[j] = float(r[i])
            self.o2[j] = torch.as_tensor(t.next_obs[i])
            self.ptr = (self.ptr + 1) % self.cap
            self.size = min(self.size + 1, self.cap)

    def sample(self, n: int, g: torch.Generator):
        idx = torch.randint(0, self.size, (n,), generator=g)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx]


# --------------------------------------------------------------------------- IQL
def mlp(i, o, h=128, ln=True, out_act=None):
    layers = [nn.Linear(i, h)] + ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(),
              nn.Linear(h, h)] + ([nn.LayerNorm(h)] if ln else []) + [nn.ReLU(), nn.Linear(h, o)]
    if out_act:
        layers.append(out_act)
    return nn.Sequential(*layers)


class IQL:
    def __init__(self, obs_dim: int, mean: np.ndarray, std: np.ndarray, expectile: float = 0.7,
                 beta: float = 3.0, lr: float = 3e-4, tau: float = 0.005, seed: int = 0):
        torch.manual_seed(seed)
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.q1, self.q2 = mlp(obs_dim + ACT_DIM, 1), mlp(obs_dim + ACT_DIM, 1)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.v = mlp(obs_dim, 1)
        self.pi = mlp(obs_dim, ACT_DIM, ln=False, out_act=nn.Tanh())
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=lr)
        self.expectile, self.beta, self.tau = expectile, beta, tau

    def n(self, o):
        return (o - self.mean) / self.std

    def update(self, o, a, r, o2) -> dict:
        o, o2 = self.n(o), self.n(o2)
        oa = torch.cat([o, a], 1)
        with torch.no_grad():
            q_t = torch.min(self.q1_t(oa), self.q2_t(oa)).squeeze(1)
            target = r + GAMMA * self.v(o2).squeeze(1)       # time-limit truncation only: always bootstrap
        # V: expectile regression towards min target Q
        v = self.v(o).squeeze(1)
        diff = q_t - v
        w = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        loss_v = (w * diff.pow(2)).mean()
        self.opt_v.zero_grad()
        loss_v.backward()
        self.opt_v.step()
        # Q: TD towards r + gamma V(s')
        loss_q = F.mse_loss(self.q1(oa).squeeze(1), target) + F.mse_loss(self.q2(oa).squeeze(1), target)
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()
        # policy: advantage-weighted regression on dataset actions
        with torch.no_grad():
            adv = q_t - self.v(o).squeeze(1)
            wts = torch.clamp(torch.exp(self.beta * adv), max=100.0)
        loss_pi = (wts * (self.pi(o) - a).pow(2).sum(1)).mean()
        self.opt_pi.zero_grad()
        loss_pi.backward()
        self.opt_pi.step()
        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
        return {"v": loss_v.item(), "q": loss_q.item(), "pi": loss_pi.item()}

    @torch.no_grad()
    def act(self, obs_aug: np.ndarray, noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
        a = self.pi(self.n(torch.as_tensor(obs_aug)[None]))[0].numpy()
        if noise > 0:
            a = a + (rng or np.random.default_rng()).normal(0, noise, ACT_DIM)
        return np.clip(a, -1, 1).astype(np.float32)


class IQLAgent:
    """IQL policy behind the evaluation harness's Agent interface."""

    def __init__(self, iql: IQL, act_hist: int):
        self.iql, self.k = iql, act_hist
        self.reset()

    def reset(self):
        self.prev = [np.zeros(ACT_DIM, np.float32) for _ in range(self.k)]
        self._last = None

    def observe(self, a):
        self._last = np.asarray(a, np.float32)

    def features(self, obs):
        if self._last is not None and self.k:
            self.prev = [self._last] + self.prev[:-1]
        self._last = None
        return np.concatenate([obs] + self.prev).astype(np.float32) if self.k else obs

    def act(self, obs, noise: float = 0.0, rng=None):
        a = self.iql.act(self.features(obs), noise, rng)
        self._last = a
        return a


# --------------------------------------------------------------------------- data
def rollout_transitions(policy_fn, variation: str, n_episodes: int, seed0: int, act_hist: int,
                        reset_fn=None) -> tuple[Transitions, list[bool]]:
    """Run `policy_fn(obs, true_state) -> action` and record transitions with
    action-history features. Returns transitions and per-episode success."""
    env = make_env(variation)
    out = {k: [] for k in ("obs", "act", "state", "next_state", "next_obs")}
    succ = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed0 + ep)
        if reset_fn:
            reset_fn()
        prev = [np.zeros(ACT_DIM, np.float32) for _ in range(act_hist)]
        feat = np.concatenate([obs] + prev).astype(np.float32)
        s = info["state"]
        for _ in range(MAX_STEPS):
            a = np.clip(np.asarray(policy_fn(obs, s, feat), np.float32), -1, 1)
            obs, _, term, trunc, info = env.step(a)
            if act_hist:
                prev = [a] + prev[:-1]
            feat2 = np.concatenate([obs] + prev).astype(np.float32)
            out["obs"].append(feat)
            out["act"].append(a)
            out["state"].append(s)
            out["next_state"].append(info["state"])
            out["next_obs"].append(feat2)
            feat, s = feat2, info["state"]
            if term or trunc:
                break
        succ.append(bool(info["is_success"]))
    env.close()
    return Transitions(*[np.asarray(out[k], np.float32) for k in
                         ("obs", "act", "state", "next_state", "next_obs")]), succ


def finetune_online(iql: IQL, offline: Replay, variation: str, env_steps: int, act_hist: int,
                    seed: int, eval_fn, eval_every: int, noise: float = 0.2, batch: int = 256,
                    log=print) -> list[dict]:
    """Online IQL fine-tuning; each update samples half offline, half online data."""
    online = Replay(Transitions.empty(offline.o.shape[1]), offline.reward, capacity=env_steps)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = IQLAgent(iql, act_hist)
    env = make_env(variation)
    curve = [{"env_steps": 0, **eval_fn(agent)}]
    log(f"    online step 0: {curve[-1]}")
    steps = ep = 0
    t0 = time.time()
    while steps < env_steps:
        obs, info = env.reset(seed=900_000 + 1000 * seed + ep)
        ep += 1
        agent.reset()
        feat = agent.features(obs)
        s = info["state"]
        for _ in range(MAX_STEPS):
            a = iql.act(feat, noise, rng)
            agent._last = a
            obs, _, term, trunc, info = env.step(a)
            feat2 = agent.features(obs)
            online.add(Transitions(feat[None], a[None], s[None], info["state"][None], feat2[None]))
            feat, s = feat2, info["state"]
            steps += 1
            h = batch // 2
            bo = offline.sample(h, g)
            bn = online.sample(h, g)
            iql.update(*[torch.cat([x, y]) for x, y in zip(bo, bn)])
            if steps % eval_every == 0:
                curve.append({"env_steps": steps, "minutes": round((time.time() - t0) / 60, 1), **eval_fn(agent)})
                log(f"    online step {steps}: {curve[-1]}")
            if term or trunc or steps >= env_steps:
                break
    env.close()
    return curve
