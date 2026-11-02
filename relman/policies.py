"""Behaviour-cloning policies: MLP (no memory), LSTM and causal transformer over a
short observation history. All share one interface so the evaluation harness can
treat them identically."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .envs import ACT_DIM, OBS_DIM


class MLPPolicy(nn.Module):
    history = 1

    def __init__(self, hidden: int = 256, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, ACT_DIM), nn.Tanh())

    def forward(self, x):            # x: (B, H, D)
        return self.net(x[:, -1])


class LSTMPolicy(nn.Module):
    def __init__(self, history: int = 8, hidden: int = 128, obs_dim: int = OBS_DIM):
        super().__init__()
        self.history = history
        self.inp = nn.Linear(obs_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, ACT_DIM), nn.Tanh())

    def forward(self, x):
        h, _ = self.lstm(torch.relu(self.inp(x)))
        return self.head(h[:, -1])


class TransformerPolicy(nn.Module):
    def __init__(self, history: int = 8, d: int = 64, layers: int = 2, heads: int = 4, obs_dim: int = OBS_DIM):
        super().__init__()
        self.history = history
        self.inp = nn.Linear(obs_dim, d)
        self.pos = nn.Parameter(torch.zeros(1, history, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ACT_DIM), nn.Tanh())
        mask = torch.triu(torch.full((history, history), float("-inf")), 1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        h = self.inp(x) + self.pos[:, -x.shape[1]:]
        h = self.enc(h, mask=self.mask[: x.shape[1], : x.shape[1]])
        return self.head(h[:, -1])


ARCHS = {"mlp": MLPPolicy, "lstm": LSTMPolicy, "transformer": TransformerPolicy}


class Agent:
    """Stateful wrapper: keeps the observation history (and, optionally, the last
    `act_hist` commanded actions) and normalises inputs."""

    def __init__(self, model: nn.Module, norm, act_hist: int = 0):
        self.model = model.eval()
        self.norm = norm
        self.act_hist = act_hist
        self.buf: deque = deque(maxlen=model.history)
        self.prev: deque = deque(maxlen=max(act_hist, 1))
        self._last = None

    def reset(self):
        self.buf.clear()
        self.prev = deque([np.zeros(ACT_DIM, np.float32)] * self.act_hist, maxlen=max(self.act_hist, 1))
        self._last = None

    def observe(self, action: np.ndarray):
        """Tell the agent which action was actually commanded (it may have been the supervisor's)."""
        self._last = np.asarray(action, np.float32)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.act_hist:
            if self._last is not None:
                self.prev.appendleft(self._last)   # most recent first, matching with_action_history
            obs = np.concatenate([obs, np.concatenate(list(self.prev))]).astype(np.float32)
        out = self._act(obs)
        self._last = out
        return out

    def _act(self, obs: np.ndarray) -> np.ndarray:
        if not self.buf:
            for _ in range(self.model.history):
                self.buf.append(obs)
        else:
            self.buf.append(obs)
        x = torch.as_tensor(np.stack(self.buf))[None]
        return self.model(self.norm(x))[0].numpy()
