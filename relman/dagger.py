"""HG-DAgger: the supervisor's takeover segments become new training data.

Each round: deploy the current policy with the supervisor, keep only the
(state, expert action) pairs from takeovers, add them to the dataset, retrain.
This is the continual-learning loop that should drive interventions to zero.
"""
from __future__ import annotations

import numpy as np

from .data import Episodes
from .envs import make_env
from .evaluate import run_episode


def collect_interventions(agent, n_episodes: int, seed0: int, variation: str = "nominal") -> tuple[Episodes, dict]:
    env = make_env(variation)
    new = Episodes([], [])
    iv = []
    for i in range(n_episodes):
        r = run_episode(env, agent, seed0 + i, assisted=True, record=True)
        iv.append(r.interventions)
        if r.takeover_obs:
            new.obs.append(np.asarray(r.takeover_obs, np.float32))
            new.acts.append(np.asarray(r.takeover_acts, np.float32))
    env.close()
    return new, {"interventions_during_collection": float(np.mean(iv)), "labelled_steps": new.n_steps()}
