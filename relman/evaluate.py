"""Reliability evaluation harness.

Two modes, run on the same seeds:

* autonomous: the policy acts alone. We report success rate, cycle time (steps
  until the object first reaches the goal) and a failure category for each miss.
* assisted: a supervisor watches for stalls and hands control to the expert for a
  few steps (a stand-in for a human teleoperator). We report interventions per
  episode, the deployment metric the policy ultimately has to drive to zero.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .envs import MAX_STEPS, make_env
from .expert import expert_action, holding

STALL_WINDOW = 10      # steps without progress before the supervisor steps in
STALL_EPS = 0.01       # metres of progress that count as "progress"
TAKEOVER_STEPS = 8     # how long the supervisor drives once it steps in


def potential(s: np.ndarray) -> float:
    """Task progress (lower is better): distance to the next thing that matters."""
    if holding(s):
        return float(np.linalg.norm(s[3:6] - s[25:28]))
    return 1.0 + float(np.linalg.norm(s[6:9]))


def classify_failure(held_ever: bool, held_end: bool, success_ever: bool) -> str:
    if success_ever:
        return "placed_then_drifted"
    if not held_ever:
        return "never_grasped"
    if held_end:
        return "carried_not_placed"
    return "dropped"


@dataclass
class EpisodeResult:
    success: bool
    cycle_time: int | None
    interventions: int
    expert_steps: int
    failure: str | None
    # filled only when recording for DAgger
    takeover_obs: list = field(default_factory=list, repr=False)
    takeover_acts: list = field(default_factory=list, repr=False)


def run_episode(env, agent, seed: int, assisted: bool, record: bool = False) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    agent.reset()
    s = info["state"]
    hist: list[float] = []
    takeover_left = 0
    interventions = expert_steps = 0
    held_ever = success_ever = False
    first_success = None
    res = EpisodeResult(False, None, 0, 0, None)
    for t in range(MAX_STEPS):
        a_pol = agent.act(obs)          # keep the policy's history up to date either way
        if assisted and takeover_left == 0 and len(hist) >= STALL_WINDOW and not info.get("is_success"):
            if hist[-STALL_WINDOW] - min(hist[-STALL_WINDOW:]) < STALL_EPS:
                interventions += 1
                takeover_left = TAKEOVER_STEPS
                hist.clear()
        if takeover_left > 0:
            a = expert_action(s)
            takeover_left -= 1
            expert_steps += 1
            if record:
                res.takeover_obs.append(obs)
                res.takeover_acts.append(a)
        else:
            a = a_pol
        obs, _, term, trunc, info = env.step(a)
        s = info["state"]
        hist.append(potential(s))
        held_ever |= holding(s)
        if info["is_success"]:
            success_ever = True
            first_success = first_success or t + 1
        if term or trunc:
            break
    res.success = bool(info["is_success"])
    res.cycle_time = first_success if res.success else None
    res.interventions = interventions
    res.expert_steps = expert_steps
    res.failure = None if res.success else classify_failure(held_ever, holding(s), success_ever)
    return res


def evaluate(agent, variation: str, n_episodes: int, seed0: int = 10_000, assisted: bool = False) -> list[EpisodeResult]:
    env = make_env(variation)
    out = [run_episode(env, agent, seed0 + i, assisted) for i in range(n_episodes)]
    env.close()
    return out


def bootstrap_ci(x, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(results: list[EpisodeResult]) -> dict:
    succ = [r.success for r in results]
    ct = [r.cycle_time for r in results if r.success]
    iv = [r.interventions for r in results]
    fails: dict[str, int] = {}
    for r in results:
        if r.failure:
            fails[r.failure] = fails.get(r.failure, 0) + 1
    return {"n": len(results), "success": bootstrap_ci(succ), "cycle_time": bootstrap_ci(ct),
            "interventions": bootstrap_ci(iv), "failures": fails}


class ExpertAgent:
    """The scripted expert behind the Agent interface (privileged: reads true state)."""

    def __init__(self, env_getter):
        self._get = env_getter

    def reset(self):
        pass

    def act(self, obs):
        return expert_action(self._get())


def timed(fn, *a, **k):
    t0 = time.time()
    r = fn(*a, **k)
    return r, time.time() - t0
