"""Scripted pick-and-place expert.

The expert is *Markov*: its action depends only on the current true state, never on
an internal phase counter. That matters for DAgger, where the expert must label
states the learner visited, not states the expert would have reached itself.
"""
from __future__ import annotations

import numpy as np

GAIN = 12.0
HOVER = 0.06        # approach height above the object (m)
XY_TOL = 0.012      # how well aligned before descending (m)
Z_TOL = 0.008
GRASP_WIDTH = 0.07  # combined finger opening below which, near the object, we count as holding


def holding(s: np.ndarray) -> bool:
    rel = s[6:9]
    width = s[9] + s[10]
    return bool(np.linalg.norm(rel) < 0.03 and width < GRASP_WIDTH)


def expert_action(s: np.ndarray) -> np.ndarray:
    grip, obj, rel, goal = s[0:3], s[3:6], s[6:9], s[25:28]
    a = np.zeros(4, np.float32)
    if holding(s):
        a[:3] = GAIN * (goal - obj)
        a[3] = -1.0                                   # keep closed
    elif np.linalg.norm(rel[:2]) > XY_TOL:
        target = obj + np.array([0, 0, HOVER])
        a[:3] = GAIN * (target - grip)
        a[3] = 1.0                                    # open
    elif rel[2] < -Z_TOL:                             # gripper above object: descend
        a[:3] = GAIN * (obj - grip)
        a[3] = 1.0
    else:
        a[3] = -1.0                                   # aligned: close
    return np.clip(a, -1, 1)


def phase(s: np.ndarray) -> str:
    """Coarse task phase, used for failure attribution."""
    if holding(s):
        return "transport"
    if np.linalg.norm(s[6:8]) > XY_TOL:
        return "approach"
    if s[8] < -Z_TOL:
        return "descend"
    return "grasp"
