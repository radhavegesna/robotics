import numpy as np

from relman.data import collect_demos, windows
from relman.envs import OBS_DIM, VARIATIONS, make_env
from relman.evaluate import classify_failure, evaluate, summarize
from relman.expert import expert_action
from relman.train import train_bc


def test_expert_solves_nominal():
    env = make_env("nominal")
    ok = 0
    for ep in range(5):
        _, info = env.reset(seed=ep)
        for _ in range(80):
            _, _, _, _, info = env.step(expert_action(info["state"]))
        ok += info["is_success"]
    assert ok == 5


def test_variations_build_and_observe():
    for v in VARIATIONS:
        o, info = make_env(v).reset(seed=0)
        assert o.shape == (OBS_DIM,) and info["state"].shape == (OBS_DIM,)


def test_windows_shapes():
    d = collect_demos(2, seed=0)
    X, Y = windows(d, 8)
    assert X.shape == (d.n_steps(), 8, OBS_DIM) and Y.shape == (d.n_steps(), 4)
    # left padding repeats the first observation
    assert np.allclose(X[0, 0].numpy(), X[0, -1].numpy())


def test_train_and_evaluate_pipeline():
    agent, info = train_bc(collect_demos(3, seed=0), "lstm", steps=20)
    s = summarize(evaluate(agent, "nominal", 2, assisted=True))
    assert s["n"] == 2 and info["params"] > 0


def test_failure_taxonomy():
    assert classify_failure(False, False, False) == "never_grasped"
    assert classify_failure(True, False, False) == "dropped"
    assert classify_failure(True, True, False) == "carried_not_placed"
    assert classify_failure(True, True, True) == "placed_then_drifted"
