"""Stage-2 experiments. Each writes JSON to results/.

  python scripts/run_stage2.py robust --train-data nominal      # E4 (run once per train-data setting)
  python scripts/run_stage2.py robust --train-data randomized
  python scripts/run_stage2.py rl --reward sparse               # E5 (run once per reward)
  python scripts/run_stage2.py rl --reward shaped
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from relman.data import collect_demos  # noqa: E402
from relman.evaluate import evaluate, summarize  # noqa: E402
from relman.expert import expert_action  # noqa: E402
from relman.rl import IQL, IQLAgent, Replay, finetune_online, rollout_transitions  # noqa: E402
from relman.train import train_bc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
E4_VARS = ["nominal", "noisy_sensor", "noisy_sensor_10mm", "latency", "latency_4"]
MODELS = [("mlp", 0), ("mlp", 3), ("lstm", 0), ("lstm", 3)]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def save(name, obj):
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1)


def exp_robust(a):
    rows = []
    for seed in range(a.seeds):
        demos = collect_demos(a.demos, seed=seed, variation=a.train_data)
        for arch, k in MODELS:
            agent, info = train_bc(demos, arch, seed=seed, steps=a.steps, act_hist=k)
            for var in E4_VARS:
                s = summarize(evaluate(agent, var, a.episodes))
                rows.append({**info, "train_data": a.train_data, "demos": a.demos, "variation": var, **s})
                log(f"robust {a.train_data} seed={seed} {arch}+ah{k} {var:18s} success={s['success'][0]:.2f}")
            save(f"e4_robust_{a.train_data}", {"config": vars(a), "rows": rows})


def exp_rl(a):
    rows = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        def eval_fn(agent, var=a.target):
            s = summarize(evaluate(agent, var, a.episodes))
            return {"success": s["success"][0], "cycle_time": s["cycle_time"][0], "failures": s["failures"]}

        # 1. demonstrations on the nominal setup (what a team would have before deploying)
        noisy = lambda: rng.random() < 0.5  # noqa: E731
        state = {"noisy": False}

        def expert_fn(obs, s, feat):
            act = expert_action(s)
            if state["noisy"]:
                act[:3] += rng.normal(0, 0.3, 3)
            return act

        def expert_reset():
            state["noisy"] = noisy()

        demo_t, _ = rollout_transitions(expert_fn, "nominal", a.demos, 400_000 + 1000 * seed, a.act_hist, expert_reset)

        # 2. BC baseline on the same kind of demos, then its deployment logs in the target setting
        bc, _ = train_bc(collect_demos(a.demos, seed=seed), "mlp", seed=seed, steps=3000, act_hist=a.act_hist)
        bc_eval = eval_fn(bc)
        log(f"rl {a.reward} seed={seed} BC baseline on {a.target}: {bc_eval}")

        def bc_fn(obs, s, feat):
            act = bc.act(obs) + rng.normal(0, 0.1, 4)
            bc.observe(np.clip(act, -1, 1))
            return act

        logs_t, logs_succ = rollout_transitions(bc_fn, a.target, a.rollouts, 500_000 + 1000 * seed, a.act_hist, bc.reset)
        data = demo_t.cat(logs_t)
        log(f"rl {a.reward} seed={seed} offline data: {len(demo_t)} demo + {len(logs_t)} rollout transitions "
            f"(rollout success {np.mean(logs_succ):.2f})")

        # 3. offline IQL
        iql = IQL(data.obs.shape[1], data.obs.mean(0), data.obs.std(0) + 1e-3, seed=seed)
        offline = Replay(data, a.reward)
        g = torch.Generator().manual_seed(seed)
        t0 = time.time()
        for i in range(a.offline_updates):
            iql.update(*offline.sample(256, g))
        off_eval = eval_fn(IQLAgent(iql, a.act_hist))
        log(f"rl {a.reward} seed={seed} offline IQL ({a.offline_updates} updates, {time.time()-t0:.0f}s): {off_eval}")

        # 4. online fine-tuning in the target setting
        curve = finetune_online(iql, offline, a.target, a.online_steps, a.act_hist, seed, eval_fn,
                                eval_every=a.eval_every, log=lambda *m: log(f"rl {a.reward} seed={seed}", *m))
        nominal_after = eval_fn(IQLAgent(iql, a.act_hist), var="nominal")
        rows.append({"seed": seed, "reward": a.reward, "target": a.target, "bc": bc_eval,
                     "bc_rollout_success": float(np.mean(logs_succ)), "offline": off_eval, "online_curve": curve,
                     "nominal_after_finetune": nominal_after,
                     "offline_transitions": {"demo": len(demo_t), "rollouts": len(logs_t)}})
        name = f"e5_rl_{a.reward}" if a.target == "latency" else f"e5_rl_{a.reward}_{a.target}"
        save(name, {"config": vars(a), "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp", choices=["robust", "rl"])
    p.add_argument("--train-data", default="randomized", choices=["nominal", "randomized"])
    p.add_argument("--reward", default="shaped", choices=["sparse", "shaped"])
    p.add_argument("--target", default="latency")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--demos", type=int, default=100)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--act-hist", type=int, default=3)
    p.add_argument("--rollouts", type=int, default=100)
    p.add_argument("--offline-updates", type=int, default=10_000)
    p.add_argument("--online-steps", type=int, default=15_000)
    p.add_argument("--eval-every", type=int, default=3_000)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    {"robust": exp_robust, "rl": exp_rl}[a.exp](a)
    log("done")
