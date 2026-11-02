"""Run the stage-1 experiments. Each writes JSON to results/.

  python scripts/run_experiments.py arch     # E1: MLP vs LSTM vs transformer, held-out variations
  python scripts/run_experiments.py scaling  # E2: how many demos does BC need?
  python scripts/run_experiments.py dagger   # E3: HG-DAgger vs more demos at equal expert effort
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from relman.data import Episodes, collect_demos  # noqa: E402
from relman.dagger import collect_interventions  # noqa: E402
from relman.envs import VARIATIONS, variation_dict  # noqa: E402
from relman.evaluate import evaluate, summarize  # noqa: E402
from relman.train import train_bc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def exp_arch(a):
    rows = []
    for seed in range(a.seeds):
        demos = collect_demos(a.demos, seed=seed)
        for arch in ["mlp", "lstm", "transformer"]:
            agent, info = train_bc(demos, arch, seed=seed, steps=a.steps)
            for var in VARIATIONS:
                s = summarize(evaluate(agent, var, a.episodes))
                rows.append({**info, "demos": a.demos, "variation": var, **s})
                log(f"arch seed={seed} {arch:12s} {var:15s} success={s['success'][0]:.2f}")
            save("e1_architectures", {"config": vars(a), "variations": {v: variation_dict(v) for v in VARIATIONS}, "rows": rows})


def exp_scaling(a):
    rows = []
    for seed in range(a.seeds):
        pool = collect_demos(max(a.sizes), seed=100 + seed)
        for n in a.sizes:
            sub = Episodes(pool.obs[:n], pool.acts[:n])
            agent, info = train_bc(sub, "mlp", seed=seed, steps=a.steps)
            for var in ["nominal", "wide_workspace"]:
                s = summarize(evaluate(agent, var, a.episodes))
                si = summarize(evaluate(agent, var, a.episodes, assisted=True))
                rows.append({**info, "demos": n, "variation": var, **s, "assisted": si})
                log(f"scaling seed={seed} n={n:4d} {var:15s} success={s['success'][0]:.2f} iv={si['interventions'][0]:.2f}")
            save("e2_demo_scaling", {"config": vars(a), "rows": rows})


def exp_dagger(a):
    rows = []
    for seed in range(a.seeds):
        base = collect_demos(a.init_demos, seed=200 + seed)
        extra_pool = collect_demos(60, seed=300 + seed)   # for the equal-effort BC baseline
        data = base
        extra_used = 0
        bc_data = base
        for rnd in range(a.rounds + 1):
            agent, info = train_bc(data, "mlp", seed=seed, steps=a.steps)
            bc_agent, _ = train_bc(bc_data, "mlp", seed=seed, steps=a.steps)
            for name, ag, d in [("hg_dagger", agent, data), ("bc_equal_effort", bc_agent, bc_data)]:
                s = summarize(evaluate(ag, "nominal", a.episodes))
                si = summarize(evaluate(ag, "nominal", a.episodes, assisted=True))
                rows.append({"method": name, "seed": seed, "round": rnd, "expert_labelled_steps": d.n_steps(),
                             **s, "assisted": si})
                log(f"dagger seed={seed} round={rnd} {name:16s} steps={d.n_steps():5d} "
                    f"success={s['success'][0]:.2f} iv={si['interventions'][0]:.2f}")
            save("e3_hg_dagger", {"config": vars(a), "rows": rows})
            if rnd == a.rounds:
                break
            new, cinfo = collect_interventions(agent, a.collect, seed0=50_000 + 1000 * rnd + 100 * seed)
            data = data.extend(new)
            # BC baseline gets the same number of expert-labelled steps, as full demos
            while bc_data.n_steps() < data.n_steps() and extra_used < len(extra_pool):
                bc_data = bc_data.extend(Episodes([extra_pool.obs[extra_used]], [extra_pool.acts[extra_used]]))
                extra_used += 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp", choices=["arch", "scaling", "dagger"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--demos", type=int, default=100)
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    p.add_argument("--init-demos", type=int, default=10)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--collect", type=int, default=20)
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    {"arch": exp_arch, "scaling": exp_scaling, "dagger": exp_dagger}[a.exp](a)
    log("done")
