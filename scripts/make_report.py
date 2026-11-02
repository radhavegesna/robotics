"""Turn results/*.json into figures and a markdown report (results/REPORT.md)."""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
VARS = ["nominal", "wide_workspace", "heavy_slippery", "noisy_sensor", "latency"]
COLORS = {"mlp": "#4C72B0", "lstm": "#DD8452", "transformer": "#55A868",
          "hg_dagger": "#C44E52", "bc_equal_effort": "#8172B3"}


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pooled_success(rows):
    k = sum(round(r["success"][0] * r["n"]) for r in rows)
    n = sum(r["n"] for r in rows)
    lo, hi = wilson(k, n)
    return k / n, lo, hi, n


def fmt(m, lo, hi, pct=True):
    if pct:
        return f"{100*m:.0f}% [{100*lo:.0f}–{100*hi:.0f}]"
    return f"{m:.2f}"


def nanmean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(x)) if x else float("nan")


lines = ["# Stage 1 results", "",
         "All numbers are produced by `scripts/run_experiments.py` on CPU. Success rates pool episodes "
         "across training seeds and show 95% Wilson intervals in brackets. Held-out variations were never "
         "seen during training.", ""]

expert = {r["variation"]: r for r in (load("expert_reference") or {"rows": []})["rows"]}

# ---------------- E1
e1 = load("e1_architectures")
if e1:
    cfg = e1["config"]
    g = defaultdict(list)
    for r in e1["rows"]:
        g[(r["arch"], r["variation"])].append(r)
    archs = [a for a in ["mlp", "lstm", "transformer"] if any(k[0] == a for k in g)]
    seeds = sorted({r["seed"] for r in e1["rows"]})
    lines += ["## E1 · Architecture vs held-out task variations", "",
              f"{cfg['demos']} demos per seed, {cfg['steps']} gradient steps, {cfg['episodes']} evaluation "
              f"episodes per variation per seed, seeds completed: {len(seeds)}.", "",
              "| Variation | Expert | " + " | ".join(a.upper() if a != "transformer" else "Transformer" for a in archs) + " |",
              "|---|---|" + "---|" * len(archs)]
    for v in VARS:
        ex = expert.get(v)
        exs = f"{100*ex['success'][0]:.0f}%" if ex else "–"
        cells = []
        for a in archs:
            if g.get((a, v)):
                m, lo, hi, _ = pooled_success(g[(a, v)])
                cells.append(fmt(m, lo, hi))
            else:
                cells.append("–")
        lines.append(f"| {v} | {exs} | " + " | ".join(cells) + " |")
    lines += ["", "Mean cycle time on successful episodes (steps, lower is faster):", "",
              "| Variation | Expert | " + " | ".join(archs) + " |", "|---|---|" + "---|" * len(archs)]
    for v in VARS:
        ex = expert.get(v)
        cells = [f"{nanmean([r['cycle_time'][0] for r in g.get((a, v), [])]):.1f}" for a in archs]
        lines.append(f"| {v} | {ex['cycle_time'][0]:.1f} | " if ex else f"| {v} | – | ")
        lines[-1] += " | ".join(cells) + " |"
    # failures
    lines += ["", "Failure categories (all seeds, all held-out variations):", "",
              "| Policy | never_grasped | dropped | carried_not_placed | placed_then_drifted |", "|---|---|---|---|---|"]
    for a in archs:
        tot = defaultdict(int)
        for v in VARS[1:]:
            for r in g.get((a, v), []):
                for k, c in r["failures"].items():
                    tot[k] += c
        lines.append(f"| {a} | {tot['never_grasped']} | {tot['dropped']} | {tot['carried_not_placed']} | {tot['placed_then_drifted']} |")
    # figure
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(archs)
    for i, a in enumerate(archs):
        ms, errs = [], [[], []]
        for v in VARS:
            if g.get((a, v)):
                m, lo, hi, _ = pooled_success(g[(a, v)])
            else:
                m = lo = hi = float("nan")
            ms.append(m)
            errs[0].append(m - lo)
            errs[1].append(hi - m)
        ax.bar(np.arange(len(VARS)) + (i - (len(archs) - 1) / 2) * w, ms, w, yerr=errs, capsize=2,
               label=a, color=COLORS[a])
    for j, v in enumerate(VARS):
        if v in expert:
            ax.hlines(expert[v]["success"][0], j - 0.42, j + 0.42, colors="k", linestyles="--", lw=1,
                      label="expert" if j == 0 else None)
    ax.set_xticks(range(len(VARS)), VARS)
    ax.set_ylabel("success rate")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.set_title("E1: success on training conditions vs held-out variations")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e1_architectures.png"), dpi=140)
    lines += ["", "![E1](figures/e1_architectures.png)", ""]

# ---------------- E2
e2 = load("e2_demo_scaling")
if e2:
    g = defaultdict(list)
    for r in e2["rows"]:
        g[(r["variation"], r["demos"])].append(r)
    sizes = sorted({r["demos"] for r in e2["rows"]})
    lines += ["## E2 · How many demonstrations does BC need? (MLP)", "",
              "| Demos | Nominal success | Nominal interventions/ep | Wide-workspace success | Wide interventions/ep |",
              "|---|---|---|---|---|"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for v, ls in [("nominal", "-"), ("wide_workspace", "--")]:
        sm, iv = [], []
        for n in sizes:
            rows = g.get((v, n), [])
            sm.append(pooled_success(rows)[0] if rows else float("nan"))
            iv.append(nanmean([r["assisted"]["interventions"][0] for r in rows]))
        axs[0].plot(sizes, sm, "o" + ls, label=v, color="#4C72B0")
        axs[1].plot(sizes, iv, "o" + ls, label=v, color="#C44E52")
    for n in sizes:
        cells = []
        for v in ["nominal", "wide_workspace"]:
            rows = g.get((v, n), [])
            if rows:
                m, lo, hi, _ = pooled_success(rows)
                cells += [fmt(m, lo, hi), f"{nanmean([r['assisted']['interventions'][0] for r in rows]):.2f}"]
            else:
                cells += ["–", "–"]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    for ax, t in zip(axs, ["autonomous success", "interventions / episode (assisted)"]):
        ax.set_xscale("log")
        ax.set_xlabel("demonstrations")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e2_demo_scaling.png"), dpi=140)
    lines += ["", "![E2](figures/e2_demo_scaling.png)", ""]

# ---------------- E3
e3 = load("e3_hg_dagger")
if e3:
    g = defaultdict(list)
    for r in e3["rows"]:
        g[(r["method"], r["round"])].append(r)
    rounds = sorted({r["round"] for r in e3["rows"]})
    lines += ["## E3 · HG-DAgger vs more demonstrations at equal expert effort (MLP)", "",
              f"Both start from {e3['config']['init_demos']} demos. Each HG-DAgger round deploys the policy with the "
              f"supervisor for {e3['config']['collect']} episodes and adds only the takeover segments. The baseline "
              "gets the same number of expert-labelled steps as extra full demonstrations.", "",
              "| Round | Method | Expert-labelled steps | Autonomous success | Interventions/ep |", "|---|---|---|---|---|"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for meth in ["hg_dagger", "bc_equal_effort"]:
        xs, sm, iv = [], [], []
        for rd in rounds:
            rows = g.get((meth, rd), [])
            if not rows:
                continue
            m, lo, hi, _ = pooled_success(rows)
            st = nanmean([r["expert_labelled_steps"] for r in rows])
            ivm = nanmean([r["assisted"]["interventions"][0] for r in rows])
            xs.append(st)
            sm.append(m)
            iv.append(ivm)
            lines.append(f"| {rd} | {meth} | {st:.0f} | {fmt(m, lo, hi)} | {ivm:.2f} |")
        axs[0].plot(xs, sm, "o-", label=meth, color=COLORS[meth])
        axs[1].plot(xs, iv, "o-", label=meth, color=COLORS[meth])
    for ax, t in zip(axs, ["autonomous success", "interventions / episode"]):
        ax.set_xlabel("expert-labelled steps")
        ax.set_title(t, fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e3_hg_dagger.png"), dpi=140)
    lines += ["", "![E3](figures/e3_hg_dagger.png)", ""]

# ---------------- E4
E4_VARS = ["nominal", "noisy_sensor", "noisy_sensor_10mm", "latency", "latency_4"]
e4 = [load(f"e4_robust_{t}") for t in ("nominal", "randomized")]
if any(e4):
    g = defaultdict(list)
    for d in e4:
        for r in (d or {"rows": []})["rows"]:
            g[(r["train_data"], r["arch"], r["act_hist"], r["variation"])].append(r)
    cfg = next(d for d in e4 if d)["config"]
    models = [("mlp", 0), ("mlp", 3), ("lstm", 0), ("lstm", 3)]
    lines += ["# Stage 2", "",
              "## E4 · Randomised training and action-history inputs", "",
              f"{cfg['demos']} demos per seed, {cfg['steps']} gradient steps, {cfg['episodes']} evaluation episodes "
              "per variation per seed. `randomized` demos sample a control latency of 0–3 steps and 0–6 mm sensor "
              "noise per episode; `+ah3` policies also see their last 3 commanded actions. `noisy_sensor` (5 mm) and "
              "`latency` (2 steps) lie inside the randomisation range; `noisy_sensor_10mm` and `latency_4` lie outside it.", "",
              "| Train data | Policy | " + " | ".join(E4_VARS) + " |", "|---|---|" + "---|" * len(E4_VARS),
              "| – | *expert (privileged)* | " + " | ".join(
                  f"*{100*expert[v]['success'][0]:.0f}%*" if v in expert else "–" for v in E4_VARS) + " |"]
    for td in ("nominal", "randomized"):
        for arch, k in models:
            cells = []
            for v in E4_VARS:
                rows = g.get((td, arch, k, v), [])
                if rows:
                    m, lo, hi, _ = pooled_success(rows)
                    cells.append(fmt(m, lo, hi))
                else:
                    cells.append("–")
            lines.append(f"| {td} | {arch}{'+ah3' if k else ''} | " + " | ".join(cells) + " |")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    w = 0.2
    for ax, td in zip(axs, ("nominal", "randomized")):
        for i, (arch, k) in enumerate(models):
            ms = []
            for v in E4_VARS:
                rows = g.get((td, arch, k, v), [])
                ms.append(pooled_success(rows)[0] if rows else float("nan"))
            ax.bar(np.arange(len(E4_VARS)) + (i - 1.5) * w, ms, w, label=f"{arch}{'+ah3' if k else ''}",
                   color=["#4C72B0", "#9DB7E0", "#DD8452", "#F0C29E"][i])
        ax.set_xticks(range(len(E4_VARS)), E4_VARS, fontsize=7.5)
        ax.set_title(f"trained on {td} demos", fontsize=10)
        ax.set_ylim(0, 1.05)
    axs[0].set_ylabel("success rate")
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e4_robust.png"), dpi=140)
    lines += ["", "![E4](figures/e4_robust.png)", ""]

# ---------------- E5
e5 = {r: load(f"e5_rl_{r}") for r in ("sparse", "shaped")}
if any(e5.values()):
    cfg = next(d for d in e5.values() if d)["config"]
    lines += ["## E5 · Offline → online RL (IQL) in the 2-step-latency setting", "",
              f"Offline data per seed: {cfg['demos']} expert demos on `nominal` + {cfg['rollouts']} deployment rollouts "
              f"of the BC policy in `{cfg['target']}` (successes and failures). IQL is trained offline for "
              f"{cfg['offline_updates']:,} updates, then fine-tuned online for {cfg['online_steps']:,} environment steps "
              "(half of each batch from the offline data). All policies are MLPs with 3-step action history. "
              f"Success is measured on {cfg['episodes']} evaluation episodes per checkpoint per seed.", "",
              "| Reward | BC | Offline IQL | Online, final | Nominal after fine-tune | Cycle time BC → final |",
              "|---|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for rw, d in e5.items():
        if not d:
            continue
        R = d["rows"]
        n_ep = cfg["episodes"] * len(R)
        def pool(vals):
            k = sum(round(v * cfg["episodes"]) for v in vals)
            lo, hi = wilson(k, n_ep)
            return fmt(k / n_ep, lo, hi)
        lines.append(f"| {rw} (seeds={len(R)}) | {pool([r['bc']['success'] for r in R])} | "
                     f"{pool([r['offline']['success'] for r in R])} | "
                     f"{pool([r['online_curve'][-1]['success'] for r in R])} | "
                     f"{pool([r['nominal_after_finetune']['success'] for r in R])} | "
                     f"{nanmean([r['bc']['cycle_time'] for r in R]):.1f} → "
                     f"{nanmean([r['online_curve'][-1]['cycle_time'] for r in R]):.1f} |")
        steps = [c["env_steps"] for c in R[0]["online_curve"]]
        curves = np.array([[c["success"] for c in r["online_curve"]] for r in R if len(r["online_curve"]) == len(steps)])
        m = curves.mean(0)
        ax.plot(steps, m, "o-", label=f"IQL, {rw} reward", color="#C44E52" if rw == "shaped" else "#8172B3")
        if len(curves) > 1:
            ax.fill_between(steps, curves.min(0), curves.max(0), alpha=0.15,
                            color="#C44E52" if rw == "shaped" else "#8172B3")
        bc = np.mean([r["bc"]["success"] for r in R])
    ax.axhline(bc, ls="--", color="k", lw=1, label="BC (same demos)")
    ax.set_xlabel("online environment steps")
    ax.set_ylabel("success rate (latency = 2)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("E5: offline IQL, then online fine-tuning (band = min–max over seeds)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "e5_rl.png"), dpi=140)
    lines += ["", "![E5](figures/e5_rl.png)", ""]

# ---------------- E5b
e5b = {r: load(f"e5_rl_{r}_latency_4") for r in ("sparse", "shaped")}
if any(e5b.values()):
    cfg = next(d for d in e5b.values() if d)["config"]
    lines += ["## E5b · Can RL learn where the teacher fails? (4-step latency)", "",
              "The scripted expert scores 0% here: its gain is stable at 2 steps of delay but oscillates at 4. "
              f"Same pipeline as E5 with {cfg['online_steps']:,} online steps. Success per seed:", "",
              "| Reward | Seed | BC | Offline IQL | Online checkpoints (every "
              f"{cfg['eval_every']:,} steps) |", "|---|---|---|---|---|"]
    for rw, d in e5b.items():
        for r in (d or {"rows": []})["rows"]:
            lines.append(f"| {rw} | {r['seed']} | {100*r['bc']['success']:.0f}% | {100*r['offline']['success']:.0f}% | "
                         + ", ".join(f"{100*c['success']:.0f}%" for c in r["online_curve"]) + " |")
    lines.append("")

open(os.path.join(RES, "REPORT.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
