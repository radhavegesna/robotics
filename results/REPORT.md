# Stage 1 results

All numbers are produced by `scripts/run_experiments.py` on CPU. Success rates pool episodes across training seeds and show 95% Wilson intervals in brackets. Held-out variations were never seen during training.

## E1 · Architecture vs held-out task variations

100 demos per seed, 3000 gradient steps, 50 evaluation episodes per variation per seed, seeds completed: 3.

| Variation | Expert | MLP | LSTM | Transformer |
|---|---|---|---|---|
| nominal | 100% | 97% [93–99] | 100% [98–100] | 99% [96–100] |
| wide_workspace | 100% | 69% [61–76] | 88% [82–92] | 97% [93–99] |
| heavy_slippery | 66% | 87% [80–91] | 79% [71–84] | 84% [77–89] |
| noisy_sensor | 100% | 59% [51–67] | 94% [89–97] | 76% [69–82] |
| latency | 100% | 73% [66–80] | 74% [66–80] | 57% [49–65] |

Mean cycle time on successful episodes (steps, lower is faster):

| Variation | Expert | mlp | lstm | transformer |
|---|---|---|---|---|
| nominal | 16.6 | 16.7 | 17.1 | 17.3 |
| wide_workspace | 17.8 | 17.4 | 19.6 | 19.5 |
| heavy_slippery | 15.5 | 16.2 | 16.2 | 16.7 |
| noisy_sensor | 16.6 | 38.8 | 19.5 | 35.5 |
| latency | 31.3 | 44.3 | 40.4 | 44.8 |

Failure categories (all seeds, all held-out variations):

| Policy | never_grasped | dropped | carried_not_placed | placed_then_drifted |
|---|---|---|---|---|
| mlp | 101 | 36 | 6 | 25 |
| lstm | 22 | 20 | 10 | 46 |
| transformer | 26 | 51 | 10 | 41 |

![E1](figures/e1_architectures.png)

## E2 · How many demonstrations does BC need? (MLP)

| Demos | Nominal success | Nominal interventions/ep | Wide-workspace success | Wide interventions/ep |
|---|---|---|---|---|
| 10 | 17% [12–24] | 2.67 | 13% [9–20] | 2.73 |
| 25 | 55% [47–62] | 0.93 | 39% [32–47] | 1.60 |
| 50 | 74% [66–80] | 0.31 | 43% [36–51] | 0.71 |
| 100 | 90% [84–94] | 0.10 | 63% [55–70] | 0.47 |
| 200 | 97% [93–99] | 0.03 | 70% [62–77] | 0.33 |

![E2](figures/e2_demo_scaling.png)

## E3 · HG-DAgger vs more demonstrations at equal expert effort (MLP)

Both start from 10 demos. Each HG-DAgger round deploys the policy with the supervisor for 20 episodes and adds only the takeover segments. The baseline gets the same number of expert-labelled steps as extra full demonstrations.

| Round | Method | Expert-labelled steps | Autonomous success | Interventions/ep |
|---|---|---|---|---|
| 0 | hg_dagger | 800 | 13% [9–20] | 3.18 |
| 1 | hg_dagger | 1316 | 58% [50–66] | 0.66 |
| 2 | hg_dagger | 1419 | 79% [71–84] | 0.41 |
| 3 | hg_dagger | 1445 | 80% [73–86] | 0.42 |
| 4 | hg_dagger | 1504 | 90% [84–94] | 0.23 |
| 5 | hg_dagger | 1555 | 93% [87–96] | 0.14 |
| 0 | bc_equal_effort | 800 | 13% [9–20] | 3.18 |
| 1 | bc_equal_effort | 1360 | 33% [26–41] | 1.86 |
| 2 | bc_equal_effort | 1440 | 35% [28–43] | 1.92 |
| 3 | bc_equal_effort | 1493 | 39% [32–47] | 1.56 |
| 4 | bc_equal_effort | 1547 | 38% [31–46] | 1.45 |
| 5 | bc_equal_effort | 1600 | 47% [40–55] | 1.13 |

![E3](figures/e3_hg_dagger.png)

