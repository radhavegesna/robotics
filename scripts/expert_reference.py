"""Expert (privileged, scripted) performance on every variation: the ceiling a
policy imitating it can reasonably reach."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relman.envs import VARIATIONS, make_env  # noqa: E402
from relman.evaluate import ExpertAgent, run_episode, summarize  # noqa: E402

rows = []
for v in VARIATIONS:
    env = make_env(v)
    holder = {}
    orig_step, orig_reset = env.step, env.reset

    def step(a, _o=orig_step):
        r = _o(a)
        holder["s"] = r[4]["state"]
        return r

    def reset(**k):
        r = orig_reset(**k)
        holder["s"] = r[1]["state"]
        return r

    env.step, env.reset = step, reset
    res = [run_episode(env, ExpertAgent(lambda: holder["s"]), 10_000 + i, assisted=False) for i in range(50)]
    s = summarize(res)
    rows.append({"variation": v, **s})
    print(v, s["success"], s["cycle_time"][0])
json.dump({"rows": rows}, open(os.path.join(os.path.dirname(__file__), "..", "results", "expert_reference.json"), "w"), indent=1)
