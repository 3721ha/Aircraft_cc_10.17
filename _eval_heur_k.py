import sys, numpy as np
sys.path.insert(0, 'D:/Desktop/Aircraft_mumt_cc/Aircraft_mumt')
from aircraft_sim.detect_env import DetectEnv
from aircraft_sim.detect_shield import DetectQPShield
from aircraft_sim.detect_baselines import HeurPolicy

def run(k, seeds, use_shield=False):
    heur = HeurPolicy()
    succ, terms = 0, []
    for seed in seeds:
        env = DetectEnv()
        shield = DetectQPShield(env.cfg)
        env.reset({'seed': seed, 'k_red': k})
        ret = 0.0
        while env.terminal is None:
            acts = heur(env)
            if use_shield:
                acts = shield(env, acts)
            res = env.step(acts)
            ret += res.reward
        terms.append((env.terminal, round(ret,1)))
        if env.terminal == 'success':
            succ += 1
    return succ, len(seeds), terms

for k in [2,4,6,8]:
    seeds = list(range(200, 200+6))
    s, n, terms = run(k, seeds, use_shield=False)
    print(f"HEUR K={k}: {s}/{n} success  {terms}")
print("--- with shield ---")
for k in [2,4]:
    seeds = list(range(200, 200+6))
    s, n, terms = run(k, seeds, use_shield=True)
    print(f"HEUR+shield K={k}: {s}/{n} success  {terms}")
