"""Multiplicity-corrected test for the crossover (paper Sec 6.5).

Why: P3 says "SOME Gamma arm overtakes the estimator somewhere in the valid range".
That is an existence claim over a family of arm x adversity cells. Quoting the
p-value of the cell where a crossover happens to appear treats a selected contrast
as if it had been fixed in advance. A max-statistic sign-flip permutation test over
the whole family gives a valid p-value for the claim as stated.

Produces: results/permutation_p3.json
"""
import json, math, pathlib, random, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core as C, common as K

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = C.gen_profiles(100); S = C.derive_seeds(20260820, 10)
PHIS = [0.0, 0.3, 0.6, 0.9, 1.2]
GAMMA_ARMS = ["gamma_learned", "gamma_argmax", "gamma_restdrive",
              "gamma_fixed", "gamma_naive", "gamma_oracle"]
NPERM = 2000

diffs = {}
for phi in PHIS:
    env = dict(phi=phi, hysteresis=True)
    _, base = K.run_arm("estimator", P, S, env)
    for arm in GAMMA_ARMS:
        _, x = K.run_arm(arm, P, S, env)
        diffs[(phi, arm)] = [u - v for u, v in zip(x, base)]

def tstat(d):
    n = len(d); return st.mean(d) / (st.stdev(d) / math.sqrt(n))

obs = {k: tstat(v) for k, v in diffs.items()}
T_obs = max(obs.values()); best = max(obs, key=obs.get)
rng = random.Random(11); count = 0
for _ in range(NPERM):
    signs = [1 if rng.random() < .5 else -1 for _ in range(len(P))]
    mx = max(tstat([s * x for s, x in zip(signs, d)]) for d in diffs.values())
    if mx >= T_obs: count += 1
p = (count + 1) / (NPERM + 1)
print(f"family: {len(GAMMA_ARMS)} Gamma arms x {len(PHIS)} adversity points = {len(diffs)} cells")
print(f"max t = {T_obs:+.2f} at {best}")
print(f"sign-flip permutation p = {p:.4f}  ({NPERM} permutations)")
json.dump({"max_t": T_obs, "argmax_cell": str(best), "p": p,
           "family_size": len(diffs), "n_perm": NPERM},
          open(ROOT / "results/permutation_p3.json", "w"), indent=1)
