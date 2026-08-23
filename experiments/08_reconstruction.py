"""How much does the agent privately know? (paper Sec 7.3)

Why: applicability condition (4) says the agent must observe its own condition in
ways an external orchestrator cannot. If that fails, homeostatic architecture is
applicable but not NECESSARY -- an outside estimator could do the same job.

Method: we run the estimator's own update rule as an external observer. It sees
only the level attempted and whether the answer was correct -- never ability,
fatigue or frustration -- and we measure |h_hat - h_eff| from session 10 onward.
Levels are spaced 1.0 apart, so an error of 0.32 means the "hidden" state is
essentially public.

Produces: results/reconstruction.json
"""
import json, math, pathlib, random, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scipy.stats import t as tdist
import core as C

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = C.gen_profiles(40); S = C.derive_seeds(20260820, 5)
out = {}
print("mean |h_hat - h_eff| for an observer of outcomes only (from session 10)")
print(f"{'phi':6}{'MAE':>9}{'95% CI':>20}{'n profiles':>12}")
for phi in [0.0, 0.6, 1.2]:
    per_profile = []
    for p in P:
        errs = []
        for sd in S:
            ag = C.BayesianAgent(p, rest=True)
            env = C.StudyEnv(p["h0"], lr_mult=p["lr_mult"], phi=phi, hysteresis=True)
            rng = random.Random(sd)
            for s in range(40):
                if ag.abandono: break
                env.tick(s); a = ag.select(rng, env)
                if a == "subir":   ag.nivel = min(5, ag.nivel + 1)
                elif a == "bajar": ag.nivel = max(1, ag.nivel - 1)
                if a == "descansar":
                    env.descansar(); ag.frust *= 0.5
                else:
                    o = env.exercise(ag.nivel, rng)
                    eb = ag.acc_ema.get(ag.nivel, 0.5)
                    ag.acc_ema[ag.nivel] = 0.8 * eb + 0.2 * float(o["correcto"])
                    C.shared_frust_update(ag, o, eb); ag.study(env, o, ag.nivel)
                    if s >= 10: errs.append(abs(ag.h_hat - env.h_eff))
                if ag.frust >= 0.95 * p["umbral"]: ag.consec_high += 1
                else: ag.consec_high = 0
                if ag.consec_high >= 3: ag.abandono = True
        if errs: per_profile.append(st.mean(errs))
    n = len(per_profile); m = st.mean(per_profile)
    se = st.stdev(per_profile) / math.sqrt(n); h = se * tdist.ppf(.975, n - 1)
    out[phi] = {"mae": m, "ci": (m - h, m + h), "n": n}
    print(f"{phi:<6}{m:>9.3f}   [{m-h:.3f}, {m+h:.3f}]{n:>12}")
print("\nLevel spacing is 1.0. At phi=0 nothing the agent 'privately' knows is private.")
json.dump({str(k): v for k, v in out.items()},
          open(ROOT / "results/reconstruction.json", "w"), indent=1, default=str)
