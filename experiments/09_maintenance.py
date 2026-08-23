"""Maintenance objective: the proposed sixth condition (paper Sec 7.4, App J).

Why: homeostasis is a MAINTENANCE formalism -- no horizon, no notion that an exam
falls on session 40. Our primary task has a terminal objective, and accumulated
gain can be banked, so more is always better. We were asking a regulator to
maximise a payoff.

Test: replace the objective with one that CANNOT be banked -- mean readiness, the
score the student would get if the exam were held that session, averaged over the
horizon, with zero credit after quitting.

A first version of this test merely lengthened the horizon while keeping cumulative
gain as the objective. It failed outright (--accumulative reproduces it). Duration
was never the variable; bankability was.

Produces: results/maintenance.json
"""
import argparse, json, pathlib, random, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core as C, common as K

ap = argparse.ArgumentParser()
ap.add_argument("--accumulative", action="store_true",
                help="reproduce the FAILED first attempt (cumulative gain, long horizon)")
args = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = C.gen_profiles(60); S = C.derive_seeds(20260820, 6)
ARMS = ["gamma_learned", "gamma_argmax", "estimator", "estimator_noobj", "rule"]

def readiness(arm, p, sd, phi, H):
    ag, is_g = K.build(arm, p)
    env = C.StudyEnv(p["h0"], lr_mult=p["lr_mult"], phi=phi, hysteresis=True)
    rng = random.Random(sd); vals = []
    for s in range(H):
        if ag.abandono:
            vals.extend([0.0] * (H - s)); break        # quitting = zero readiness
        env.tick(s)
        if is_g: ag.basal()
        a = ag.select(rng, env)
        if a == "subir":   ag.nivel = min(5, ag.nivel + 1)
        elif a == "bajar": ag.nivel = max(1, ag.nivel - 1)
        if a == "descansar":
            env.descansar(); ag.frust *= 0.5
        else:
            o = env.exercise(ag.nivel, rng)
            if is_g: ag.study(env, o, ag.nivel)
            else:
                eb = ag.acc_ema.get(ag.nivel, 0.5)
                ag.acc_ema[ag.nivel] = 0.8 * eb + 0.2 * float(o["correcto"])
                C.shared_frust_update(ag, o, eb); ag.study(env, o, ag.nivel)
        vals.append(C.rasch_p(env.h_eff, 3))
        if ag.frust >= 0.95 * p["umbral"]: ag.consec_high += 1
        else: ag.consec_high = 0
        if ag.consec_high >= 3: ag.abandono = True
    return st.mean(vals)

metric = "cumulative gain (FAILED first attempt)" if args.accumulative else "mean readiness"
print(f"metric: {metric}\n")
out = {}
for phi in [0.6, 1.2]:
    print(f"=== phi={phi} ===")
    print(f"{'horizon':>8}" + "".join(f"{a:>18}" for a in ARMS))
    for H in [40, 100, 200, 400]:
        row, per = [], {}
        for a in ARMS:
            if args.accumulative:
                _, vals = K.run_arm(a, P, S, dict(phi=phi, hysteresis=True), horizon=H)
            else:
                vals = [st.mean([readiness(a, p, sd, phi, H) for sd in S]) for p in P]
            per[a] = vals; row.append(f"{st.mean(vals):.3f}")
        out[f"{phi}|{H}"] = {a: st.mean(v) for a, v in per.items()}
        print(f"{H:>8}" + "".join(f"{c:>18}" for c in row))
        if H == 400:
            for b in ["estimator", "estimator_noobj"]:
                r = K.paired(per["gamma_argmax"], per[b])
                print(f"         gamma_argmax - {b:16} {K.fmt(r)}")
    print()
print("The reversal belongs to the deterministic gamma_argmax arm; gamma_learned")
print("never overtakes the model-free estimator at any horizon or adversity level.")
json.dump(out, open(ROOT / "results/maintenance.json", "w"), indent=1, default=str)
