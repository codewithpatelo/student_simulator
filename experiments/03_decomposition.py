"""Signal / policy / interaction / residual decomposition (paper Sec 6.2, App F).

Why: the estimator beats the homeostatic agent by a wide margin at phi=0. Is that
because the agent's satiation signal is poor, because its action policy is badly
parameterised, or because of something neither intervention touches? A 2x2 over
signal quality (constant vs oracle g) and policy (hand-chosen vs re-tuned
coefficients) answers it, with the interaction reported so the parts sum exactly.

IMPORTANT: everything here is evaluated on the 50 profiles NEVER used for any
tuning, and everything here is specific to phi=0 -- the gap itself shrinks sharply
across the sweep, so these shares do not transfer.

Produces: results/decomposition.json
"""
import json, pathlib, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core as C, common as K

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALL = C.gen_profiles(100); SEEDS = C.derive_seeds(20260820, 10)
HELD_OUT = ALL[50:]                       # never used for tuning
ENV = dict(phi=0.0, hysteresis=True)
RETUNED = json.load(open(ROOT / "results/tuning.json"))["best_coef"] \
    if (ROOT / "results/tuning.json").exists() else \
    {"fixed": (1.09, -4.11, -1.95, 4.77, 1.97, -8.48),
     "oracle": (1.09, -4.11, -1.95, 4.77, 1.97, -8.48)}

cells = {}
for pol, coef in [("hand", K.BASE_COEF), ("retuned", None)]:
    for sig in ["gamma_fixed", "gamma_oracle", "gamma_learned"]:
        c = coef if coef else tuple(RETUNED.get(sig.split("_")[1], RETUNED["fixed"]))
        _, per = K.run_arm(sig, HELD_OUT, SEEDS, ENV, coef=c)
        cells[(pol, sig)] = per
_, est = K.run_arm("estimator", HELD_OUT, SEEDS, ENV)
_, est_noobj = K.run_arm("estimator_noobj", HELD_OUT, SEEDS, ENV)

M = {k: st.mean(v) for k, v in cells.items()}
print(f"{'policy':10}{'constant g':>12}{'learned g':>12}{'oracle g':>11}")
for pol in ["hand", "retuned"]:
    print(f"{pol:10}{M[(pol,'gamma_fixed')]:>+12.3f}{M[(pol,'gamma_learned')]:>+12.3f}"
          f"{M[(pol,'gamma_oracle')]:>+11.3f}")
print(f"\nestimator {st.mean(est):+.3f} | estimator without objective model {st.mean(est_noobj):+.3f}")

out = {}
for label, baseline in [("model_based", st.mean(est)), ("model_free", st.mean(est_noobj))]:
    tot = baseline - M[("hand", "gamma_fixed")]
    sig = M[("hand", "gamma_oracle")] - M[("hand", "gamma_fixed")]
    pol = M[("retuned", "gamma_fixed")] - M[("hand", "gamma_fixed")]
    both = M[("retuned", "gamma_oracle")] - M[("hand", "gamma_fixed")]
    inter = both - sig - pol
    res = baseline - M[("retuned", "gamma_oracle")]
    out[label] = {"total": tot, "signal": sig, "policy": pol,
                  "interaction": inter, "residual": res}
    print(f"\nvs {label} estimator (gap {tot:+.3f}), phi=0 only:")
    for k, v in [("signal", sig), ("policy", pol), ("interaction", inter), ("residual", res)]:
        print(f"   {k:12} {v:+.3f} ({100*v/tot:5.1f}%)")

print("\nThe residual is what the two interventions we ran do not explain.")
print("It also absorbs unsearched policy space and the fixed action representation.")
json.dump({"cells": {f"{a}|{b}": st.mean(v) for (a, b), v in cells.items()},
           "estimator": st.mean(est), "estimator_noobj": st.mean(est_noobj),
           "shares": out}, open(ROOT / "results/decomposition.json", "w"), indent=1)
