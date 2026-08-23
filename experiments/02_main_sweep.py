"""Main dose-response sweep (paper Section 6.1, Appendices D/N).

Why a sweep and not a single setting: an earlier iteration produced a null at one
operating point, which admits two incompatible readings -- the signal does not
matter, or the environment never applied pressure enough to make it matter. Only
varying the pressure separates them.

10 arms x 100 profiles x 10 seeds x 6 adversity points = 60,000 episodes (~31s).
Produces: results/sweep.json with gain, dropout, time-in-band (both denominators),
exam diagnostic, rest counts and signal validity per cell, plus paired contrasts
for P2 (robustness slopes), P3 (crossover), P4 (policy) and P5 (signal).
"""
import json, pathlib, statistics as st, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core as C, common as K

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILES = C.gen_profiles(100); SEEDS = C.derive_seeds(20260820, 10)
PHIS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.6]; VALID = [p for p in PHIS if p <= 1.2]
ARMS = ["gamma_learned", "gamma_argmax", "gamma_restdrive", "gamma_fixed",
        "gamma_naive", "gamma_oracle", "rule", "rule_norest",
        "estimator", "estimator_norest"]

t0 = time.time(); SWEEP, PER = {}, {}
for phi in PHIS:
    SWEEP[phi] = {}
    for arm in ARMS:
        rows, per = K.run_arm(arm, PROFILES, SEEDS, dict(phi=phi, hysteresis=True))
        PER[(phi, arm)] = per
        gv = [r["g_validity"] for r in rows if r.get("g_validity") is not None]
        SWEEP[phi][arm] = {
            "gain": st.mean(r["ganancia"] for r in rows),
            "drop": 100 * sum(r["dropout"] for r in rows) / len(rows),
            "tiz_conditional": st.mean(r["tiz"] for r in rows),
            "tib_full": st.mean(r["tib_full"] for r in rows),
            "exam_eff": st.mean(r["exam_eff"] for r in rows),
            "rest": st.mean(r["n_descansar"] for r in rows),
            "g_val": st.mean(gv) if gv else None}
print(f"{len(PHIS)*len(ARMS)*1000} episodes in {time.time()-t0:.1f}s\n")
print(f"{'arm':22}" + "".join(f"phi={p:<9}" for p in PHIS))
for a in ARMS:
    print(f"{a:22}" + "".join(f"{SWEEP[p][a]['gain']:+.3f}/{SWEEP[p][a]['drop']:>3.0f}% " for p in PHIS))

# ---- P2: robustness as a per-profile slope, not a ratio of aggregates ----
xm = st.mean(VALID); sxx = sum((x - xm) ** 2 for x in VALID)
def slopes(arm):
    series = [PER[(phi, arm)] for phi in VALID]
    out = []
    for j in range(len(PROFILES)):
        ys = [series[i][j] for i in range(len(VALID))]
        ym = st.mean(ys)
        out.append(sum((VALID[i] - xm) * (ys[i] - ym) for i in range(len(VALID))) / sxx)
    return out
r = K.paired(slopes("gamma_learned"), slopes("estimator"))
print(f"\nP2 slope difference (gamma_learned - estimator): {K.fmt(r)}")

CON = {}
print("\nphi   P5 oracle-constant        P4 argmax-softmax       P3 argmax-estimator")
for phi in VALID:
    p5 = K.paired(PER[(phi, "gamma_oracle")], PER[(phi, "gamma_fixed")])
    p4 = K.paired(PER[(phi, "gamma_argmax")], PER[(phi, "gamma_learned")])
    p3 = K.paired(PER[(phi, "gamma_argmax")], PER[(phi, "estimator")])
    p5["tost"] = K.tost(PER[(phi, "gamma_oracle")], PER[(phi, "gamma_fixed")])
    CON[phi] = {"P5": p5, "P4": p4, "P3": p3}
    print(f"{phi:<6}{p5['diff']:+.3f} (TOST {p5['tost']:.3f})     "
          f"{p4['diff']:+.3f} dz={p4['dz']:+.2f}      {p3['diff']:+.3f} dz={p3['dz']:+.2f}")

json.dump({"sweep": {str(k): v for k, v in SWEEP.items()},
           "contrasts": {str(k): v for k, v in CON.items()},
           "p2_slope": r, "phis": PHIS, "valid": VALID},
          open(ROOT / "results/sweep.json", "w"), indent=1, default=str)
print("\nsaved results/sweep.json")
