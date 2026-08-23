"""Model misspecification the estimator cannot re-estimate away (paper Sec 6.6, App K).

Why: multiplying fatigue mid-episode is a change of SCALE, and an outcome-driven
estimator absorbs it by re-estimating. To test whether situated regulation beats
modelled estimation when the model is wrong, we instead move the PEAK of the
learning curve at session 20. The estimator has the original peak hard-coded, so
its ability estimate stays right while its level choice goes systematically wrong.

Result (P7 refuted): the gap WIDENS. This does not refute situated information --
the manipulation corrupts the competitor's model rather than giving Gamma a private
signal, and Gamma's own rule (anchored at accuracy 0.5) ends up further from the
true peak than the estimator's fixed assumption.

Produces: results/shape_shift.json
"""
import json, pathlib, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core as C, common as K

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = C.gen_profiles(100); S = C.derive_seeds(20260820, 10)
PEAKS = [0.5, 1.0, 1.5, 2.0]; PHI = 0.6
out = {}
print(f"{'peak':6}{'argmax':>10}{'oracle':>10}{'estimator':>12}{'gap':>9}{'dz':>8}  gate")
for pk in PEAKS:
    env = dict(phi=PHI, hysteresis=True, peak_new=pk, peak_at=20)
    g = {}
    for mode in ["always1", "always5", "random", "oracle"]:
        vals, drops = [], []
        for p in P:
            for sd in S:
                m, _ = C.run_episode(C.FixedAgent(p, mode), p, sd, False, env)
                vals.append(m["ganancia"]); drops.append(m["dropout"])
        g[mode] = (st.mean(vals), 100 * sum(drops) / len(drops))
    margin = g["oracle"][0] - g["random"][0]
    ok = g["always1"][0] < 0.15 and g["always5"][1] > 60 and margin > 0.15
    _, a = K.run_arm("gamma_argmax", P, S, env)
    _, o = K.run_arm("gamma_oracle", P, S, env)
    _, e = K.run_arm("estimator", P, S, env)
    r = K.paired(a, e)
    out[pk] = {"argmax": st.mean(a), "oracle": st.mean(o), "estimator": st.mean(e),
               "contrast": r, "gate_margin": margin, "gate_pass": ok}
    print(f"{pk:<6}{st.mean(a):>+10.3f}{st.mean(o):>+10.3f}{st.mean(e):>+12.3f}"
          f"{r['diff']:>+9.3f}{r['dz']:>+8.2f}  {'PASS' if ok else 'FAIL'}")
gaps = [out[p]["contrast"]["diff"] for p in PEAKS]
print(f"\nP7 {'CONFIRMED' if gaps[-1] > gaps[0] else 'REFUTED'}: "
      f"gap {gaps[0]:+.3f} -> {gaps[-1]:+.3f}")
json.dump({str(k): v for k, v in out.items()},
          open(ROOT / "results/shape_shift.json", "w"), indent=1, default=str)
