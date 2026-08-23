"""Quality gate at every sweep point (paper Section 5.2, Appendix C).

Why: at extreme adversity an environment can stop discriminating between good and
bad policies, and comparisons there are meaningless. Before trusting any contrast
we check the environment still has signal, using four fixed policies.

Criteria: the level-1 trap stays unprofitable (<0.15), the absorbing boundary bites
(always5 dropout >60%), and an oracle policy beats random by >0.15.

Produces: results/gate.json. Paper: gate passes for phi<=1.2, fails at phi=1.6
where severe fatigue makes hiding at level 1 viable and the trap dissolves.
"""
import json, pathlib, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core as C

PROFILES = C.gen_profiles(100); SEEDS = C.derive_seeds(20260820, 10)
PHIS = [0.0, 0.3, 0.6, 0.9, 1.2, 1.6]
gate = {}
print(f"{'phi':6}{'always1':>10}{'a5 drop':>10}{'random':>10}{'oracle':>10}{'margin':>10}  gate")
for phi in PHIS:
    r = {}
    for mode in ["always1", "always5", "random", "oracle"]:
        g, d = [], []
        for p in PROFILES:
            for sd in SEEDS:
                m, _ = C.run_episode(C.FixedAgent(p, mode), p, sd, False,
                                     dict(phi=phi, hysteresis=True))
                g.append(m["ganancia"]); d.append(m["dropout"])
        r[mode] = (st.mean(g), 100 * sum(d) / len(d))
    margin = r["oracle"][0] - r["random"][0]
    ok = (r["always1"][0] < 0.15 and r["always5"][1] > 60
          and margin > 0.15 and r["random"][0] < r["oracle"][0])
    gate[phi] = {"always1": r["always1"][0], "always5_drop": r["always5"][1],
                 "random": r["random"][0], "oracle": r["oracle"][0],
                 "margin": margin, "pass": ok}
    print(f"{phi:<6}{r['always1'][0]:>+10.3f}{r['always5'][1]:>9.0f}%"
          f"{r['random'][0]:>+10.3f}{r['oracle'][0]:>+10.3f}{margin:>+10.3f}  "
          f"{'PASS' if ok else 'FAIL'}")
valid = [p for p in PHIS if gate[p]["pass"]]
print("\nvalid range:", valid)
json.dump({str(k): v for k, v in gate.items()},
          open(pathlib.Path(__file__).resolve().parents[1] / "results/gate.json", "w"), indent=1)
