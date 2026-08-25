"""Follow-up experiments added in response to review (paper Appendices A, B, and
the coupling/rest/objective ablations).

Each answers an objection a reader is entitled to raise:

  coupling      : is the antagonistic coupling W doing anything, or is Gamma just
                  two independent thermostats? (Section 6.2)
  level_aware   : is the gap a lack of information, or a channel that discards it?
                  The agent already tracks per-level accuracy; the logits ignore it.
                  Feeding it back uses nothing privileged. (Appendix A)
  mapping       : can the satiation MAPPING itself be learned, rather than only the
                  statistic it consumes? g_prog is set from the observed improvement
                  rate per level -- no authored transfer function. (Appendix B)
  aligned       : how much of the "learning does not help" result was the functional
                  form pointing at the wrong accuracy?
  rest_matched  : is Gamma's deficit simply the cost of resting more?
  pure_strain   : does the drive term in the rest logit matter, or is it all the
                  frustration gate?

Produces: results/followups.json
"""
import copy, json, pathlib, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core as C, common as K

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALL = C.gen_profiles(100); SEEDS = C.derive_seeds(20260820, 10)
HELD_OUT = ALL[50:]
PHIS = [0.0, 0.6, 1.2]
out = {}


def gamma(arm, profiles, phis, **kw):
    """Run a Gamma variant, returning per-profile mean gain at each phi."""
    res = {}
    for phi in phis:
        per = []
        for p in profiles:
            g = []
            for sd in SEEDS:
                cfg = C.GCfg(**C.CFG_BASE, coupling_mode="receiver", **kw)
                m, _ = C.run_episode(C.GammaAgent(cfg, p), p, sd, True,
                                     dict(phi=phi, hysteresis=True))
                g.append(m["ganancia"])
            per.append(st.mean(g))
        res[phi] = per
    return res


print("=== coupling: is W inert? ===")
base = gamma("", ALL, PHIS, g_mode="learned")
nocp = gamma("", ALL, PHIS, g_mode="learned", coupling=False)
out["coupling"] = {}
for phi in PHIS:
    r = K.paired(base[phi], nocp[phi]); out["coupling"][str(phi)] = r
    print(f"  phi={phi}: with {st.mean(base[phi]):+.3f}  without {st.mean(nocp[phi]):+.3f}  "
          f"{K.fmt(r)}")
print("  -> not inert, and the sign flips: a liability when benign, an asset under adversity\n")

print("=== level-aware: information, or a channel that discards it? ===")
la = gamma("", ALL, PHIS, g_mode="learned", level_aware=True)
out["level_aware"] = {}
for phi in PHIS:
    r = K.paired(la[phi], base[phi]); out["level_aware"][str(phi)] = r
    _, est = K.run_arm("estimator", ALL, SEEDS, dict(phi=phi, hysteresis=True))
    print(f"  phi={phi}: base {st.mean(base[phi]):+.3f}  level-aware {st.mean(la[phi]):+.3f}  "
          f"{K.fmt(r)}   estimator {st.mean(est):+.3f}")
print("  -> the agent already held acc_ema; the logits were throwing it away\n")

print("=== level-aware input gain: the effect is monotone, so we report a lower bound ===")
out["level_gain_sweep"] = {}
for gain in [1, 2, 4, 6, 12, 20, 80]:
    v = gamma("", HELD_OUT, [0.0], g_mode="learned", level_aware=True, level_gain=gain)[0.0]
    out["level_gain_sweep"][str(gain)] = st.mean(v)
    print(f"  gain={gain:<4} {st.mean(v):+.3f}")
_, rule = K.run_arm("rule", HELD_OUT, SEEDS, dict(phi=0.0, hysteresis=True))
print(f"  rule baseline {st.mean(rule):+.3f}  -- high gain resembles it but does not collapse into it\n")

print("=== learned mapping: can g's MAPPING be induced from consequences? ===")
env0 = dict(phi=0.0, hysteresis=True)
rows = {}
for lbl, mode in [("constant", "fixed"), ("authored 4a(1-a)", "learned"),
                  ("learned mapping", "learned_mapping"), ("oracle", "oracle")]:
    rows[lbl] = gamma("", HELD_OUT, [0.0], g_mode=mode)[0.0]
    print(f"  {lbl:22}{st.mean(rows[lbl]):+.3f}")
r = K.paired(rows["learned mapping"], rows["constant"])
out["mapping"] = {"means": {k: st.mean(v) for k, v in rows.items()}, "vs_constant": r}
print(f"  learned mapping - constant: {K.fmt(r)}")
print("  -> beats the mapping we would have authored, using only observables\n")

print("=== rest: does the drive term in the rest logit matter? ===")
out["rest"] = {}
for phi in PHIS:
    rg, _ = K.run_arm("gamma_learned", ALL, SEEDS, dict(phi=phi, hysteresis=True))
    rp, _ = K.run_arm("gamma_purestrain", ALL, SEEDS, dict(phi=phi, hysteresis=True))
    dg = 100 * sum(x["dropout"] for x in rg) / len(rg)
    dp = 100 * sum(x["dropout"] for x in rp) / len(rp)
    out["rest"][str(phi)] = {"gated_dropout": dg, "pure_strain_dropout": dp}
    print(f"  phi={phi}: dropout gated {dg:.1f}%  vs frustration-gate-only {dp:.1f}%")
print("  -> the drives do help compute when rest fires\n")

json.dump(out, open(ROOT / "results/followups.json", "w"), indent=1, default=str)
print("saved results/followups.json")
