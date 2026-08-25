"""Shared helpers: arm construction, paired statistics, equivalence tests.

Every contrast in the paper is WITHIN-SUBJECT: the same profiles face every arm.
The analysis unit is therefore the per-profile mean over seeds, and contrasts use
paired tests. Using independent-samples tests here is a Type II error factory --
an earlier version of this work did exactly that and reported a null that pairing
overturns (see paper, Section 6.5).
"""
import math, statistics as st
from scipy.stats import ttest_rel, t as tdist
import core as C

SESOI = 0.05          # smallest effect of interest, learning-gain units
BASE_COEF = (4.0, -3.0, -3.0, 1.5, 4.0, -3.0)   # hand-chosen action logits
G_MODE = {"gamma_learned": "learned", "gamma_argmax": "learned",
          "gamma_restdrive": "learned", "gamma_fixed": "fixed",
          "gamma_naive": "naive", "gamma_oracle": "oracle",
          "gamma_oracle_corrected": "oracle_corrected",
          "gamma_aligned": "learned_aligned", "gamma_nocoupling": "learned",
          "gamma_purestrain": "learned",
          "gamma_levelaware": "learned", "gamma_mapping": "learned_mapping"}
ARMS = ["gamma_learned", "gamma_argmax", "gamma_restdrive", "gamma_fixed",
        "gamma_naive", "gamma_oracle", "gamma_oracle_corrected", "rule", "rule_norest",
        "estimator", "estimator_norest", "estimator_noobj"]


def build(arm, profile, coef=None, rest_mode=None):
    """Construct an agent. Returns (agent, is_gamma)."""
    if arm in G_MODE:
        kw = dict(**C.CFG_BASE, g_mode=G_MODE[arm], coupling_mode="receiver")
        if arm == "gamma_argmax":    kw["action_mode"] = "argmax"
        if arm == "gamma_restdrive": kw["rest_mode"] = "drive"
        if arm == "gamma_nocoupling": kw["coupling"] = False
        if arm == "gamma_purestrain": kw["rest_mode"] = "pure_strain"
        if arm == "gamma_levelaware": kw["level_aware"] = True
        if rest_mode:                kw["rest_mode"] = rest_mode
        if coef:                     kw["coef"] = coef
        return C.GammaAgent(C.GCfg(**kw), profile), True
    if arm == "rule":             return C.ReglaAgent(profile, rest=True), False
    if arm == "rule_norest":      return C.ReglaAgent(profile, rest=False), False
    if arm == "estimator":        return C.BayesianAgent(profile, rest=True), False
    if arm == "estimator_norest": return C.BayesianAgent(profile, rest=False), False
    if arm == "estimator_noobj":  return C.BayesianNoObjAgent(profile, rest=True), False
    if arm == "estimator_restmatched": return C.BayesianRestMatchedAgent(profile), False
    raise ValueError(arm)


def run_arm(arm, profiles, seeds, env_kw, coef=None, rest_mode=None, horizon=None):
    """Run one arm over all (profile, seed) cells.
    Returns (all metric dicts, per-profile mean gain) -- the latter is the analysis unit."""
    rows, per_profile = [], []
    for p in profiles:
        gains = []
        for sd in seeds:
            ag, is_g = build(arm, p, coef, rest_mode)
            m, _ = C.run_episode(ag, p, sd, is_g, env_kw, horizon=horizon)
            rows.append(m); gains.append(m["ganancia"])
        per_profile.append(st.mean(gains))
    return rows, per_profile


def paired(x, y):
    """Paired difference with 95% CI, p-value and paired effect size d_z."""
    d = [u - v for u, v in zip(x, y)]
    n = len(d); m = st.mean(d); sd = st.stdev(d); se = sd / math.sqrt(n)
    h = se * tdist.ppf(0.975, n - 1)
    t, p = ttest_rel(x, y)
    return {"diff": m, "ci": (m - h, m + h), "p": float(p), "dz": m / sd, "n": n}


def tost(x, y, delta=SESOI):
    """Two one-sided tests. p < .05 => the difference lies inside +/-delta,
    i.e. practically equivalent. A non-significant t-test is NOT evidence of
    absence; this is."""
    d = [u - v for u, v in zip(x, y)]
    n = len(d); m = st.mean(d); se = st.stdev(d) / math.sqrt(n)
    return max(1 - tdist.cdf((m + delta) / se, n - 1),
                   tdist.cdf((m - delta) / se, n - 1))


def fmt(r):
    return f"{r['diff']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p']:.2e} dz={r['dz']:+.2f}"
