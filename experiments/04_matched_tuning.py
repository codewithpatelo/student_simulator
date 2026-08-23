"""Matched policy tuning per signal (paper Sec 6.3, App U).

Why: the shared re-tuned coefficients were optimised against stable signals, so
applying them to a noisy learned signal is not a fair test. Here each signal gets
its OWN search with an identical budget, multiple independent search seeds, and
validation on 50 profiles never used for tuning.

This is the experiment that overturned two of our own claims. A single search per
signal (validating on the tuning profiles) produced a non-monotonic ordering that
five independent searches erase; and the "learned g is equivalent to a constant"
result holds at 250 candidates but not at 1200.

Runtime: ~3 min at BUDGET=250, ~25 min at BUDGET=1200.
Produces: results/tuning.json
"""
import argparse, json, math, pathlib, random, statistics as st, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import core as C, common as K

ap = argparse.ArgumentParser()
ap.add_argument("--budget", type=int, default=250)
ap.add_argument("--search-seeds", type=int, nargs="+", default=[7, 13, 29, 41, 53])
args = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parents[1]
ALL = C.gen_profiles(100); SEEDS = C.derive_seeds(20260820, 10)
TUNE, HELD_OUT = ALL[:50], ALL[50:]          # strictly disjoint
SCORE_P, SCORE_S = TUNE[:20], SEEDS[:3]      # cheap scoring subset
ENV = dict(phi=0.0, hysteresis=True)
SIGNALS = ["gamma_fixed", "gamma_learned", "gamma_oracle"]

def score(arm, coef):
    return st.mean(K.run_arm(arm, SCORE_P, SCORE_S, ENV, coef=coef)[1])

def search(arm, seed, budget):
    r = random.Random(seed); best = (K.BASE_COEF, score(arm, K.BASE_COEF))
    for _ in range(budget):
        c = (r.uniform(0, 10), r.uniform(-10, 2), r.uniform(-10, 0),
             r.uniform(-2, 5), r.uniform(0, 10), r.uniform(-10, 2))
        v = score(arm, c)
        if v > best[1]: best = (c, v)
    return best[0]

RES = {s: [] for s in SIGNALS}; POOL = {s: [[] for _ in HELD_OUT] for s in SIGNALS}
BEST = {}
for sd in args.search_seeds:
    for sig in SIGNALS:
        coef = search(sig, sd, args.budget)
        _, per = K.run_arm(sig, HELD_OUT, SEEDS, ENV, coef=coef)
        RES[sig].append(st.mean(per)); BEST.setdefault(sig, coef)
        for i, v in enumerate(per): POOL[sig][i].append(v)
        print(f"  seed={sd} {sig:14} held-out {st.mean(per):+.3f}", flush=True)

print(f"\n=== budget {args.budget}, {len(args.search_seeds)} search seeds, "
      f"validated on {len(HELD_OUT)} held-out profiles ===")
for s in SIGNALS:
    sd_ = st.stdev(RES[s]) if len(RES[s]) > 1 else 0.0
    print(f"  {s:14} {[round(x,3) for x in RES[s]]}  mean {st.mean(RES[s]):+.3f} sd {sd_:.3f}")
wins = sum(1 for i in range(len(RES['gamma_oracle']))
           if RES['gamma_oracle'][i] > RES['gamma_fixed'][i])
print(f"  oracle > constant in {wins}/{len(RES['gamma_oracle'])} searches")

P = {s: [st.mean(a) for a in POOL[s]] for s in SIGNALS}
print("\npaired held-out contrasts (n=%d profiles, averaged over search seeds):" % len(HELD_OUT))
for a, b, lbl in [("gamma_learned", "gamma_fixed", "learned - constant"),
                  ("gamma_oracle", "gamma_fixed", "oracle - constant"),
                  ("gamma_oracle", "gamma_learned", "oracle - learned")]:
    r = K.paired(P[a], P[b]); t = K.tost(P[a], P[b])
    verdict = "EQUIVALENT" if (t < .05 and r["p"] >= .05) else ("differs" if r["p"] < .05 else "inconclusive")
    print(f"  {lbl:20} {K.fmt(r)} TOST={t:.4f}  {verdict}")

json.dump({"budget": args.budget, "per_seed": RES, "best_coef": {k: list(v) for k, v in BEST.items()},
           "held_out_profiles": len(HELD_OUT)},
          open(ROOT / f"results/tuning_b{args.budget}.json", "w"), indent=1)
if args.budget == 250:
    json.dump({"best_coef": {k: list(v) for k, v in BEST.items()}},
              open(ROOT / "results/tuning.json", "w"), indent=1)
