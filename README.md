# Knowing What You Need Is Not Enough: The Policy Bottleneck in Homeostatic Agents

> Reproduction package for the paper submitted to the **LatinX in AI Workshop @ NeurIPS 2026**.
> Contains the simulator, all agent arms, every experiment referenced in the manuscript,
> the preregistered predictions and their commitment hash, and the figure generator.
>
> **Research question:** can a homeostatic agent learn its satiation signal `g` from the
> observable consequences of its own actions, and use that learned signal to regulate
> toward an externally useful goal?

Everything in the paper is reproducible from this repository on a laptop, in minutes,
with no GPU and no API keys. There is no external dataset and no pretrained model:
the environment is a simulator specified in full in `core.py`.

---

## Research line

This paper is not a standalone result. It belongs to a broader agenda on **agent
cybernetics** — the bet that the next useful move in agent design is not another
scaffolding layer around a capable model, but internal regulatory structure that
decides *when and why* a capability should be exercised at all. The field moved from
prompt engineering to context engineering to harness and loop engineering; each layer
added capability, none added motivation.

**Prior work in this line.** The **Pro-Action Operator (Γ)** was presented at the
LatinX in AI workshop at **ICML 2026**, where it was selected best paper. Γ wraps an
LLM executor in *n* coupled homeostatic subsystems: each drive decays basally, is
satiated by outcomes through a signal `g`, and is coupled antagonistically to the
others. That paper established feasibility in an iterated social dilemma.

A second study, presented as a poster at **SANLP 2026**, reduced Γ to a single drive
in adversarial multi-agent debate. It found that the regulatory loop works without any
reinforcement learning — and surfaced the limitation this paper attacks. The loop is
robust to *how* `g` is defined but blind to *what* `g` measures. Tied to a structural
metric it regulated participation but not semantic fidelity; tied to an LLM judge,
context collapse quintupled. The mechanism worked, toward the wrong thing.

That is the **Satiation Signal Problem (SSP)**: given a homeostatic mechanism, how is
`g` defined so the system regulates toward a desired external goal? It is value
alignment in the agent's internal loop rather than in its output. The SSP is one
problem inside the agenda, not the agenda itself — but it gates the rest, because the
mechanism is value-neutral and the signal feeding it decides what it regulates toward.

**This paper** takes the SSP as its direct object, in a deliberately interpretable
domain with no LLM in the loop, so that the regulatory dynamics can be measured rather
than inferred.

---

## Contents

1. [What this paper does](#1-what-this-paper-does)
2. [Findings](#2-findings)
3. [Repository structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Reproducing the paper](#5-reproducing-the-paper)
6. [Mapping: paper claim → script](#6-mapping-paper-claim--script)
7. [Methodology and why it is built this way](#7-methodology-and-why-it-is-built-this-way)
8. [Claims we withdrew](#8-claims-we-withdrew)
9. [Limitations](#9-limitations)
10. [Future directions](#10-future-directions)
11. [LLM disclosure](#11-llm-disclosure)
12. [Citation](#12-citation)
13. [License](#13-license)

---

## 1. What this paper does

The domain is a self-regulated learner with a fixed budget of **40 study sessions
before an exam**. The agent *is* the student. It chooses exercise difficulty guided by
two antagonistic drives — `δ_prog`, which pushes toward harder material, and
`δ_conf`, which brakes when errors accumulate — and it never observes its own ability,
fatigue or frustration. It sees only the level it chose, whether it answered
correctly, and how long it took.

We compare it against non-homeostatic baselines across a **dose–response sweep** of
environmental adversity, rather than at a single operating point. That choice is the
methodological core: an earlier iteration produced a null at one setting, which admits
two incompatible readings — the signal genuinely does not matter, or the environment
never applied the pressure that would make it matter. Only varying the pressure
separates them.

---

## 2. Findings

**Signal quality is real but small, and complementary to the policy.** Holding the
agent's hand-chosen action policy fixed, a *perfect* oracle `g` buys 7.7% of the gap
to a state-estimating baseline while re-tuning six policy coefficients buys 25.0%. But
the two are not independent: search the policy harder and the value of a good signal
rises from +0.047 to +0.158, and their interaction flips from negative to strongly
positive. A satiation signal is worth little until the channel can carry it.

**Roughly 59% of the gap survives both interventions.** That residual is what the two
interventions we ran do not explain — it also absorbs unsearched policy space and the
fixed action representation. It is not a proven structural constant.

**Homeostasis buys viability, but narrowly.** Estimation with no rest mechanism reaches
100% dropout under adversity; the homeostatic agent with rest disabled reaches 93.5% —
the predicted direction, not a difference in kind. With rest available both are safe
and the estimator additionally wins on gain. Viability is bought by having *some* rest
mechanism.

**Why regulation lost, and a proposed sixth condition.** Of the five applicability
conditions our framework declares, only one — that the agent holds information an
external observer lacks — makes homeostatic architecture *necessary* rather than merely
applicable. Our domain fails it: an outside observer of the outcome stream alone
reconstructs the agent's effective ability to within 0.32, on a scale where difficulty
levels sit 1.0 apart. We add a sixth condition — **the objective must be the
maintenance itself, not a terminal payoff that maintenance merely enables** — and show
that under a non-bankable objective, a long horizon and severe adversity, the ordering
reverses for the deterministic arm.

---

## 3. Repository structure

```
.
├── core.py                        environment, agents, episode runner
├── common.py                      arm construction, paired stats, TOST
├── predictions.txt                preregistered predictions (hashed before running)
├── run_all.py                     reproduce everything in order
├── requirements.txt
├── experiments/
│   ├── 00_predictions.py          verify the commitment hash
│   ├── 01_quality_gate.py         per-point environment discriminativeness
│   ├── 02_main_sweep.py           main sweep + P2/P3/P4/P5 contrasts
│   ├── 03_decomposition.py        signal / policy / interaction / residual
│   ├── 04_matched_tuning.py       per-signal policy search, held-out validation
│   ├── 05_ablations.py            objective-model, viability, rest-gate
│   ├── 06_shape_shift.py          misspecification the estimator cannot absorb
│   ├── 07_permutation_p3.py       multiplicity-corrected existence test
│   ├── 08_reconstruction.py       how private is the agent's internal state
│   └── 09_maintenance.py          maintenance objective (sixth condition)
├── figures/make_figure.py         six-panel summary, Okabe-Ito palette
└── results/                       JSON written by each experiment
```

---

## 4. Installation

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. Dependencies are `scipy` (statistics) and `matplotlib` (figure only).
Everything else is the standard library.

---

## 5. Reproducing the paper

```bash
python run_all.py            # ~10 min, everything except the large-budget search
python run_all.py --full     # ~35 min, adds the 1200-candidate tuning sweep
python figures/make_figure.py
```

Each script also runs standalone and prints the paper numbers it produces:

```bash
python experiments/02_main_sweep.py
python experiments/04_matched_tuning.py --budget 1200 --search-seeds 101 211 307
python experiments/09_maintenance.py --accumulative   # reproduces our FAILED first attempt
```

**Determinism.** Ten seeds are derived from master seed `20260820` via SHA-256, each
mapped to the next prime above `10007 + (digest mod 900000)`; profiles come from a
separate fixed seed. Re-running any script reproduces the published numbers exactly.

**Compute.** Intel Core i7-1165G7, 4 cores, 32 GB RAM, no GPU. The main sweep — 60,000
episodes across 6 adversity points and 10 arms — takes 31.4 s single-threaded. A single
agent decision costs 14 µs, consistent with the operator's `O(n² + A)` per-session
arithmetic: no gradients, no value function, no replay buffer. The whole project
consumed less compute than one GPU-minute.

---

## 6. Mapping: paper claim → script

| Paper | Claim / number | Script |
|---|---|---|
| §5.3 | Commitment hash `6e888c7e…` | `00_predictions.py` |
| §5.2, App C | Gate passes φ≤1.2, fails at φ=1.6 | `01_quality_gate.py` |
| §6.1 | Main sweep; estimator +1.132 → +0.156 | `02_main_sweep.py` |
| §6.1 | P2 slopes: −0.070 vs −0.808, d_z=+4.65 | `02_main_sweep.py` |
| §6.5 | P4 argmax > softmax, d_z +0.21 → +3.23 | `02_main_sweep.py` |
| §6.5 | P5 oracle − constant, +0.063 then equivalence | `02_main_sweep.py` |
| §6.5 | P3 crossover +0.045 at φ=1.2 | `02_main_sweep.py` |
| §6.5 | Multiplicity-corrected P3, p=.0005 | `07_permutation_p3.py` |
| §6.2, App F | 7.7 / 25.0 / −2.1 / 69.5 % decomposition | `03_decomposition.py` |
| §6.3, App U | Matched tuning, 250 and 1200 budgets | `04_matched_tuning.py` |
| §6.3 | learned − constant equivalence (TOST) | `04_matched_tuning.py` |
| §6.4, App H | Objective-model ablation | `05_ablations.py` |
| §7.2, App I | Viability ablation (rest removed both sides) | `05_ablations.py` |
| §7.2 | restdrive: best at φ=0, 4–17% dropout | `05_ablations.py` |
| §6.6, App K | Shape shift; P7 refuted | `06_shape_shift.py` |
| §7.3 | Reconstruction error 0.32 → 0.86 | `08_reconstruction.py` |
| §7.4, App J | Maintenance objective; sixth condition | `09_maintenance.py` |
| App M | Six-panel figure | `figures/make_figure.py` |

---

## 7. Methodology and why it is built this way

**Paired analysis.** Every profile faces every arm, so the design is within-subject and
the analysis unit is the per-profile mean over seeds. Contrasts use paired *t*-tests
with 95% CIs and the paired effect size *d_z*. This is not a detail: an earlier version
of this work applied independent-samples tests to the same paired data and reported a
null that pairing overturns.

**Equivalence, not non-significance.** Several conclusions rest on the *absence* of an
effect. A non-significant *p* is not evidence of absence, so we run two-one-sided-tests
against a smallest effect of interest δ=0.05, with sensitivity reported across
δ ∈ [0.02, 0.10].

**Existence claims get multiplicity control.** "Some Γ arm overtakes the estimator
somewhere" ranges over a family of arm × adversity cells. Quoting the winning cell
treats a selected contrast as pre-specified, so we use a max-statistic sign-flip
permutation test over the whole family.

**Held-out tuning.** Policy coefficients are scored on a 20-profile subset and validated
on 50 profiles never used for tuning, with multiple independent search seeds. A single
search validated in-sample produced an ordering that five searches erase.

**Quality gating.** At extreme adversity an environment can stop discriminating between
good and bad policies. We verify discriminativeness at every sweep point and report
only over the passing range — while also reporting the excluded point, where the
headline contrast is larger, so the exclusion is conservative.

**Baselines get ablated too.** It is easy to credit an architecture for something a
one-line rule was doing. Every non-homeostatic baseline is run with and without its
rest mechanism.

---

## 8. Claims we withdrew

Reported in the paper body rather than a footnote, because each was overturned by a
specific methodological choice and the pattern is the transferable lesson.

| Claim | What overturned it |
|---|---|
| Signal quality does not reach behaviour at all | Paired tests on the paired design: at φ=0 the effect is +0.063, d_z=+1.59 |
| A learned `g` beats a perfect one (non-monotonic ordering) | Five independent searches with held-out validation: monotone, oracle above constant 5/5 |
| A learned `g` is counterproductive versus a constant | Matched per-signal tuning: equivalent at 250 candidates, better at 1200 |
| Signal quality is simply the smallest term | True under a fixed policy; a larger search triples its value and flips the interaction positive |
| Homeostasis uniquely buys viability | Removing rest from both sides: the estimator hits 100% dropout, Γ 93.5% |
| Homeostasis integrates stopping into the same machinery | The default rest logit is frustration-gated; the drives barely participate |
| P7 refutes the value of situated information | The manipulation corrupts the competitor's model rather than granting Γ a private signal |

---

## 9. Limitations

- **The crossover is small, is not the main agent's, and does not survive the stronger
  baseline.** It is +0.045 — below our own δ=0.05 practical threshold — belongs to the
  deterministic `gamma_argmax` ablation, and fails against the model-free estimator.
- **Signal validity is incompletely operationalised.** Only `g_prog`, only against
  terminal expected gain, only as a five-point correlation; `g_conf` is unvalidated.
  Additionally `4a(1−a)` peaks at accuracy 0.5 while expected gain is maximised where
  accuracy is 0.354, so fidelity is capped by functional form.
- **Policy tuning was a random search**, not an optimum. The residual is an upper bound
  on what is structural.
- **Researcher degrees of freedom.** The commitment hash is self-attested rather than
  registered with a third party; the gate thresholds were outside the hashed text; the
  sixth condition is post-hoc.
- **Applicability condition (2)** — irreducible antagonistic tensions — has no dedicated
  test.
- **The simulator is simple**, the domain single-task, and 40 sessions across 5 levels
  give roughly six visits per level, which is thin for estimating `g` — itself one of
  our findings rather than only a limitation.

---

## 10. Future directions

**Learn the policy, not the signal.** With most of the gap unexplained by signal or
coefficients, the channel is the priority. We are deliberately non-committal about the
method: prior work in this line found value-based RL integrated poorly with homeostatic
drives, for reasons we do not yet understand, so direct policy search over the
coefficient vector is the conservative first step, with evolutionary methods natural
given how few parameters the operator has. Since matched tuning shows signal and policy
must be co-adapted, they should be searched jointly rather than in sequence.

**A stopping rule for signal learning.** If a learned `g` matches a constant under a
modest budget, the culprit may be that it *keeps* learning. A constant has zero
variance; a signal estimated from six visits per level injects noise into the drives at
every step, indefinitely. Would a convergence-stopped signal — annealing η → 0, or
freezing `g` once its moving average stabilises — keep the bias reduction while
shedding the variance?

**Test condition (4) properly.** The shape-shift manipulation corrupts the competitor's
model rather than granting the agent a private signal. A real test needs information
available only to Γ.

**Allostasis, via a world model.** Homeostasis corrects after deviation; allostasis
anticipates it. The jump demands a learned predictor of environmental dynamics and of
how the agent's actions propagate through them. What separates it from a self-model is
that the regularities live in the world — harder material produces errors at a rate set
by the gap to ability, effort compounds non-linearly, recovery has a threshold. The
agent authors none of these laws and must infer them; its drives are merely where they
register. Knowing your own state predicts nothing unless you also know how the world
will act on it. The system to model is the coupled pair, not the agent alone.

**Beyond simulation.** Validation against real educational data would test whether these
regulatory dynamics transfer.

---

## 11. LLM disclosure

All AI models were used under continuous human supervision. The authors retain full
responsibility for the content.

**No language model participates in the experiments.** The agent, the environment, the
baselines and the analysis are explicit numerical simulations. There is no LLM in any
measured loop.

| Tool | Role |
|---|---|
| Devin (IDE) with Kimi 3 and DeepSeek v4 Pro | Code assistance |
| Claude Opus 5, Claude Fable 5 | Code assistance, writing and editing assistance |

All experimental design decisions, the committed predictions, the statistical
procedures and every interpretation are the authors' own. Every reported number was
produced by executing the scripts in this repository.

---

## 12. Citation

```bibtex
@misc{anonymous2026policybottleneck,
  author       = {Anonymous},
  title        = {Knowing What You Need Is Not Enough:
                  The Policy Bottleneck in Homeostatic Agents},
  year         = {2026},
  howpublished = {LatinX in AI Workshop, NeurIPS 2026},
  note         = {Code: https://anonymous.4open.science/r/student_simulator-0AA6}
}
```

---

## 13. License

Distributed under the terms of [`LICENSE`](LICENSE).
