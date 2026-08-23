# Learning the Satiation Signal Is Not Enough: Decomposing the Bottleneck in Homeostatic Agents

Supplementary material for the NeurIPS 2026 LatinX in AI Workshop submission.

**Question:** Can a homeostatic agent learn its satiation signal *g* from observable consequences, and use it to regulate toward an externally useful goal?

**Answer (in two halves):** *g* is learnable from observable productivity (validity +0.471 against a +0.996 oracle ceiling), but the learned signal does not reach behavior — an oracle arm with *perfect* *g* is statistically indistinguishable from a fixed-*g* arm at all five valid adversity levels. The bottleneck is the drive-to-action channel (Regulatory Lag Problem), decomposed into removable policy stochasticity (argmax beats softmax at 5/5 points, d=+1.47) and a persistent structural component. Homeostatic regulation earns its place for viability, not optimality: pure state estimation reaches 100% dropout under severe adversity while the homeostatic agent stays below 4% with no explicit viability rule.

## Contents

| File | Description |
|------|-------------|
| `paper_lxai2026_v4.tex` / `.pdf` | Paper (LaTeX source + compiled) |
| `checklist.tex` | NeurIPS checklist |
| `neurips_2026.sty` | Style file (required to compile) |
| `fig_v4.png` | Six-panel results figure |
| `tutor_experiment_v4.ipynb` | Reproducible notebook (full experiment) |
| `core_v4.py` | Experiment core (environment + agents) |
| `experiment_v4_results.json` | Full results: aggregates, quality gate, prediction verdicts |
| `predictions.txt` | Preregistered predictions P1–P7 (written before execution) |

## Reproduce

```
jupyter nbconvert --execute tutor_experiment_v4.ipynb
```

12,000 episodes (20 profiles × 10 seeds × 10 arms × 6 adversity points) + quality gate + regime-shift sweep in ~6.3 s, single-threaded, no GPU. Pure Python; dependencies: `scipy`, `matplotlib`.

Seeds derived from master seed `20260820` via SHA-256.

## Compile the paper

```
pdflatex paper_lxai2026_v4.tex
pdflatex paper_lxai2026_v4.tex
```
