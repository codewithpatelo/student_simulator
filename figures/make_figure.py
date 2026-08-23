"""Six-panel summary figure (paper Appendix M).

Uses the Okabe-Ito colourblind-safe palette, with distinct markers and line styles
so every panel remains readable in greyscale and for all three forms of colour
vision deficiency. Reads results/sweep.json, writes figures/fig_v4.png.
"""
import json, pathlib, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
S = json.load(open(ROOT / "results/sweep.json"))["sweep"]
V = ["0.0", "0.3", "0.6", "0.9", "1.2"]; X = [float(v) for v in V]
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "skyblue": "#56B4E9", "purple": "#CC79A7",
      "black": "#000000"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130})
STY = {"estimator": (OI["vermillion"], "-", "o", "estimator"),
       "estimator_norest": (OI["vermillion"], "--", "X", "estimator (no rest)"),
       "rule": (OI["orange"], "-", "s", "rule"),
       "rule_norest": (OI["orange"], "--", "P", "rule (no rest)"),
       "gamma_argmax": (OI["blue"], "-", "^", r"$\Gamma$ argmax"),
       "gamma_learned": (OI["blue"], "--", "v", r"$\Gamma$ learned"),
       "gamma_restdrive": (OI["black"], "-", "d", r"$\Gamma$ drive-rest"),
       "gamma_oracle": (OI["green"], "-", "D", r"$\Gamma$ oracle"),
       "gamma_fixed": (OI["purple"], "-.", "P", r"$\Gamma$ constant"),
       "gamma_naive": (OI["skyblue"], ":", "*", r"$\Gamma$ naive")}
def g(a, k="gain"): return [S[v][a][k] for v in V]

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(r"Dose--response across environmental adversity ($\phi$)", fontsize=12, y=.99)
panels = [
    (ax[0][0], ["estimator", "estimator_norest", "rule", "gamma_argmax", "gamma_learned",
                "gamma_oracle"], "gain", "learning gain", "A. Performance vs adversity"),
    (ax[0][1], ["estimator_norest", "rule_norest", "gamma_restdrive", "estimator",
                "gamma_learned"], "drop", "dropout (%)", "B. Viability"),
    (ax[0][2], ["gamma_oracle", "gamma_learned", "gamma_fixed", "gamma_naive"],
     "gain", "learning gain", "C. Signal quality, fixed policy")]
for a, arms, key, ylab, title in panels:
    for k in arms:
        c, ls, m, l = STY[k]
        a.plot(X, g(k, key), ls, color=c, marker=m, label=l, lw=1.8, ms=5)
    a.set(xlabel=r"$\phi$", ylabel=ylab, title=title); a.legend(fontsize=7, frameon=False)

a = ax[1][0]
for k in ["estimator", "rule", "gamma_learned", "gamma_oracle"]:
    c, ls, m, l = STY[k]; b = S["0.0"][k]["gain"]
    a.plot(X, [100 * S[v][k]["gain"] / b for v in V], ls, color=c, marker=m, label=l, lw=1.8, ms=5)
a.axhline(100, color=OI["black"], ls=":", lw=1)
a.set(xlabel=r"$\phi$", ylabel=r"% of own $\phi{=}0$ gain", title="D. Relative robustness")
a.legend(fontsize=7, frameon=False)

a = ax[1][1]
for k in ["gamma_oracle", "gamma_argmax", "gamma_learned"]:
    c, ls, m, l = STY[k]
    a.plot(X, g(k, "g_val"), ls, color=c, marker=m, label=l, lw=1.8, ms=5)
a.set(xlabel=r"$\phi$", ylabel=r"$g$ validity", title="E. Signal learnability")
a.legend(fontsize=7, frameon=False)

a = ax[1][2]; w = .36; xs = np.arange(len(V))
a.bar(xs - w/2, g("gamma_learned"), w, label="softmax", color=OI["blue"])
a.bar(xs + w/2, g("gamma_argmax"), w, label="argmax", color=OI["skyblue"])
a.set_xticks(list(xs)); a.set_xticklabels(V)
a.set(xlabel=r"$\phi$", ylabel="learning gain", title="F. Policy: stochastic vs deterministic")
a.legend(fontsize=7, frameon=False); a.grid(axis="x", alpha=0)

plt.tight_layout(); plt.savefig(ROOT / "figures/fig_v4.png", bbox_inches="tight")
print("wrote figures/fig_v4.png")
