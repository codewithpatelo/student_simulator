"""Reproduce every number in the paper, in order.

    python run_all.py            # ~10 min: everything except the 1200-candidate search
    python run_all.py --full     # ~35 min: adds the large-budget tuning sweep

Each step prints the paper numbers it produces and writes JSON to results/.
"""
import argparse, pathlib, subprocess, sys, time
ROOT = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--full", action="store_true")
args = ap.parse_args()
STEPS = [
    ("00_predictions.py", [], "commitment hash"),
    ("01_quality_gate.py", [], "quality gate per sweep point"),
    ("02_main_sweep.py", [], "main sweep + P2/P3/P4/P5 contrasts"),
    ("04_matched_tuning.py", ["--budget", "250"], "matched tuning, 250 candidates"),
    ("03_decomposition.py", [], "signal/policy/interaction/residual"),
    ("05_ablations.py", [], "objective-model, viability, rest-gate"),
    ("06_shape_shift.py", [], "model misspecification (P7)"),
    ("07_permutation_p3.py", [], "multiplicity-corrected crossover"),
    ("08_reconstruction.py", [], "how private is the agent's state"),
    ("09_maintenance.py", [], "maintenance objective (sixth condition)"),
    ("10_followups.py", [], "coupling, level-aware channel, learned mapping, rest"),
]
if args.full:
    STEPS.insert(4, ("04_matched_tuning.py",
                     ["--budget", "1200", "--search-seeds", "101", "211", "307"],
                     "matched tuning, 1200 candidates"))
for script, extra, desc in STEPS:
    print("\n" + "=" * 78); print(f"  {script} -- {desc}"); print("=" * 78, flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "experiments" / script)] + extra)
    if r.returncode: sys.exit(f"FAILED: {script}")
    print(f"[{time.time()-t:.1f}s]")
print("\nAll steps completed. Figure: python figures/make_figure.py")
