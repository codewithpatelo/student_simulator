"""Hash the preregistered predictions BEFORE any experiment runs (paper Sec 5.3).

The digest is published in the paper. Anyone can recompute it from predictions.txt.
Self-attested commitment device, not a third-party registry: it makes post-hoc
editing detectable, nothing more.
"""
import hashlib, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
PUBLISHED = "6e888c7e46e1dd306b6adc601a45cc08ff914a24d2f474a913a1ff2907bfc084"
text = (ROOT / "predictions.txt").read_text().rstrip("\n")
digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
print("SHA-256 of predictions.txt:", digest)
print("published in the paper    :", PUBLISHED)
print("MATCH" if digest == PUBLISHED else "MISMATCH -- file edited since publication")
sys.exit(0 if digest == PUBLISHED else 1)
