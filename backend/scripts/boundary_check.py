"""Does the audio model's score on fake_video clips peak at segment BOUNDARIES (splice artifact)
or stay high across the whole segment (content effect)?
  python -m scripts.boundary_check --subset ~/lavdf_big/subset_small.json --emb ~/lavdf_norm/emb
"""
import argparse, json, pathlib
import numpy as np
from scripts.shortcut_checks import fit
from scripts.train_windows import feats, load

W = 0.25
BINS = [-2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2]  # seconds; negative = outside the fake segment


def signed_dist(t, periods):
    best = None
    for s, e in periods:
        d = min(t - s, e - t) if s <= t <= e else -min(abs(t - s), abs(t - e))
        if best is None or abs(d) < abs(best):
            best = d
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--emb", required=True)
    a = ap.parse_args()
    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    train = load([e for e in entries if e["split"] == "train"], a.emb)
    test = load([e for e in entries if e["split"] == "test"], a.emb)
    rng = np.random.default_rng(0)
    m = fit(train, "audio", True, rng)
    for cat in ("fake_video", "fake_audio"):
        acc = {i: [] for i in range(len(BINS) - 1)}
        for c in test:
            if c["e"]["category"] != cat:
                continue
            p = m.predict_proba(feats(c, "audio", True))[:, 1]
            for i, s in enumerate(p):
                d = signed_dist((i + 0.5) * W, c["e"]["fake_periods"])
                for b in range(len(BINS) - 1):
                    if BINS[b] <= d < BINS[b + 1]:
                        acc[b].append(s)
        print(f"\n{cat}: mean audio score by distance to nearest segment edge (negative = outside)")
        for b, v in acc.items():
            if v:
                print(f"  {BINS[b]:>5} to {BINS[b+1]:>5} s   {np.mean(v):.3f}   (n={len(v)})")


if __name__ == "__main__":
    main()
