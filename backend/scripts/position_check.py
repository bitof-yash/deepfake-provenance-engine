"""Is the within-clip result just 'fake segments sit in the middle of the clip'?

  python -m scripts.position_check --subset ~/lavdf_big/subset_small.json --emb ~/lavdf_norm/emb

1. Position-only baseline: within-clip AUC of distance-from-clip-edge. No audio or video content used.
2. Re-run the within-clip test using only interior windows (first and last --edge windows dropped,
   for both training and testing), so edge effects cannot help.
"""
import argparse
import json
import pathlib

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.shortcut_checks import FAKES, fit, within
from scripts.train_windows import load


def position_auc(test):
    out = {}
    for cat in FAKES:
        aucs = []
        for c in test:
            n = len(c["y"])
            if c["e"]["category"] == cat and 0 < c["y"].sum() < n:
                d = np.minimum(np.arange(n), n - 1 - np.arange(n))
                aucs.append(roc_auc_score(c["y"], d))
        out[cat] = round(float(np.mean(aucs)), 3) if aucs else None
    return out


def interior(clips, k):
    out = []
    for c in clips:
        n = len(c["y"])
        if n > 2 * k + 4:
            out.append(dict(c, a=c["a"][k:n - k], v=c["v"][k:n - k], y=c["y"][k:n - k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--edge", type=int, default=4, help="windows dropped at each end (4 = 1 s)")
    a = ap.parse_args()
    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    train = load([e for e in entries if e["split"] == "train"], a.emb)
    test = load([e for e in entries if e["split"] == "test"], a.emb)
    rng = np.random.default_rng(0)

    print("1. POSITION-ONLY within-clip AUC (distance from clip edge; 0.5 = position tells nothing)")
    for cat, v in position_auc(test).items():
        print(f"  {cat:<11} {v}")

    tr_i, te_i = interior(train, a.edge), interior(test, a.edge)
    print(f"\n2. WITHIN-CLIP AUC, interior windows only (drop {a.edge} at each end; {len(te_i)} test clips)")
    for modality in ("audio", "video"):
        m = fit(tr_i, modality, True, rng)
        for cat, r in within(m, te_i, modality, True).items():
            print(f"  {modality:<6} model on {cat:<11} AUC {r['mean_auc']}  ({r['clips']} clips)")


if __name__ == "__main__":
    main()
