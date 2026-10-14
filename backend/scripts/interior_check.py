"""Is there detector signal away from the edit points?
Positives: windows >= --margin s INSIDE a fake segment. Negatives: windows from real clips.
Windows near or outside segment edges in fake clips are dropped from training and testing.
fake_video is the control: if the model is mostly a splice detector, it should drop toward 0.5 there.
  python -m scripts.interior_check --subset ~/lavdf_big/subset_small.json --emb ~/lavdf_norm/emb
"""
import argparse, json, pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.train_windows import feats, load

W = 0.25


def inside_depth(t, periods):
    ds = [min(t - s, e - t) for s, e in periods if s <= t <= e]
    return max(ds) if ds else None


def build(clips, modality, clipnorm, margin, cat_keep):
    X, y = [], []
    for c in clips:
        cat = c["e"]["category"]
        if cat != "real" and cat != cat_keep:
            continue
        F = feats(c, modality, clipnorm)
        if cat == "real":
            X.append(F); y += [0] * len(F)
        else:
            keep = []
            for i in range(len(F)):
                d = inside_depth((i + 0.5) * W, c["e"]["fake_periods"])
                if d is not None and d >= margin:
                    keep.append(i)
            if keep:
                X.append(F[keep]); y += [1] * len(keep)
    return np.concatenate(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--margin", type=float, default=0.25)
    a = ap.parse_args()
    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    train = load([e for e in entries if e["split"] == "train"], a.emb)
    test = load([e for e in entries if e["split"] == "test"], a.emb)
    rng = np.random.default_rng(0)
    print(f"margin = {a.margin}s inside a segment; negatives = real-clip windows")
    for modality in ("audio", "video"):
        for clipnorm in (False, True):
            for cat in ("fake_audio", "fake_video", "fake_av"):
                Xtr, ytr = build(train, modality, clipnorm, a.margin, cat)
                Xte, yte = build(test, modality, clipnorm, a.margin, cat)
                if len(set(yte)) < 2 or len(set(ytr)) < 2:
                    print(f"  {modality} {cat}: not enough data"); continue
                if len(ytr) > 120000:
                    k = rng.choice(len(ytr), 120000, replace=False); Xtr, ytr = Xtr[k], ytr[k]
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1, class_weight="balanced"))
                m.fit(Xtr, ytr)
                auc = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
                print(f"  {modality:<5} {'clipnorm' if clipnorm else 'raw     '} {cat:<11} "
                      f"AUC {auc:.3f}  (test pos {int(yte.sum())}, neg {int((yte==0).sum())})")


if __name__ == "__main__":
    main()
