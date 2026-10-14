"""Train and evaluate a per-window detector on the cached embeddings (train split -> test split).

  python -m scripts.train_windows --subset ~/lavdf_big/subset.json --emb ~/lavdf_big/emb \
         --out ~/lavdf_big/results_windows.json

Window label = fake if >=50% of the window overlaps a labelled fake segment. Fits audio-only,
video-only and both, each raw and clip-normalised (clip mean subtracted, which removes the
clip-level differences such as encoding/length that leaked in the handcrafted baseline).
The localisation threshold is chosen on held-out TRAIN clips, never on the test split.
"""
import argparse
import json
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.evaluate import auc, auc_ci, seg_iou, strat_auc_ci


def overlap(a0, a1, ps):
    return sum(max(0.0, min(a1, p1) - max(a0, p0)) for p0, p1 in ps)


def load(entries, emb_dir):
    clips = []
    for e in entries:
        p = pathlib.Path(emb_dir) / e["split"] / (pathlib.Path(e["file"]).stem + ".npz")
        if not p.exists():
            continue
        d = np.load(p)
        w = float(d["window"])
        n = len(d["a"])
        gt = [tuple(x) for x in e["fake_periods"]]
        y = np.array([int(overlap(i * w, (i + 1) * w, gt) >= 0.5 * w) for i in range(n)])
        clips.append(dict(e=e, a=d["a"].astype(np.float32), v=d["v"].astype(np.float32), y=y, w=w, gt=gt,
                          label=int(e["n_fakes"] > 0)))
    return clips


def feats(c, modality, clipnorm):
    parts = []
    for key in (("a",) if modality == "audio" else ("v",) if modality == "video" else ("a", "v")):
        x = c[key]
        parts.append(x - x.mean(axis=0, keepdims=True) if clipnorm else x)
    return np.concatenate(parts, axis=1)


def smooth(p, k=3):
    return np.convolve(p, np.ones(k) / k, mode="same") if len(p) >= k else p


def segments(p, thr, w):
    out, start = [], None
    for i, flag in enumerate(list(p > thr) + [False]):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            out.append((start * w, i * w))
            start = None
    return out


def mean_iou(clips, probs, thr):
    ious = [seg_iou(segments(p, thr, c["w"]), c["gt"]) for c, p in zip(clips, probs) if c["label"]]
    return float(np.mean(ious)), float(np.mean([i >= 0.5 for i in ious]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-train-windows", type=int, default=120000)
    a = ap.parse_args()

    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    train = load([e for e in entries if e["split"] == "train"], a.emb)
    test = load([e for e in entries if e["split"] == "test"], a.emb)
    print(f"{len(train)} train clips, {len(test)} test clips with embeddings")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(train))
    n_val = max(1, len(train) // 5)
    val, fit = [train[i] for i in idx[:n_val]], [train[i] for i in idx[n_val:]]
    yte_clip = np.array([c["label"] for c in test])
    dur_te = [c["e"]["duration"] for c in test]
    results = {}

    for modality in ("audio", "video", "both"):
        for clipnorm in (False, True):
            name = f"{modality}{'+clipnorm' if clipnorm else ''}"
            X = np.concatenate([feats(c, modality, clipnorm) for c in fit])
            y = np.concatenate([c["y"] for c in fit])
            if len(y) > a.max_train_windows:
                keep = rng.choice(len(y), a.max_train_windows, replace=False)
                X, y = X[keep], y[keep]
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1, class_weight="balanced"))
            model.fit(X, y)

            def predict(cs):
                return [smooth(model.predict_proba(feats(c, modality, clipnorm))[:, 1]) for c in cs]

            pv, pt = predict(val), predict(test)
            thr = max(np.linspace(0.3, 0.9, 25), key=lambda t: mean_iou(val, pv, t)[0])   # val only
            yw = np.concatenate([c["y"] for c in test])
            pw = np.concatenate(pt)
            clip_score = [float(p.max()) for p in pt]
            iou, f05 = mean_iou(test, pt, thr)
            cats = {}
            for cat in sorted({c["e"]["category"] for c in test if c["e"]["category"] != "real"}):
                ii = [i for i, c in enumerate(test) if c["e"]["category"] in ("real", cat)]
                cats[cat] = round(auc(yte_clip[ii], np.array(clip_score)[ii]), 3)
            results[name] = {
                "window_auc": round(auc(yw, pw), 3),
                "clip_auc": auc_ci(yte_clip, clip_score),
                "clip_auc_duration_controlled": strat_auc_ci(yte_clip, clip_score, dur_te),
                "clip_auc_by_category": cats,
                "localization": {"mean_iou": round(iou, 3), "frac_iou_ge_0.5": round(f05, 3),
                                 "threshold": round(float(thr), 2), "window_s": test[0]["w"] if test else None},
            }
            print(name, json.dumps(results[name]))
    json.dump(results, open(pathlib.Path(a.out).expanduser(), "w"), indent=2)
    print("\nsaved", a.out, "- compare against the handcrafted baseline (clip AUC 0.50, IoU 0.098).")


if __name__ == "__main__":
    main()
