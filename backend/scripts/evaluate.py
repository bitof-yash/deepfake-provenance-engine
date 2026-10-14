"""Train a baseline on cached LAV-DF features and report honest test-split metrics.

Run from the backend folder (pip install scikit-learn joblib first):
  python -m scripts.evaluate --features ~/lavdf/features.jsonl --out ~/lavdf/results.json
Train on split=train only, report on split=test only. Never tune on test.
"""
import argparse
import json
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import fusion


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


META_IDX = [30, 34, 35, 36]   # speech fraction, lip availability, face_found, n_windows (see clipmodel.py)


def auc_ci(y, s, n=500, seed=0):
    """[AUC, 95% bootstrap low, high] over the test clips."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(auc(y, s), 3), round(float(lo), 3), round(float(hi), 3)]


def strat_auc(y, s, d, edges):
    """AUC computed inside duration bins, then averaged (weighted by bin size)."""
    vals, wts = [], []
    for b in range(len(edges) - 1):
        hi = (d <= edges[b + 1]) if b == len(edges) - 2 else (d < edges[b + 1])
        m = (d >= edges[b]) & hi
        if m.sum() >= 20 and len(set(y[m])) == 2:
            vals.append(roc_auc_score(y[m], s[m]))
            wts.append(m.sum())
    return float(np.average(vals, weights=wts)) if vals else float("nan")


def strat_auc_ci(y, s, d, nbins=6, n=300, seed=0):
    """[duration-controlled AUC, 95% low, high]; bins are test-duration quantiles."""
    y, s, d = np.asarray(y), np.asarray(s, dtype=float), np.asarray(d, dtype=float)
    edges = np.quantile(d, np.linspace(0, 1, nbins + 1))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = strat_auc(y[i], s[i], d[i], edges)
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return [float('nan')] * 3
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [round(strat_auc(y, s, d, edges), 3), round(float(lo), 3), round(float(hi), 3)]


def overlap(a0, a1, periods):
    return sum(max(0.0, min(a1, p1) - max(a0, p0)) for p0, p1 in periods)


def union_len(iv):
    iv = sorted(iv)
    tot, cur_end = 0.0, -1e9
    for s, e in iv:
        s = max(s, cur_end)
        if e > s:
            tot += e - s
            cur_end = e
    return tot


def seg_iou(pred, gt):
    inter = sum(overlap(s, e, gt) for s, e in pred)
    union = union_len(pred + gt)
    return inter / union if union > 0 else 0.0


def load(path):
    rows = [json.loads(l) for l in open(pathlib.Path(path).expanduser()) if l.strip()]
    bad = [r for r in rows if "error" in r or not r.get("clip_vector")]
    good = [r for r in rows if r not in bad]
    print(f"{len(rows)} clips, {len(bad)} unusable (errors / no clip vector)")
    return good


def window_metrics(rows):
    y, cols = [], {"fused": [], "video": [], "audio": [], "lipsync": []}
    for r in rows:
        for w in r["windows"]:
            y.append(int(overlap(w["start"], w["end"], r["fake_periods"]) > 0.1))
            for k in cols:
                cols[k].append(np.nan if w[k] is None else w[k])
    y = np.array(y)
    out = {}
    for k, v in cols.items():
        v = np.array(v)
        m = ~np.isnan(v)
        out[k] = auc(y[m], v[m]) if m.sum() else float("nan")
    return out, int(y.sum()), len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-model", default=None, help="save the logreg baseline here (e.g. models/baseline.joblib)")
    ap.add_argument("--drop-meta", action="store_true",
                    help="drop speech-fraction / availability / face-found / window-count columns (leakage check)")
    a = ap.parse_args()
    if a.drop_meta and a.save_model:
        raise SystemExit("--drop-meta is a diagnostic only; do not combine with --save-model")
    rows = load(a.features)
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    L = {len(r["clip_vector"]) for r in rows}
    assert len(L) == 1, f"clip vectors differ in length: {L}"
    Xtr, ytr = np.array([r["clip_vector"] for r in tr]), np.array([r["label"] for r in tr])
    Xte, yte = np.array([r["clip_vector"] for r in te]), np.array([r["label"] for r in te])
    if a.drop_meta:
        keep = [i for i in range(Xtr.shape[1]) if i not in META_IDX]
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        print(f"--drop-meta: removed columns {META_IDX}")
    print(f"train {len(tr)} (fake {ytr.sum()}), test {len(te)} (fake {yte.sum()}), vector dim {Xtr.shape[1]}")

    res = {"n_train": len(tr), "n_test": len(te), "clip_auc": {}, "clip_auc_by_category": {}}
    res["clip_auc"]["heuristic (no training)"] = auc_ci(yte, [r["heuristic_score"] for r in te])
    for mod in ("video", "audio", "lipsync"):
        res["clip_auc"][f"{mod} max-window"] = auc_ci(
            yte, [max([w[mod] for w in r["windows"] if w[mod] is not None], default=0.0) for r in te])
    models = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
              "grad_boost": HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05)}
    probs = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        probs[name] = m.predict_proba(Xte)[:, 1]
        res["clip_auc"][name] = auc_ci(yte, probs[name])
    dtr = np.array([[r["duration"]] for r in tr])
    dte = np.array([[r["duration"]] for r in te])
    dm = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(dtr, ytr)
    res["clip_auc"]["LEAKAGE CHECK: clip duration only"] = auc_ci(yte, dm.predict_proba(dte)[:, 1])
    dur_te = [r["duration"] for r in te]
    res["duration_controlled_auc"] = {
        "note": "AUC within duration bins (6 quantile bins), averaged. If a number here is ~0.5, "
                "its signal was clip length, not manipulation.",
        "heuristic": strat_auc_ci(yte, [r["heuristic_score"] for r in te], dur_te),
        "logreg": strat_auc_ci(yte, probs["logreg"], dur_te),
        "grad_boost": strat_auc_ci(yte, probs["grad_boost"], dur_te),
        "duration only (sanity, should be ~0.5)": strat_auc_ci(yte, dm.predict_proba(dte)[:, 1], dur_te),
    }
    if a.save_model:
        import joblib
        p = pathlib.Path(a.save_model).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(models["logreg"], p)
        print("saved baseline to", p)
    cats = sorted({r["category"] for r in te if r["category"] != "real"})
    for c in cats:
        idx = [i for i, r in enumerate(te) if r["category"] in ("real", c)]
        yy = yte[idx]
        res["clip_auc_by_category"][c] = {
            "heuristic": auc(yy, [te[i]["heuristic_score"] for i in idx]),
            **{n: auc(yy, probs[n][idx]) for n in probs}}

    wm, npos, ntot = window_metrics(te)
    res["window_auc"] = wm
    res["window_counts"] = {"fake_windows": npos, "all_windows": ntot}
    fakes = [r for r in te if r["label"] == 1]
    ious = [seg_iou([(s["start"], s["end"]) for s in fusion.segments_from_windows(r["windows"])],
                    [tuple(p) for p in r["fake_periods"]]) for r in fakes]
    res["localization"] = {"mean_iou_fake_clips": float(np.mean(ious)) if ious else None,
                           "frac_iou_ge_0.5": float(np.mean([i >= 0.5 for i in ious])) if ious else None,
                           "n_fake_clips": len(ious),
                           "window_resolution_s": 1.0}
    json.dump(res, open(pathlib.Path(a.out).expanduser(), "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\n0.5 AUC = coin flip. Report these numbers as they are.")


if __name__ == "__main__":
    main()
