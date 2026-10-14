"""Within-clip test: windows deep inside a fake segment vs windows far outside it, same clip.
Same speaker, mic, room and encoding for both classes, so speaker leakage cannot help."""
import argparse, json, pathlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.train_windows import feats, load
from scripts.boundary_check import signed_dist

W = 0.25


def split_windows(c, margin, gap):
    n = len(c["y"])
    d = np.array([signed_dist((i + 0.5) * W, c["e"]["fake_periods"]) for i in range(n)])
    return np.where(d >= margin)[0], np.where(d <= -gap)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--gap", type=float, default=0.5)
    a = ap.parse_args()
    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    train = load([e for e in entries if e["split"] == "train"], a.emb)
    test = load([e for e in entries if e["split"] == "test"], a.emb)
    print(f"pos: >= {a.margin}s inside segment; neg: >= {a.gap}s outside, same clip")
    for modality in ("audio", "video"):
        for clipnorm in (False, True):
            for cat in ("fake_audio", "fake_video", "fake_av"):
                X, y = [], []
                for c in train:
                    if c["e"]["category"] != cat:
                        continue
                    pos, neg = split_windows(c, a.margin, a.gap)
                    if len(pos) and len(neg):
                        F = feats(c, modality, clipnorm)
                        X += [F[pos], F[neg]]
                        y += [1] * len(pos) + [0] * len(neg)
                if len(set(y)) < 2:
                    print(f"  {modality} {cat}: not enough train data")
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1, class_weight="balanced"))
                m.fit(np.concatenate(X), np.array(y))
                aucs = []
                for c in test:
                    if c["e"]["category"] != cat:
                        continue
                    pos, neg = split_windows(c, a.margin, a.gap)
                    if len(pos) and len(neg):
                        F = feats(c, modality, clipnorm)
                        idx = np.concatenate([pos, neg])
                        lab = np.array([1] * len(pos) + [0] * len(neg))
                        aucs.append(roc_auc_score(lab, m.predict_proba(F[idx])[:, 1]))
                tag = "clipnorm" if clipnorm else "raw"
                if aucs:
                    print(f"  {modality:<5} {tag:<8} {cat:<11} mean per-clip AUC {np.mean(aucs):.3f} ({len(aucs)} test clips)")
                else:
                    print(f"  {modality:<5} {tag:<8} {cat:<11} no test clips qualify")


if __name__ == "__main__":
    main()
