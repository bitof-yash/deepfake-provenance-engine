"""Two checks that separate real deepfake detection from dataset shortcuts.

  python -m scripts.shortcut_checks --subset ~/lavdf_big/subset_small.json --emb ~/lavdf_norm/emb

1. Cross-category: train on real + one fake type, test on real + a DIFFERENT fake type.
   An audio model trained on fake_audio should score ~0.5 on fake_video clips (their audio is
   supposed to be genuine). A high score means it is detecting something other than fake audio.
2. Within-clip: for each fake clip, AUC of windows inside the fake segment vs windows outside it
   in the SAME clip. Nothing about the clip as a whole can help here, so it is the cleanest test.
"""
import argparse
import json
import pathlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.train_windows import feats, load, smooth

FAKES = ("fake_audio", "fake_video", "fake_av")


def fit(clips, modality, clipnorm, rng, cap=120000):
    X = np.concatenate([feats(c, modality, clipnorm) for c in clips])
    y = np.concatenate([c["y"] for c in clips])
    if len(y) > cap:
        keep = rng.choice(len(y), cap, replace=False)
        X, y = X[keep], y[keep]
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1, class_weight="balanced"))
    return m.fit(X, y)


def cross(train, test, tr_cat, te_cat, modality, clipnorm, rng):
    tr = [c for c in train if c["e"]["category"] in ("real", tr_cat)]
    te = [c for c in test if c["e"]["category"] in ("real", te_cat)]
    m = fit(tr, modality, clipnorm, rng)
    s = [float(smooth(m.predict_proba(feats(c, modality, clipnorm))[:, 1]).max()) for c in te]
    return round(float(roc_auc_score([c["label"] for c in te], s)), 3)


def within(model, test, modality, clipnorm):
    out = {}
    for cat in FAKES:
        aucs = []
        for c in test:
            if c["e"]["category"] == cat and 0 < c["y"].sum() < len(c["y"]):
                aucs.append(roc_auc_score(c["y"], model.predict_proba(feats(c, modality, clipnorm))[:, 1]))
        out[cat] = {"mean_auc": round(float(np.mean(aucs)), 3) if aucs else None, "clips": len(aucs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    train = load([e for e in entries if e["split"] == "train"], a.emb)
    test = load([e for e in entries if e["split"] == "test"], a.emb)
    print(f"{len(train)} train clips, {len(test)} test clips")
    rng = np.random.default_rng(0)
    res = {"cross": {}, "within": {}}

    print("\n1. CROSS-CATEGORY clip AUC (train on real + A, test on real + B). ~0.5 = nothing transfers")
    for modality in ("audio", "video"):
        for clipnorm in (False, True):
            for tr_cat, te_cat in (("fake_audio", "fake_video"), ("fake_video", "fake_audio")):
                name = f"{modality}{'+clipnorm' if clipnorm else ''}: train {tr_cat} -> test {te_cat}"
                res["cross"][name] = cross(train, test, tr_cat, te_cat, modality, clipnorm, rng)
                print(f"  {name:<60} {res['cross'][name]}")

    print("\n2. WITHIN-CLIP window AUC, fake windows vs real windows of the same clip (clip-normalised)")
    for modality in ("audio", "video"):
        m = fit(train, modality, True, rng)
        res["within"][modality] = within(m, test, modality, True)
        for cat, r in res["within"][modality].items():
            print(f"  {modality:<6} model on {cat:<11} AUC {r['mean_auc']}  ({r['clips']} clips)")

    if a.out:
        json.dump(res, open(pathlib.Path(a.out).expanduser(), "w"), indent=2)
        print("\nsaved", a.out)


if __name__ == "__main__":
    main()
