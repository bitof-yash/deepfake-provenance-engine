"""Re-score saved FakeAVCeleb predictions on confound-controlled subsets.
Run scripts.fakeav_meta_baseline first (it writes the metadata cache this joins against)."""
import json, os, pathlib
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = pathlib.Path("~/lavdf_transfer").expanduser()
ROOT = pathlib.Path("~/fakeavceleb/FakeAVCeleb_v1.2").expanduser()
MODELS = ["audio", "audio_clipnorm", "both", "both_clipnorm"]


def auc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def strat(y, s, groups, min_n=10):
    """AUC inside each group (needs >= min_n of each class), averaged weighted by group size."""
    vals, wts = [], []
    for g in set(groups):
        m = groups == g
        pos = int(y[m].sum())
        if pos >= min_n and m.sum() - pos >= min_n:
            vals.append(roc_auc_score(y[m], s[m]))
            wts.append(m.sum())
    return (float(np.average(vals, weights=wts)), len(vals), int(sum(wts))) if vals else (float("nan"), 0, 0)


def main():
    meta = [json.loads(l) for l in open(OUT / "fakeav_meta.jsonl") if l.strip()]
    meta = [m for m in meta if "error" not in m]
    by_file = {m["file"]: m for m in meta}
    by_key = {m["key"]: m for m in meta}

    def resolve(f):
        f = str(f)
        if f.startswith(str(ROOT)):
            f = os.path.relpath(f, ROOT)
        f = f.removeprefix("FakeAVCeleb/")
        return by_file.get(f) or by_key.get(f) or by_key.get(pathlib.Path(f).stem)

    preds = json.load(open(OUT / "fakeav_predictions.json"))
    for name in MODELS:
        if name not in preds:
            continue
        P = preds[name]
        ms = [resolve(f) for f in P["files"]]
        ok = np.array([m is not None for m in ms])
        print(f"\n{'=' * 70}\n{name}: matched {ok.sum()}/{len(ok)} predictions to metadata")
        if ok.sum() < 0.9 * len(ok):
            print("  too few matched - check what P['files'] looks like and fix resolve()")
            continue
        rows = [m for m in ms if m]
        s = np.array(P["scores"])[ok]
        y = np.array(P["labels_audio"])[ok]
        typ = np.array([m["type"] for m in rows])
        meth = np.array([m["method"] for m in rows])
        ch = np.array([m["channels"] for m in rows])
        br = np.array([m["a_bitrate"] for m in rows], float)

        print(f"  ALL clips                                   AUC {auc(y, s):.3f}   (n={len(y)})")
        fv = np.char.startswith(typ, "FakeVideo")
        print(f"  FakeVideo only (FVFA vs FVRA)               AUC {auc(y[fv], s[fv]):.3f}   (n={fv.sum()})")

        # within-channel
        for c in sorted(set(ch[fv])):
            m = fv & (ch == c)
            print(f"    channels={int(c)}                             AUC {auc(y[m], s[m]):.3f}   "
                  f"(pos={int(y[m].sum())}, neg={int(m.sum() - y[m].sum())})")

        # bitrate bins (quantiles of FakeVideo subset), then channels x bitrate
        b = np.where(np.isnan(br), np.nanmedian(br[fv]), br)
        edges = np.quantile(b[fv], np.linspace(0, 1, 7)[1:-1])
        bbin = np.digitize(b, edges)
        a1, k1, n1 = strat(y[fv], s[fv], bbin[fv].astype(str))
        print(f"  FakeVideo, within bitrate bins              AUC {a1:.3f}   ({k1} bins, n={n1})")
        grp = np.array([f"{c}|{x}" for c, x in zip(ch, bbin)])
        a2, k2, n2 = strat(y[fv], s[fv], grp[fv])
        print(f"  FakeVideo, within channels x bitrate bins   AUC {a2:.3f}   ({k2} cells, n={n2})   <- headline")

        # per method (positives of one method vs real-audio negatives of the same video type)
        print("  per method (pos = that method, neg = real-audio clips of same video type):")
        for mth in sorted(set(meth[y == 1])):
            pos = (y == 1) & (meth == mth)
            vtype = "FakeVideo" if typ[pos][0].startswith("FakeVideo") else "RealVideo"
            neg = (y == 0) & np.char.startswith(typ, vtype)
            m = pos | neg
            print(f"    {mth:<18} vs {vtype + '-RealAudio':<20} AUC {auc(y[m], s[m]):.3f}   "
                  f"(pos={int(pos.sum())}, neg={int(neg.sum())})")


if __name__ == "__main__":
    main()
