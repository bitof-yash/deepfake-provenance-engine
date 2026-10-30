"""End-to-end check of the live pipeline on real LAV-DF test clips: AUC, error rates at the routing
thresholds, and seconds per clip.

  python -m scripts.smoke_embed_api --n 30
  python -m scripts.smoke_embed_api --n 30 --root ~/lavdf_norm/LAV-DF     # re-encoded clips
Labels are by AUDIO: real = 0, fake_audio / fake_av = 1 (fake_video is skipped: its audio is genuine).
"""
import argparse
import json
import pathlib
import random
import statistics
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from app import config, pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="~/lavdf_big/subset_small.json")
    ap.add_argument("--root", default="~/lavdf_big/LAV-DF")
    ap.add_argument("--n", type=int, default=30, help="clips per class")
    a = ap.parse_args()
    root = pathlib.Path(a.root).expanduser()
    entries = [e for e in json.load(open(pathlib.Path(a.subset).expanduser())) if e["split"] == "test"]
    rng, picks = random.Random(0), []
    for label, cats in ((0, ("real",)), (1, ("fake_audio", "fake_av"))):
        pool = [e for e in entries if e["category"] in cats and (root / e["file"]).exists()]
        rng.shuffle(pool)
        picks += [(label, e) for e in pool[: a.n]]
    print(f"{len(picks)} clips from {root}")

    y, s, times, used = [], [], [], 0
    for label, e in picks:
        t0 = time.time()
        r = pipeline.analyze_file(str(root / e["file"]), pathlib.Path(e["file"]).name)
        times.append(time.time() - t0)
        y.append(label)
        s.append(r["detection_score"])
        used += any("learned wav2vec2" in n for n in r["notes"])
    y, s = np.array(y), np.array(s)
    t = times[1:] or times      # first clip includes model loading
    print(f"learned detector used on {used}/{len(y)} clips")
    print(f"AUC {roc_auc_score(y, s):.3f}   (real={int((y == 0).sum())}, fake={int((y == 1).sum())})")
    for name, thr in (("suspect", config.SUSPECT_THRESHOLD), ("fake", config.FAKE_THRESHOLD)):
        print(f"  at {name} threshold {thr}: real flagged {np.mean(s[y == 0] >= thr):.0%}, "
              f"fake flagged {np.mean(s[y == 1] >= thr):.0%}")
    print(f"seconds per clip (excl. first): mean {statistics.mean(t):.2f}, median {statistics.median(t):.2f}, "
          f"max {max(t):.2f}   | first clip (incl. model load) {times[0]:.2f}")


if __name__ == "__main__":
    main()
