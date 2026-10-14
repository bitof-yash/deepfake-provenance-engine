"""Best possible localization IoU for a given window size, assuming a PERFECT detector."""
import json, pathlib, sys
import numpy as np


def overlap(a0, a1, ps):
    return sum(max(0.0, min(a1, p1) - max(a0, p0)) for p0, p1 in ps)


def union_len(iv):
    iv, tot, end = sorted(iv), 0.0, -1e9
    for s, e in iv:
        s = max(s, end)
        if e > s:
            tot += e - s
            end = e
    return tot


def iou(pred, gt):
    inter = sum(overlap(s, e, gt) for s, e in pred)
    u = union_len(pred + gt)
    return inter / u if u else 0.0


def run(clips, w):
    res = {"any_overlap": [], "majority_overlap": []}
    for c in clips:
        gt = [tuple(p) for p in c["fake_periods"]]
        n = int(np.ceil(c["duration"] / w))
        wins = [(i * w, min((i + 1) * w, c["duration"])) for i in range(n)]
        res["any_overlap"].append(iou([x for x in wins if overlap(*x, gt) > 1e-6], gt))
        res["majority_overlap"].append(iou([x for x in wins if overlap(*x, gt) >= 0.5 * (x[1] - x[0])], gt))
    return {k: (float(np.mean(v)), float(np.mean(np.array(v) >= 0.5))) for k, v in res.items()}


clips = [c for c in json.load(open(pathlib.Path(sys.argv[1]).expanduser())) if c["n_fakes"] > 0 and c["split"] == "test"]
lens = [p1 - p0 for c in clips for p0, p1 in c["fake_periods"]]
print(f"{len(clips)} fake test clips; fake segment length: median {np.median(lens):.2f}s, "
      f"{100*np.mean(np.array(lens) < 1.0):.0f}% shorter than 1 s")
print("Perfect-detector ceiling (mean IoU, fraction with IoU>=0.5):")
for w in (1.0, 0.5, 0.25):
    for k, (m, f) in run(clips, w).items():
        print(f"  window {w:>4}s  {k:<17} IoU {m:.2f}   >=0.5: {f:.0%}")
