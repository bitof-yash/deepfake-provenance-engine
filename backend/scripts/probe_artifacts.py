"""Look for dataset shortcuts: can plain file/stream metadata tell real from fake?"""
import collections, json, os, pathlib, subprocess, sys
import numpy as np
from sklearn.metrics import roc_auc_score

subset, root = (pathlib.Path(p).expanduser() for p in sys.argv[1:3])
entries = [e for e in json.load(open(subset)) if e["split"] == "test"]
import random; random.Random(0).shuffle(entries); entries = entries[:800]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def fps(r):
    try:
        a, b = r.split("/")
        return float(a) / float(b)
    except Exception:
        return np.nan


rows = []
for e in entries:
    p = str(root / e["file"])
    d = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", p],
                                  capture_output=True, text=True).stdout or "{}")
    v = next((s for s in d.get("streams", []) if s["codec_type"] == "video"), {})
    a = next((s for s in d.get("streams", []) if s["codec_type"] == "audio"), {})
    rows.append(dict(
        cat=e["category"], label=int(e["n_fakes"] > 0),
        num=dict(a_bitrate=num(a.get("bit_rate")), v_bitrate=num(v.get("bit_rate")),
                 a_dur=num(a.get("duration")), v_dur=num(v.get("duration")),
                 dur_gap=num(a.get("duration")) - num(v.get("duration")),
                 a_start=num(a.get("start_time")), v_start=num(v.get("start_time")),
                 a_sr=num(a.get("sample_rate")), fps=fps(v.get("r_frame_rate", "")),
                 width=num(v.get("width")), height=num(v.get("height")),
                 file_size=os.path.getsize(p), size_per_sec=os.path.getsize(p) / max(e["duration"], 0.1),
                 nb_frames=num(v.get("nb_frames")), clip_dur=e["duration"]),
        cat_vals=dict(a_codec=a.get("codec_name"), a_profile=a.get("profile"), v_codec=v.get("codec_name"),
                      v_profile=v.get("profile"), pix_fmt=v.get("pix_fmt"), a_layout=a.get("channel_layout"))))

y = np.array([r["label"] for r in rows])
print(f"{len(rows)} test clips ({int(y.sum())} fake). AUC of each plain metadata number (0.5 = no signal):")
out = []
for k in rows[0]["num"]:
    x = np.array([r["num"][k] for r in rows])
    ok = ~np.isnan(x)
    if ok.sum() > 50 and len(set(y[ok])) == 2 and np.nanstd(x) > 0:
        out.append((abs(roc_auc_score(y[ok], x[ok]) - 0.5), k, roc_auc_score(y[ok], x[ok])))
for _, k, a in sorted(out, reverse=True):
    per = {}
    for c in ("fake_audio", "fake_video", "fake_av"):
        idx = [i for i, r in enumerate(rows) if r["cat"] in ("real", c)]
        xx = np.array([rows[i]["num"][k] for i in idx])
        yy = y[idx]
        ok = ~np.isnan(xx)
        per[c] = round(float(roc_auc_score(yy[ok], xx[ok])), 2) if len(set(yy[ok])) == 2 else None
    print(f"  {k:<14} AUC {a:.2f}   by category: {per}")
print("\nCategorical stream properties by category:")
for k in rows[0]["cat_vals"]:
    t = collections.defaultdict(collections.Counter)
    for r in rows:
        t[r["cat"]][str(r["cat_vals"][k])] += 1
    print(f"  {k}: " + "; ".join(f"{c}: {dict(v)}" for c, v in sorted(t.items())))
