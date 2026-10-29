"""Can plain stream metadata separate fake-audio from real-audio clips in FakeAVCeleb?"""
import collections, json, os, pathlib, random, subprocess
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path("~/fakeavceleb/FakeAVCeleb_v1.2").expanduser()
recs = json.load(open(pathlib.Path("~/lavdf_transfer/fakeav_manifest.json").expanduser()))
random.Random(0).shuffle(recs)
by = collections.defaultdict(list)
for r in recs:
    by[r["type"]].append(r)
sample = [r for t in by for r in by[t][:400]]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


rows = []
for r in sample:
    p = ROOT / r["file"]
    d = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
                                  capture_output=True, text=True).stdout or "{}")
    v = next((s for s in d.get("streams", []) if s["codec_type"] == "video"), {})
    a = next((s for s in d.get("streams", []) if s["codec_type"] == "audio"), {})
    rows.append(dict(type=r["type"], fake_audio=int("FakeAudio" in r["type"]),
        num=dict(a_sr=num(a.get("sample_rate")), a_bitrate=num(a.get("bit_rate")), a_dur=num(a.get("duration")),
                 v_dur=num(v.get("duration")), dur_gap=num(a.get("duration")) - num(v.get("duration")),
                 v_bitrate=num(v.get("bit_rate")), width=num(v.get("width")),
                 size_per_s=os.path.getsize(p) / max(num(d.get("format", {}).get("duration")), 0.1)),
        cat=dict(a_codec=a.get("codec_name"), channels=a.get("channels"), v_codec=v.get("codec_name"),
                 pix_fmt=v.get("pix_fmt"))))


def report(name, sub):
    y = np.array([x["fake_audio"] for x in sub])
    print(f"\n{name}: {len(sub)} clips, {int(y.sum())} fake-audio")
    for k in sub[0]["num"]:
        s = np.array([x["num"][k] for x in sub]); ok = ~np.isnan(s)
        if ok.sum() > 50 and len(set(y[ok])) == 2 and np.nanstd(s) > 0:
            print(f"  {k:<11} AUC {roc_auc_score(y[ok], s[ok]):.2f}")
    for k in sub[0]["cat"]:
        t = collections.defaultdict(collections.Counter)
        for x in sub:
            t[x["type"]][str(x["cat"][k])] += 1
        print(f"  {k}: " + "; ".join(f"{a}: {dict(b)}" for a, b in sorted(t.items())))


report("ALL", rows)
report("FakeVideo only (controls the video pipeline)", [x for x in rows if x["type"].startswith("FakeVideo")])
