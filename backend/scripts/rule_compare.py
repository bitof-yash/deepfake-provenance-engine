"""Compare clip-score rules (max / top20 / mean, all after 3-window smoothing) on LAV-DF test and FakeAVCeleb.
Thresholds are chosen on FakeAVCeleb speaker-half A only."""
import json, math, pathlib, zlib
import numpy as np, joblib
from sklearn.metrics import roc_auc_score, roc_curve

H = pathlib.Path.home()
model = joblib.load(pathlib.Path(__file__).resolve().parent.parent / "models" / "audio_window.joblib")
RULES = ("max", "top20", "mean")


def scores(a):
    p = model.predict_proba(a)[:, 1]
    if len(p) >= 3:
        p = np.convolve(p, np.ones(3) / 3, mode="same")
    q = np.sort(p)[::-1]
    k = max(1, math.ceil(0.2 * len(q)))
    return {"max": q[0], "top20": q[:k].mean(), "mean": p.mean()}


# LAV-DF test split (label by audio: real=0, fake_audio/fake_av=1; fake_video skipped, its audio is genuine)
ly, lcat, ls = [], [], {r: [] for r in RULES}
for e in json.load(open(H / "lavdf_big" / "subset.json")):
    if e["split"] != "test" or e["category"] not in ("real", "fake_audio", "fake_av"):
        continue
    f = H / "lavdf_big" / "emb" / "test" / (pathlib.Path(e["file"]).stem + ".npz")
    if not f.exists():
        continue
    sc = scores(np.load(f)["a"].astype(np.float32))
    ly.append(int(e["category"] != "real")); lcat.append(e["category"])
    for r in RULES:
        ls[r].append(sc[r])
ly, lcat = np.array(ly), np.array(lcat)
ls = {r: np.array(v) for r, v in ls.items()}

# FakeAVCeleb
T = H / "lavdf_transfer"
fy, spk, fs = [], [], {r: [] for r in RULES}
for r_ in json.load(open(T / "fakeav_manifest.json")):
    f = T / "fakeav_embeddings" / f"{r_['key']}.npz"
    if not f.exists():
        continue
    with np.load(f) as d:
        a, has = np.asarray(d["a"], np.float32), bool(np.asarray(d["has_audio"]).item())
    if not has or a.ndim != 2 or len(a) == 0:
        continue
    sc = scores(a)
    fy.append(int("fakeaudio" in r_["type"].lower().replace("_", "-").replace(" ", "")))
    spk.append(r_["file"].split("/")[-2])
    for r in RULES:
        fs[r].append(sc[r])
fy = np.array(fy); fs = {r: np.array(v) for r, v in fs.items()}
half = np.array([zlib.crc32(x.encode()) % 2 for x in spk]); A, B = half == 0, half == 1
print(f"LAV-DF test: {len(ly)} clips (real={int((ly==0).sum())}, fake={int(ly.sum())}) | FakeAVCeleb: {len(fy)} clips")

print("\nAUC by rule")
print(f"{'rule':6} {'LAV-DF':>8} {'LAV fake_audio':>15} {'LAV fake_av':>12} {'FakeAV':>8}")
for r in RULES:
    s = ls[r]
    m_a = (lcat == "real") | (lcat == "fake_audio"); m_v = (lcat == "real") | (lcat == "fake_av")
    print(f"{r:6} {roc_auc_score(ly, s):8.3f} {roc_auc_score(ly[m_a], s[m_a]):15.3f} "
          f"{roc_auc_score(ly[m_v], s[m_v]):12.3f} {roc_auc_score(fy, fs[r]):8.3f}")

print("\nThreshold chosen at FPR<=5% on FakeAV half A -> (real flagged / fake flagged)")
for r in RULES:
    fpr, _, thr = roc_curve(fy[A], fs[r][A])
    t = thr[np.searchsorted(fpr, 0.05, side="right") - 1]
    out = lambda s, y: f"{np.mean(s[y==0]>=t):5.1%} / {np.mean(s[y==1]>=t):5.1%}"
    print(f"{r:6} thr {t:.3f} | FakeAV B: {out(fs[r][B], fy[B])} | LAV-DF test: {out(ls[r], ly)}")
