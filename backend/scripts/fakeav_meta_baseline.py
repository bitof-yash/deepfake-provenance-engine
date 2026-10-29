"""Metadata-only baseline on FakeAVCeleb: how well can stream metadata alone detect fake audio?
Grouped CV by speaker identity so the same person is never in train and test."""
import json, os, pathlib, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path("~/fakeavceleb/FakeAVCeleb_v1.2").expanduser()
OUT = pathlib.Path("~/lavdf_transfer").expanduser()
MANIFEST, CACHE = OUT / "fakeav_manifest.json", OUT / "fakeav_meta.jsonl"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def probe(rec):
    p = ROOT / rec["file"]
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)],
                             capture_output=True, text=True, timeout=60).stdout
        d = json.loads(out or "{}")
        v = next((s for s in d.get("streams", []) if s["codec_type"] == "video"), {})
        a = next((s for s in d.get("streams", []) if s["codec_type"] == "audio"), {})
        dur = num(d.get("format", {}).get("duration"))
        return dict(key=rec["key"], file=rec["file"], type=rec["type"], method=rec["method"],
                    source=pathlib.Path(rec["file"]).parent.name,
                    fake_audio=int("FakeAudio" in rec["type"]),
                    a_sr=num(a.get("sample_rate")), a_bitrate=num(a.get("bit_rate")),
                    channels=num(a.get("channels")), a_dur=num(a.get("duration")),
                    v_dur=num(v.get("duration")), v_bitrate=num(v.get("bit_rate")),
                    dur_gap=num(a.get("duration")) - num(v.get("duration")),
                    size_per_s=os.path.getsize(p) / max(dur, 0.1) if dur == dur else float("nan"),
                    a_is_mp3=int(a.get("codec_name") == "mp3"),
                    v_is_h264=int(v.get("codec_name") == "h264"),
                    v_is_mpeg4=int(v.get("codec_name") == "mpeg4"))
    except Exception as e:
        return dict(key=rec["key"], error=f"{type(e).__name__}: {e}")


def load_or_probe(records):
    done = {}
    if CACHE.exists():
        for l in CACHE.open():
            if l.strip():
                r = json.loads(l)
                done[r["key"]] = r
    todo = [r for r in records if r["key"] not in done]
    print(f"{len(done)} cached, {len(todo)} to probe")
    if todo:
        with ThreadPoolExecutor(8) as ex, CACHE.open("a") as f:
            for i, r in enumerate(ex.map(probe, todo), 1):
                f.write(json.dumps(r) + "\n")
                done[r["key"]] = r
                if i % 1000 == 0:
                    f.flush()
                    print(f"  {i}/{len(todo)}")
    good = [r for r in done.values() if "error" not in r]
    print(f"{len(good)} usable, {len(done) - len(good)} probe errors")
    return good


def oof_auc(rows, feats):
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["fake_audio"] for r in rows])
    g = np.array([r["source"] for r in rows])
    s = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05).fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, s), int(y.sum()), len(y) - int(y.sum())


def main():
    rows = load_or_probe(json.load(open(MANIFEST)))
    all_feats = ["a_sr", "a_bitrate", "channels", "a_dur", "v_dur", "dur_gap", "v_bitrate",
                 "size_per_s", "a_is_mp3", "v_is_h264", "v_is_mpeg4"]
    audio_feats = ["a_sr", "a_bitrate", "channels", "a_is_mp3"]
    subsets = {
        "ALL clips": rows,
        "FakeVideo only (FVFA vs FVRA)": [r for r in rows if r["type"].startswith("FakeVideo")],
    }
    print("\nMetadata-only AUC for fake-audio vs real-audio (speaker-grouped 5-fold CV)")
    print(f"{'subset':<34}{'features':<18}{'AUC':>6}{'pos':>8}{'neg':>8}")
    for name, sub in subsets.items():
        for fname, feats in (("all metadata", all_feats), ("audio stream only", audio_feats)):
            a, p, n = oof_auc(sub, feats)
            print(f"{name:<34}{fname:<18}{a:>6.3f}{p:>8}{n:>8}")
    print("\nCompare with your model's 0.975 (ALL). If metadata gets close, the shortcut explains the result.")


if __name__ == "__main__":
    main()
