#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

ROOT = Path.home() / "lavdf_transfer"
MANIFEST = ROOT / "fakeav_manifest.json"
EMBED_DIR = ROOT / "fakeav_embeddings"
MODEL_DIR = ROOT / "models"


def rows_from_manifest(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for field in ("videos", "items", "records", "data", "manifest"):
            if isinstance(obj.get(field), list):
                return obj[field]
        if obj and all(isinstance(v, dict) for v in obj.values()):
            result = []
            for key, value in obj.items():
                item = dict(value)
                item.setdefault("key", key)
                result.append(item)
            return result
    raise ValueError("Unknown fakeav_manifest.json structure")


def first(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return None


def auc(y, scores):
    return float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None


def evaluate(name, model, rows):
    ys, scores = defaultdict(list), defaultdict(list)
    skipped = no_audio = 0

    for row in rows:
        key = first(row, ("key", "sha1", "id", "embedding_key", "feature_key"))
        category = first(row, ("type", "category", "label", "class"))
        if key is None or category is None:
            skipped += 1
            continue

        category = str(category).lower().replace("_", "-").replace(" ", "")
        fake_video = int("fakevideo" in category)
        fake_audio = int("fakeaudio" in category)
        fake_any = int(fake_video or fake_audio)

        file = EMBED_DIR / f"{key}.npz"
        if not file.exists():
            skipped += 1
            continue
        try:
            with np.load(file) as data:
                a = np.asarray(data["a"], dtype=np.float32)
                v = np.asarray(data["v"], dtype=np.float32)
                has_audio = bool(np.asarray(data["has_audio"]).item())
        except Exception:
            skipped += 1
            continue

        if name.startswith(("audio", "both")) and not has_audio:
            no_audio += 1
            continue

        if name.startswith("audio"):
            x, targets = a, {"fake_audio": fake_audio}
        elif name.startswith("video"):
            x, targets = v, {"fake_video": fake_video}
        else:
            if len(a) != len(v):
                skipped += 1
                continue
            x = np.concatenate((a, v), axis=1)
            targets = {
                "any_manipulation": fake_any,
                "fake_video": fake_video,
                "fake_audio": fake_audio,
            }

        if x.ndim != 2 or len(x) == 0:
            skipped += 1
            continue

        probs = model.predict_proba(x)
        classes = list(model.classes_)
        positive_index = classes.index(1) if 1 in classes else classes.index(True)
        clip_score = float(np.mean(probs[:, positive_index]))

        for target_name, target in targets.items():
            ys[target_name].append(target)
            scores[target_name].append(clip_score)

    report = {
        "clips_evaluated": len(next(iter(ys.values()))) if ys else 0,
        "skipped": skipped,
        "skipped_no_audio": no_audio,
        "metrics": {},
    }
    for target_name, y in ys.items():
        p = scores[target_name]
        pred = [int(s >= 0.5) for s in p]
        report["metrics"][target_name] = {
            "auc": auc(y, p),
            "accuracy_at_0.5": float(accuracy_score(y, pred)),
            "precision_at_0.5": float(precision_score(y, pred, zero_division=0)),
            "recall_at_0.5": float(recall_score(y, pred, zero_division=0)),
            "f1_at_0.5": float(f1_score(y, pred, zero_division=0)),
            "positive_clips": int(sum(y)),
            "negative_clips": int(len(y) - sum(y)),
        }
    return report


def main():
    for required in (MANIFEST, EMBED_DIR, MODEL_DIR):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    with MANIFEST.open(encoding="utf-8") as f:
        rows = rows_from_manifest(json.load(f))

    results = {
        "dataset": "FakeAVCeleb",
        "training_dataset": "LAV-DF",
        "evaluation_level": "clip-level; mean window probability",
        "num_manifest_rows": len(rows),
        "models": {},
    }

    names = ("audio", "audio_clipnorm", "video", "video_clipnorm",
             "both", "both_clipnorm")
    for name in names:
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"Skipping missing model: {path}")
            continue
        print(f"Evaluating {name}...")
        results["models"][name] = evaluate(name, joblib.load(path), rows)
        print(json.dumps(results["models"][name], indent=2))

    output = Path.home() / "lavdf_transfer" / "fakeav_evaluation_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFinished. Results saved to: {output}")


if __name__ == "__main__":
    main()
