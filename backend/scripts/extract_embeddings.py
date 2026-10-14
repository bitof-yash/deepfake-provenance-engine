"""Per-window pretrained embeddings for LAV-DF clips (0.25 s windows by default).

  audio : wav2vec2-base  (768-d, mean-pooled over each window's ~20 ms frames)
  video : CLIP ViT-B/32 image encoder on the face crop at each window centre (512-d)

Run from the backend folder with the venv active:
  pip install torch transformers
  python -m scripts.extract_embeddings --subset ~/lavdf_big/subset.json --root ~/lavdf_big/LAV-DF \
         --out ~/lavdf_big/emb --limit 50          # pilot: time it first!
Resumable (skips clips already saved). Auto-uses CUDA, then Apple MPS, then CPU.
--dry-run swaps the two encoders for random vectors to test the plumbing without any model download.
"""
import argparse
import json
import pathlib
import time

import cv2
import numpy as np

from app import config, media

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class Encoders:
    def __init__(self, dry_run: bool):
        self.dry = dry_run
        if dry_run:
            self.rng = np.random.default_rng(0)
            self.device = "dry-run"
            return
        import torch
        from transformers import CLIPVisionModelWithProjection, Wav2Vec2Model
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device).eval()

    def video(self, crops: np.ndarray) -> np.ndarray:           # (n,224,224,3) float32 normalised
        if self.dry:
            return self.rng.standard_normal((len(crops), 512)).astype(np.float32)
        t = self.torch.from_numpy(crops).permute(0, 3, 1, 2).to(self.device)
        with self.torch.no_grad():
            return self.clip(pixel_values=t).image_embeds.float().cpu().numpy()

    def audio(self, wav: np.ndarray) -> np.ndarray:             # (T,768) frames at 50 Hz
        if self.dry:
            return self.rng.standard_normal((max(1, len(wav) // 320), 768)).astype(np.float32)
        x = (wav - wav.mean()) / (wav.std() + 1e-7)
        t = self.torch.from_numpy(x.astype(np.float32))[None].to(self.device)
        with self.torch.no_grad():
            return self.w2v(t).last_hidden_state[0].float().cpu().numpy()


def face_crops(path: str, centres: np.ndarray):
    """Face crop (224x224, CLIP-normalised) at each time in `centres`; returns (crops, face_fraction)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    want = {}
    for i, t in enumerate(centres):
        want.setdefault(min(int(t * fps), n_frames - 1), []).append(i)
    crops = np.zeros((len(centres), 224, 224, 3), dtype=np.float32)
    found, last_bbox, idx, got = 0, None, 0, set()
    while len(got) < len(want):
        ok = cap.grab()
        if not ok:
            break
        if idx in want:
            ok, frame = cap.retrieve()
            if ok:
                H, W = frame.shape[:2]
                scale = 320.0 / W
                small = cv2.cvtColor(cv2.resize(frame, (320, int(H * scale))), cv2.COLOR_BGR2GRAY)
                faces = CASCADE.detectMultiScale(small, 1.1, 5, minSize=(30, 30))
                if len(faces):
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    last_bbox = (x / scale, y / scale, w / scale, h / scale)
                    found += 1
                if last_bbox is None:
                    x0, y0, x1, y1 = W // 4, H // 6, 3 * W // 4, 5 * H // 6
                else:
                    x, y, w, h = last_bbox
                    m = 0.2
                    x0, y0 = int(max(0, x - m * w)), int(max(0, y - m * h))
                    x1, y1 = int(min(W, x + (1 + m) * w)), int(min(H, y + (1 + m) * h))
                roi = cv2.cvtColor(cv2.resize(frame[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA),
                                   cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                roi = (roi - CLIP_MEAN) / CLIP_STD
                for i in want[idx]:
                    crops[i] = roi
                got.add(idx)
        idx += 1
    cap.release()
    return crops, found / max(1, len(want))


def process(entry, root, enc, w):
    path = str(pathlib.Path(root) / entry["file"])
    n = max(1, int(entry["duration"] // w))
    centres = (np.arange(n) + 0.5) * w
    crops, face_frac = face_crops(path, centres)
    v = enc.video(crops)
    wav = media.extract_audio(path)
    a = np.zeros((n, 768), dtype=np.float32)
    has_audio = wav is not None and len(wav) > 0
    if has_audio:
        frames = enc.audio(wav)
        for i in range(n):
            lo, hi = int(i * w / 0.02), max(int(i * w / 0.02) + 1, int((i + 1) * w / 0.02))
            seg = frames[min(lo, len(frames) - 1): min(hi, len(frames))]
            a[i] = (seg if len(seg) else frames[-1:]).mean(axis=0)
    return a.astype(np.float16), v.astype(np.float16), has_audio, face_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (pilot run)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    entries = json.load(open(pathlib.Path(a.subset).expanduser()))
    if a.limit:
        entries = entries[: a.limit]
    root, out = str(pathlib.Path(a.root).expanduser()), pathlib.Path(a.out).expanduser()
    enc = Encoders(a.dry_run)
    print("device:", enc.device)
    todo = []
    for e in entries:
        p = out / e["split"] / (pathlib.Path(e["file"]).stem + ".npz")
        if not p.exists():
            todo.append((e, p))
    print(f"{len(entries) - len(todo)} already done, {len(todo)} to go")
    t0, errs = time.time(), 0
    for k, (e, p) in enumerate(todo, 1):
        try:
            av, vv, has_a, ff = process(e, root, enc, a.window)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, a=av, v=vv, has_audio=has_a, face_frac=ff, window=a.window)
        except Exception as ex:
            errs += 1
            print("ERROR", e["file"], f"{type(ex).__name__}: {ex}")
        if k % 10 == 0 or k == len(todo):
            rate = k / (time.time() - t0)
            print(f"{k}/{len(todo)} errors={errs} {rate:.2f} clips/s ETA {(len(todo) - k) / rate / 60:.0f} min")


if __name__ == "__main__":
    main()
