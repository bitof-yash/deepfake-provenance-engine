# Deepfake Forensics Provenance Engine (BT-225)

Multi-modal deepfake forensics MVP: audio-visual detection, segment-level localization and C2PA provenance
fused into one trust decision. API-first (FastAPI) with a React/Vite web UI.

## Run it

Needs Python 3.11+, Node 20+, and `ffmpeg`/`ffprobe` on PATH.

```bash
# 1. Backend  (http://localhost:8000, docs at /docs)
cd backend
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# 2. Frontend (http://localhost:5173)  -- in a second terminal
cd frontend
npm install
npm run dev
```

Tests: `cd backend && python -m pytest -q` (generates synthetic clips with ffmpeg, no datasets needed).

API: `POST /analyze` (multipart field `file`) returns windows, flagged segments, provenance and the routing decision.
`GET /health` reports whether `ffmpeg` and `c2patool` were found.

## Pipeline

```
video ──► frames ──► visual artifact score (per 1 s window) ──┐
      └─► audio  ──► spectral score (per 1 s window) ──────────┼─► attention-weighted fusion ─► segments
          mouth motion × audio energy ► lip-sync score ────────┘                │
file bytes ──► C2PA check (c2patool) ─────────────────────────────────► trust routing ─► Trust Report
```

| File | Role |
|---|---|
| `app/video_analysis.py` | Face-region high-frequency energy, sharpness, temporal jitter (within-clip outliers) |
| `app/audio_analysis.py` | Spectral flatness, high-band ratio, spectral flux (within-clip outliers) |
| `app/lipsync.py` | Mouth-motion vs audio-energy correlation |
| `app/fusion.py` | Per-window fusion, cross-modal agreement boost, segment merging |
| `app/provenance.py` | C2PA verification via `c2patool` |
| `app/routing.py` | Trust-routing policy (ACCEPT / ACCEPT_UNVERIFIED / REVIEW / ESCALATE / REJECT) |
| `app/config.py` | All thresholds and limits |

Trust routing handles the conflict cases the project is about: valid provenance + detector says fake -> `ESCALATE`;
invalid (broken) provenance -> `REJECT`/`REVIEW`; no provenance -> detection alone, capped at "Medium Trust".

## What is real and what is a placeholder (read before presenting)

- **Detectors are heuristics, not trained models.** They flag windows that differ strongly from the rest of the
  same clip. The tests show they catch crude corruption (noise bursts) and localize it; they will **not** reliably
  catch modern face-swap or voice-clone deepfakes. No accuracy figure exists yet, so do not claim one.
- **"Cross-attention fusion" is a stand-in.** `fusion.fuse_window` uses confidence-driven softmax weights. The
  learned cross-attention model is future work (see below); the pipeline is built so it can replace that one function.
- **Lip-sync is a weak signal** and is down-weighted. A Haar face detector can also fire on non-face content.
- **C2PA needs `c2patool`.** Without it the engine only detects C2PA markers and reports `present_unverified`, which
  is never treated as valid. Install `c2patool` (Content Authenticity Initiative) and put it on PATH for real validation.
- The trust score is an uncalibrated heuristic (`100 x (1 - detection)` adjusted by provenance), not a probability.
- Only the first 60 s of a file are analysed; uploads are capped at 200 MB.

## Next steps toward the approved abstract

1. Extract features / train a fusion model on FakeAVCeleb and LAV-DF (Colab/Kaggle GPU), replacing `fuse_window`
   and the heuristic scorers; report real metrics (AUC, localization IoU) for the paper.
2. Replace the Haar face detector with a landmark-based mouth tracker for a proper lip-sync (SyncNet-style) score.
3. Install `c2patool`, add signed/tampered C2PA test files, and test the ESCALATE and invalid-signature paths on real manifests.
4. Add an API key / rate limit before exposing it beyond localhost.
