# Deepfake Forensics Provenance Engine (BT-225)

Our capstone project. You upload a video, and it tells you whether it looks fake, which seconds look suspicious, and whether the file carries valid C2PA provenance. Then it combines those into one trust decision. The backend is FastAPI and the UI is React + Vite.

## Running it

You need Python 3.11+, Node 20+ and ffmpeg/ffprobe on your PATH.

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The API is at http://localhost:8000 (docs at /docs) and the UI is at http://localhost:5173.

Tests: `cd backend && python -m pytest -q`. They make their own clips with ffmpeg, so you do not need any dataset.

`POST /analyze` takes a video in the `file` field and returns per-window scores, flagged segments, the provenance result and the routing decision. `GET /health` tells you whether ffmpeg and c2patool were found.

## How it works

Video frames and audio are scored in short windows, and the scores are fused into one score per window. Neighbouring suspicious windows get merged into segments. Separately, the file is checked for a C2PA manifest, and a routing policy combines that with the detection result.

The routing policy covers the awkward cases:
- valid provenance but the detector says fake: ESCALATE
- broken or tampered provenance: REJECT or REVIEW
- no provenance: use detection alone, capped at "Medium Trust"
- signed by an untrusted signer: handled as its own status, not treated as valid

Code layout (all under `backend/app/`):
- `video_analysis.py`, `audio_analysis.py`, `lipsync.py`: the original handcrafted detectors
- `clipmodel.py`: the learned per-window model
- `fusion.py`: fusing window scores and merging segments
- `provenance.py`: C2PA check via c2patool
- `routing.py`: the trust routing policy
- `config.py`: thresholds and limits

## Results so far (LAV-DF, 4000-clip subset)

The first version used handcrafted heuristics (spectral flatness, face sharpness and so on). They did not work. Clip-level AUC was 0.50, which is a coin flip, and localization IoU was 0.098.

So we switched to pretrained embeddings: wav2vec2 for audio and CLIP for the face crop, taken every 0.25 s, with a logistic regression on top (`scripts/extract_embeddings.py` and `scripts/train_windows.py`).

| model | clip AUC (95% CI) | localization mean IoU |
|---|---|---|
| audio only | 0.93 (0.91-0.94) | 0.56 |
| video only | 0.66 (0.63-0.68) | 0.23 |
| audio + video | 0.91 (0.90-0.93) | 0.54 |

These numbers use clip-normalization, which means subtracting each clip's mean embedding so the model cannot just recognise a whole clip's "style". The raw numbers are higher (up to 0.98), but we do not trust them, because of the next section.

For scale: LAV-DF fake segments are short (median 0.66 s), so even a perfect detector with 0.25 s windows only gets IoU of about 0.73-0.79. An IoU of 0.55 is therefore not as far from the ceiling as it sounds.

## Dataset shortcut problem (important)

We checked whether plain file metadata can separate real from fake in LAV-DF, without looking at any content. It can:
- audio/video duration gap: AUC 0.85
- audio bitrate: AUC 0.77
- clip duration: AUC 0.66

All three fake categories show it equally, which suggests the fakes were re-muxed differently from the reals. A model can learn that instead of learning anything about deepfakes. Clip-normalization and duration-controlled AUC (both in `train_windows.py`) reduce the problem but probably do not remove it.

We then re-encoded every clip with identical ffmpeg settings and cut audio and video to the same length (`scripts/normalize_clips.py`). The metadata shortcuts dropped to about chance (duration gap 0.85 to 0.51, audio bitrate 0.77 to 0.53) and we re-ran everything on the clean clips. The audio results did not change, but the raw video-only AUC fell from 0.84 to 0.79, so part of the earlier video score was encoding artifacts.

One red flag turned out to be informative: the audio-only model scores about 0.80 on fake_video clips, where the audio is supposed to be genuine. Position in the clip does not explain it (position-only within-clip AUC 0.56-0.59, and dropping the first and last second changes nothing). A boundary analysis does: on fake_video clips the audio score peaks in the 0.25 s window at the segment edge (0.64 inside, 0.35 just outside, about 0.11 far away) and is back near baseline within 0.5 s. So much of the audio model's performance is splice/edit-point detection. On fake_audio clips the score also stays high deeper into the segment (0.78-0.85). An interior-only test (`scripts/interior_check.py`) confirms real synthesis signal: using only windows at least 0.25 s inside a segment against windows from real clips, audio AUC is 0.97 on fake_audio (clip-normalised; 0.995 raw) and 0.66 on the fake_video control, and at a 0.5 s margin it is 0.99 on fake_audio against about 0.55 on fake_video (only 9 positive windows, so treat that as consistent with chance). A within-clip version of the test (`scripts/interior_within.py`) removes the speaker caveat: comparing windows at least 0.25 s inside a fake segment against windows at least 0.5 s outside it in the same clip (same speaker, microphone, room and encoding), audio AUC is 0.98 on fake_audio (156 test clips) and 0.98 on fake_av (196), against 0.65 on the fake_video control (only 24 clips, so indicative). Caveat: this is one dataset with one synthesis system, so it says nothing yet about other generators. Cross-category transfer (0.76-0.95 for audio) is probably inflated by the shared boundary signal. Treat the numbers above as upper bounds, not real-world accuracy. Cross-dataset results are in the next section.

## Cross-dataset results (LAV-DF to FakeAVCeleb)

We trained on LAV-DF only and tested on FakeAVCeleb (21,544 clips; Wav2Lip, FSGAN,
FaceSwap and RTVC fakes), with no training on FakeAVCeleb. Labels are fake-audio vs
real-audio.

| setting | audio only | audio + video |
|---|---|---|
| raw clips, all 21,544 | 0.975 | 0.878 |
| re-encoded identically (500/500/500/383 per type) | 0.960 | 0.854 |
| re-encoded, real video only (RVFA vs RVRA, 500 vs 500) | 0.963 | 0.862 |

The real-video comparison is the cleanest test, because the video is identical and only
the audio differs. Its 95% interval is roughly +/-0.03.

Findings:
- Audio appears to transfer, with a caveat: a residual bitrate/size shortcut (metadata
  AUC 0.686 after re-encoding) cannot be fully ruled out. Video does not help:
  audio+video is lower than audio alone.
- FakeAVCeleb has the same kind of metadata shortcut as LAV-DF, and a stronger one.
  Metadata alone gives AUC 0.993 for fake vs real audio (RTVC clips are mp3/stereo, real
  clips aac/mono). The raw 0.975 therefore cannot be read as evidence of transfer.
- After identical re-encoding, duration fields are clean (0.48-0.49). The remainder is
  bitrate and file size (single-feature AUC 0.57-0.62), most likely the encoder
  allocating bits by content. The classifier sees only wav2vec2 embeddings, not these
  fields.
- Clip-normalised models do not transfer (0.55-0.58). FakeAVCeleb fakes cover the whole
  clip, so subtracting the clip mean removes the signal.

Scope: one training dataset, one test dataset, Wav2Lip/RTVC-family fakes. No modern
TTS or diffusion-based fakes. Treat all numbers as upper bounds.

Scripts: `prepare_transfer.py`, `extract_fakeav_embeddings.py`, `evaluate_fakeav.py`,
`fakeav_meta_baseline.py`, `fakeav_matched_eval.py`, `fakeav_normalized.py`.

## Known limitations

- Audio does most of the work. The video-only model is weak once clip-level shortcuts are removed.
- The lip-sync score is weak. It uses a Haar face detector, which sometimes fires on things that are not faces.
- Fusion is still a simple confidence-weighted softmax, not a learned cross-attention model.
- The trust score is a heuristic (100 x (1 - detection), adjusted by provenance). It is not a calibrated probability.
- C2PA validation needs `c2patool` installed. Without it the engine can only see that C2PA markers exist, and reports `present_unverified`, which is never treated as valid.
- Only the first 60 s of a file is analysed, and uploads are capped at 200 MB.
- No auth or rate limiting, so keep it on localhost.

## What we want to do next

- Held-out speaker split of LAV-DF (FakeAVCeleb transfer is done, see above).
- Replace the Haar mouth tracking with landmarks for a real lip-sync score.
- Test the ESCALATE and invalid-signature paths on more real C2PA manifests.
