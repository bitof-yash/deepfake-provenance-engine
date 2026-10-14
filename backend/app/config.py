"""Tunable constants for the Deepfake Forensics Provenance Engine (MVP)."""

MAX_UPLOAD_BYTES = 200 * 1024 * 1024      # reject uploads larger than this
MAX_ANALYZE_SECONDS = 60                  # only the first N seconds are analysed
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

FRAME_FPS = 10                            # analysis frame rate
FRAME_W, FRAME_H = 320, 240               # analysis frame size (grayscale)
FACE_DETECT_EVERY = 5                     # run face detector every N frames
AUDIO_SR = 16000                          # analysis sample rate
WINDOW_SECONDS = 1.0                      # localisation resolution
LIPSYNC_CONTEXT_SECONDS = 3.0             # correlation span around each window

# Within-clip outlier scoring: z at which a window is 50% suspicious.
Z_MIDPOINT = 3.0
Z_SLOPE = 1.5

# Fusion
MODALITY_PRIORS = {"video": 1.0, "audio": 1.0, "lipsync": 0.6}
AGREEMENT_BOOST = 0.10                    # added per extra agreeing modality
FLAG_THRESHOLD = 0.5                      # window flagged as manipulated

# Trust routing (on the global detection score, 0..1)
SUSPECT_THRESHOLD = 0.40
FAKE_THRESHOLD = 0.60
PROVENANCE_VALID_BONUS = 10               # trust-score points
PROVENANCE_INVALID_PENALTY = 25
