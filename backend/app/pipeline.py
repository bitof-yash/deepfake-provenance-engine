"""End-to-end analysis pipeline: media -> modality scores -> fusion -> provenance -> routing."""
from . import (audio_analysis, clipmodel, config, embedmodel, fusion, lipsync, media, provenance, routing,
               video_analysis)

DISCLAIMER_HEURISTIC = (
    "MVP prototype: detectors are uncalibrated heuristics (within-clip outlier analysis), not trained "
    "models. Scores are indicative only and must not be used for real forensic or legal decisions."
)
DISCLAIMER_MODEL = (
    "Research prototype: the detection score comes from a baseline classifier trained on handcrafted "
    "audio-visual features, and localization is heuristic. Results depend on the training data and must "
    "not be used for real forensic or legal decisions."
)


DISCLAIMER_EMBED = (
    "Research prototype: the detection score comes from a wav2vec2-embedding classifier trained on "
    "LAV-DF (one dataset, one synthesis system). It uses audio only; video and lip-sync are not part "
    "of the score. Results must not be used for real forensic or legal decisions."
)


def compute(path: str) -> dict:
    """Run every analysis stage and return the intermediate results (also used for training)."""
    info = media.probe(path)
    if not info["has_video"]:
        raise media.MediaError("File has no video stream")

    notes = []
    frames = media.extract_frames(path)
    analyzed = len(frames) / config.FRAME_FPS
    if info["duration"] > config.MAX_ANALYZE_SECONDS:
        notes.append(f"Only the first {config.MAX_ANALYZE_SECONDS}s were analysed")
    if analyzed < 4:
        notes.append("Clip shorter than 4s: within-clip localisation is unreliable")

    vres = video_analysis.analyze(frames)
    n_windows = len(vres.window_scores)
    if not vres.face_found:
        notes.append("No face detected: analysed the central frame region; lip-sync unavailable")

    ares, audio_scores, lip_scores = None, None, None
    audio = media.extract_audio(path) if info["has_audio"] else None
    if audio is None:
        notes.append("No usable audio track: audio and lip-sync analysis skipped")
    else:
        ares = audio_analysis.analyze(audio, n_windows)
        audio_scores = ares.window_scores
        lip_scores = lipsync.analyze(vres.mouth_motion, ares.envelope, n_windows)

    windows = fusion.build_windows(vres.window_scores, audio_scores, lip_scores, analyzed)
    return {
        "info": info, "analyzed": analyzed, "notes": notes, "vres": vres, "ares": ares,
        "audio_scores": audio_scores, "lip_scores": lip_scores, "audio": audio, "windows": windows,
        "heuristic_score": fusion.global_score(windows),
        "clip_vector": clipmodel.clip_vector(vres, ares, lip_scores),
    }


def analyze_file(path: str, filename: str) -> dict:
    c = compute(path)
    notes = list(c["notes"])
    windows = c["windows"]
    segments = fusion.segments_from_windows(windows)

    score, disclaimer = c["heuristic_score"], DISCLAIMER_HEURISTIC
    bundle = clipmodel.load_model()
    if bundle is not None:
        score = clipmodel.predict(bundle, c["clip_vector"])
        disclaimer = DISCLAIMER_MODEL
        notes.append("Detection score from the trained baseline model; localization is heuristic")

    modalities = None
    if c["audio"] is not None:
        try:
            det = embedmodel.detect(c["audio"])
        except Exception as exc:  # keep the API up; fall back to the heuristics
            det = None
            notes.append(f"Learned audio detector failed ({type(exc).__name__}); using fallback")
        if det is not None:
            score, disclaimer = det["score"], DISCLAIMER_EMBED
            windows, segments = det["windows"], det["segments"]
            modalities = {"video": False, "audio": True, "lipsync": False}
            notes.append("Detection from the learned wav2vec2 audio model (LAV-DF-trained, 0.25 s windows); "
                         "video and lip-sync are not used in the score")

    prov = provenance.check(path)
    route = routing.route(score, prov)

    audio_scores, lip_scores = c["audio_scores"], c["lip_scores"]
    return {
        "filename": filename,
        "duration": round(c["info"]["duration"], 2),
        "analyzed_seconds": round(c["analyzed"], 2),
        "detection_score": round(score, 3),
        "modalities": modalities or {
            "video": True,
            "audio": audio_scores is not None and any(s is not None for s in audio_scores),
            "lipsync": lip_scores is not None and any(s is not None for s in lip_scores),
        },
        "windows": windows,
        "segments": segments,
        "provenance": prov,
        "routing": route,
        "notes": notes,
        "disclaimer": disclaimer,
    }
