"""ffmpeg/ffprobe helpers: probing, frame extraction, audio extraction."""
import json
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile

from . import config


class MediaError(Exception):
    pass


def probe(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration:stream=codec_type",
        path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=True).stdout
        info = json.loads(out)
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise MediaError("Could not read media file (is it a valid video?)") from exc
    streams = {s.get("codec_type") for s in info.get("streams", [])}
    try:
        duration = float(info.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    return {"duration": duration, "has_video": "video" in streams, "has_audio": "audio" in streams}


def extract_frames(path: str) -> np.ndarray:
    """Return grayscale frames, shape (n, H, W), sampled at config.FRAME_FPS."""
    w, h = config.FRAME_W, config.FRAME_H
    cmd = [
        "ffmpeg", "-v", "error", "-i", path, "-an", "-t", str(config.MAX_ANALYZE_SECONDS),
        "-vf", f"fps={config.FRAME_FPS},scale={w}:{h}",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    if proc.returncode != 0:
        raise MediaError("Failed to decode video frames")
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (w * h)
    if n == 0:
        raise MediaError("No video frames could be decoded")
    return buf[: n * w * h].reshape(n, h, w)


def extract_audio(path: str) -> np.ndarray | None:
    """Return mono float32 audio at config.AUDIO_SR in [-1, 1], or None."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", path, "-vn", "-ac", "1",
            "-ar", str(config.AUDIO_SR), "-t", str(config.MAX_ANALYZE_SECONDS), wav_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
        if proc.returncode != 0:
            return None
        sr, data = wavfile.read(wav_path)
        if data.size == 0:
            return None
        data = data.astype(np.float32)
        if data.dtype.kind == "i" or np.abs(data).max() > 1.5:
            data /= 32768.0
        return data
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass
