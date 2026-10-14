import os
import subprocess

import numpy as np
import pytest
from scipy.io import wavfile

SR = 16000
DUR = 10


def _audio(path, manipulated):
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(0)
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    x += 0.01 * rng.standard_normal(len(t))
    if manipulated:
        a, b = 4 * SR, 6 * SR
        x[a:b] = 0.25 * rng.standard_normal(b - a)  # broadband noise burst
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def _make(tmp_path, name, manipulated):
    wav = str(tmp_path / f"{name}.wav")
    out = str(tmp_path / f"{name}.mp4")
    _audio(wav, manipulated)
    vf = "scale=320:240"
    if manipulated:
        vf += ",noise=alls=80:allf=t:enable='between(t,4,6)'"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=25:duration={DUR}",
         "-i", wav, "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out],
        check=True,
    )
    return out


@pytest.fixture(scope="session")
def clean_video(tmp_path_factory):
    return _make(tmp_path_factory.mktemp("clean"), "clean", False)


@pytest.fixture(scope="session")
def manipulated_video(tmp_path_factory):
    return _make(tmp_path_factory.mktemp("manip"), "manip", True)
