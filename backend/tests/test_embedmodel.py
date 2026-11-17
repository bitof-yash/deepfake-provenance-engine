"""Unit tests for the learned scoring path, using a fake classifier (no wav2vec2, no model file)."""
import numpy as np
import pytest

from app import config, embedmodel


class FakeClf:
    def __init__(self, p):
        self.p = np.asarray(p, dtype=float)

    def predict_proba(self, x):
        p = self.p[: len(x)]
        return np.column_stack([1 - p, p])


@pytest.fixture()
def patch(monkeypatch):
    def _patch(probs):
        monkeypatch.setattr(embedmodel, "enabled", lambda: True)
        monkeypatch.setattr(embedmodel, "_load", lambda: {"clf": FakeClf(probs)})
        monkeypatch.setattr(embedmodel, "window_embeddings", lambda wav: np.zeros((len(probs), 768)))
    return _patch


WAV = np.zeros(config.AUDIO_SR * 4, dtype=np.float32)


def test_clean_audio_scores_low_and_has_no_segments(patch):
    patch([0.05] * 16)
    r = embedmodel.detect(WAV)
    assert r["score"] < config.SUSPECT_THRESHOLD
    assert r["segments"] == []
    assert len(r["windows"]) == 4


def test_fake_segment_is_localised_and_scored_high(patch):
    patch([0.1] * 8 + [0.95] * 8)
    r = embedmodel.detect(WAV)
    assert r["score"] >= config.FAKE_THRESHOLD
    assert len(r["segments"]) == 1
    assert 1.5 <= r["segments"][0]["start"] <= 2.5
    assert r["windows"][3]["flagged"] and not r["windows"][0]["flagged"]


def test_too_short_audio_returns_none(patch):
    patch([0.9] * 4)
    assert embedmodel.detect(np.zeros(100, dtype=np.float32)) is None


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setenv("DFPE_EMBED", "0")
    assert embedmodel.detect(WAV) is None
