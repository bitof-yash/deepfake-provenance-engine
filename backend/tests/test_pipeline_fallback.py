"""Pipeline behaviour around the learned audio detector: fallback on error / None, and override on success."""
from app import pipeline


def test_falls_back_to_heuristics_when_detector_raises(manipulated_video, monkeypatch):
    def boom(_wav):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(pipeline.embedmodel, "detect", boom)
    res = pipeline.analyze_file(manipulated_video, "m.mp4")
    assert any("failed" in n for n in res["notes"])
    assert res["modalities"]["video"] is True
    assert res["disclaimer"] != pipeline.DISCLAIMER_EMBED
    assert res["segments"], "heuristic localisation should still work"


def test_falls_back_when_detector_returns_none(manipulated_video, monkeypatch):
    monkeypatch.setattr(pipeline.embedmodel, "detect", lambda _wav: None)
    res = pipeline.analyze_file(manipulated_video, "m.mp4")
    assert res["modalities"]["video"] is True
    assert res["disclaimer"] != pipeline.DISCLAIMER_EMBED


def test_learned_detector_overrides_score_and_modalities(manipulated_video, monkeypatch):
    win = {"start": 0.0, "end": 1.0, "video": None, "audio": 0.9, "lipsync": None,
           "fused": 0.9, "dominant": "audio", "flagged": True}
    fake = {"score": 0.9, "windows": [win], "segments": [], "analyzed": 1.0}
    monkeypatch.setattr(pipeline.embedmodel, "detect", lambda _wav: fake)
    res = pipeline.analyze_file(manipulated_video, "m.mp4")
    assert res["detection_score"] == 0.9
    assert res["modalities"] == {"video": False, "audio": True, "lipsync": False}
    assert res["disclaimer"] == pipeline.DISCLAIMER_EMBED
    assert res["windows"] == [win]
