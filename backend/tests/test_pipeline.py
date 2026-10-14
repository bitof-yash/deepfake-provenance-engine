from fastapi.testclient import TestClient

from app import config, fusion, pipeline, routing
from app.main import app
from app.provenance import STATUS_ABSENT, STATUS_INVALID, STATUS_UNVERIFIED, STATUS_VALID

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_pipeline_localises_manipulated_segment(manipulated_video):
    res = pipeline.analyze_file(manipulated_video, "manip.mp4")
    assert res["modalities"]["video"] and res["modalities"]["audio"]
    assert res["segments"], "expected at least one flagged segment"
    # at least one flagged segment overlaps the injected 4-6 s corruption
    assert any(s["start"] < 6 and s["end"] > 4 for s in res["segments"])
    # nothing flagged far away from it (allow 1 s of boundary spill)
    assert all(s["end"] > 3 and s["start"] < 7 for s in res["segments"])


def test_manipulated_scores_higher_than_clean(clean_video, manipulated_video):
    clean = pipeline.analyze_file(clean_video, "clean.mp4")
    manip = pipeline.analyze_file(manipulated_video, "manip.mp4")
    assert manip["detection_score"] > clean["detection_score"] + 0.2
    assert not clean["segments"]


def test_api_analyze_roundtrip(manipulated_video):
    with open(manipulated_video, "rb") as fh:
        r = client.post("/analyze", files={"file": ("manip.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provenance"]["status"] in (STATUS_ABSENT, STATUS_UNVERIFIED)
    assert 0 <= body["routing"]["trust_score"] <= 100
    assert body["disclaimer"]


def test_api_rejects_bad_type_and_garbage():
    r = client.post("/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    r = client.post("/analyze", files={"file": ("x.mp4", b"not a video at all", "video/mp4")})
    assert r.status_code == 422


def test_fusion_agreement_and_none():
    assert fusion.fuse_window({"video": None, "audio": None, "lipsync": None}) == (None, "none")
    single, _ = fusion.fuse_window({"video": 0.9, "audio": 0.1, "lipsync": None})
    both, dom = fusion.fuse_window({"video": 0.9, "audio": 0.9, "lipsync": None})
    assert both > single and dom == "audio+video"


def _prov(status, ai=False):
    return {"status": status, "verifier": "test", "notes": [], "declared_ai_generated": ai, "issuer": None}


def test_routing_matrix():
    hi, mid, lo = 0.9, 0.5, 0.1
    assert routing.route(lo, _prov(STATUS_VALID))["decision"] == "ACCEPT"
    assert routing.route(mid, _prov(STATUS_VALID))["decision"] == "REVIEW"
    conflict = routing.route(hi, _prov(STATUS_VALID))
    assert conflict["decision"] == "ESCALATE" and conflict["conflict"] and conflict["trust_score"] <= 50
    assert routing.route(hi, _prov(STATUS_INVALID))["decision"] == "REJECT"
    assert routing.route(lo, _prov(STATUS_INVALID))["decision"] == "REVIEW"
    assert routing.route(lo, _prov(STATUS_ABSENT))["decision"] == "ACCEPT_UNVERIFIED"
    assert routing.route(hi, _prov(STATUS_ABSENT))["decision"] == "REJECT"
    # an unverified marker must never be treated as valid provenance
    assert routing.route(hi, _prov(STATUS_UNVERIFIED))["decision"] == "REJECT"
    assert config.SUSPECT_THRESHOLD < config.FAKE_THRESHOLD
