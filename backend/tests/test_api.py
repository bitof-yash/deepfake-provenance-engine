"""HTTP-layer checks for /health and the /analyze error paths (no dataset or model needed)."""
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("DFPE_EMBED", "0")      # never load wav2vec2 in tests
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body) >= {"c2patool", "ffmpeg"}


def test_rejects_unsupported_type(client):
    assert client.post("/analyze", files={"file": ("x.txt", b"hi")}).status_code == 415


def test_rejects_empty_file(client):
    assert client.post("/analyze", files={"file": ("e.mp4", b"")}).status_code == 400


def test_junk_video_is_422_not_500(client):
    r = client.post("/analyze", files={"file": ("f.mp4", b"not a video")})
    assert r.status_code == 422
