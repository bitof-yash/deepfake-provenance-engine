import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, media, pipeline
from .schemas import AnalysisResult

app = FastAPI(title="Deepfake Forensics Provenance Engine", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    import shutil
    return {
        "status": "ok",
        "c2patool": shutil.which("c2patool") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


@app.post("/analyze", response_model=AnalysisResult)
def analyze(file: UploadFile = File(...)):
    name = file.filename or "upload"
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in config.ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Allowed: {sorted(config.ALLOWED_SUFFIXES)}")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        size = 0
        with os.fdopen(fd, "wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File too large")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty file")
        try:
            return pipeline.analyze_file(tmp_path, os.path.basename(name))
        except media.MediaError as exc:
            raise HTTPException(422, str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
