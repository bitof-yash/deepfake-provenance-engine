from typing import Optional

from pydantic import BaseModel


class Window(BaseModel):
    start: float
    end: float
    video: Optional[float]
    audio: Optional[float]
    lipsync: Optional[float]
    fused: Optional[float]
    dominant: str
    flagged: bool


class Segment(BaseModel):
    start: float
    end: float
    modality: str
    peak_score: float


class Provenance(BaseModel):
    status: str
    verifier: str
    notes: list[str]
    declared_ai_generated: bool
    issuer: Optional[str] = None


class Routing(BaseModel):
    decision: str
    label: str
    trust_score: float
    detection_level: str
    conflict: bool
    reasons: list[str]


class AnalysisResult(BaseModel):
    filename: str
    duration: float
    analyzed_seconds: float
    detection_score: float
    modalities: dict[str, bool]
    windows: list[Window]
    segments: list[Segment]
    provenance: Provenance
    routing: Routing
    notes: list[str]
    disclaimer: str
