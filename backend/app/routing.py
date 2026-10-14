"""Trust-routing policy: combines the detection score with the C2PA result.

The interesting cases are the *conflicts* between the two signals:
  * valid provenance + detector says fake  -> ESCALATE (signed-but-synthetic, or a false positive)
  * invalid provenance (broken signature)   -> content was altered after signing
  * absent provenance                       -> fall back to detection alone, with lower confidence
"""
from . import config
from .provenance import (STATUS_ABSENT, STATUS_INVALID, STATUS_UNTRUSTED, STATUS_UNVERIFIED,
                         STATUS_VALID)


def route(detection: float, prov: dict) -> dict:
    status = prov["status"]
    if detection >= config.FAKE_THRESHOLD:
        level = "fake"
    elif detection >= config.SUSPECT_THRESHOLD:
        level = "suspect"
    else:
        level = "clean"

    reasons = []
    conflict = False

    if status == STATUS_VALID:
        reasons.append("C2PA manifest present and validated")
        if level == "clean":
            decision, label = "ACCEPT", "High Trust"
        elif level == "suspect":
            decision, label = "REVIEW", "Review"
            reasons.append("Minor anomalies detected despite valid provenance")
        else:
            decision, label, conflict = "ESCALATE", "Conflict", True
            reasons.append("Detector flags manipulation although provenance is valid "
                           "(possible signed synthetic media, or a detector false positive)")
    elif status == STATUS_INVALID:
        reasons.append("C2PA manifest present but failed validation (content altered after signing?)")
        if level == "clean":
            decision, label, conflict = "REVIEW", "Review", True
        else:
            decision, label = "REJECT", "Low Trust"
    elif status == STATUS_UNTRUSTED:
        reasons.append("C2PA signature and content hashes are intact, but the issuer is not on a trust list; "
                       "treated as no trusted provenance")
        decision, label = _detection_only(level, reasons)
    elif status == STATUS_UNVERIFIED:
        reasons.append("C2PA markers present but signature not verified; treated as no provenance")
        decision, label = _detection_only(level, reasons)
    else:
        reasons.append("No C2PA provenance; decision based on detection alone")
        decision, label = _detection_only(level, reasons)

    if prov.get("declared_ai_generated"):
        reasons.append("Manifest declares the content as AI-generated")

    trust = 100.0 * (1.0 - detection)
    if status == STATUS_VALID:
        trust += config.PROVENANCE_VALID_BONUS
    elif status == STATUS_INVALID:
        trust -= config.PROVENANCE_INVALID_PENALTY
    if conflict and decision == "ESCALATE":
        trust = min(trust, 50.0)
    trust = max(0.0, min(100.0, trust))

    return {
        "decision": decision,
        "label": label,
        "trust_score": round(trust, 1),
        "detection_level": level,
        "conflict": conflict,
        "reasons": reasons,
    }


def _detection_only(level: str, reasons: list):
    if level == "clean":
        reasons.append("No manipulation detected, but origin cannot be verified")
        return "ACCEPT_UNVERIFIED", "Medium Trust"
    if level == "suspect":
        return "REVIEW", "Review"
    return "REJECT", "Low Trust"
