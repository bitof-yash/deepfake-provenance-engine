"""C2PA provenance verification.

If the `c2patool` CLI is installed it is used for real manifest validation.
Otherwise we can only detect whether C2PA/JUMBF markers are present, and we
say so explicitly: status becomes "present_unverified" (never "valid").
"""
import json
import shutil
import subprocess

STATUS_VALID = "valid"
STATUS_INVALID = "invalid"
STATUS_ABSENT = "absent"
STATUS_UNVERIFIED = "present_unverified"
STATUS_UNTRUSTED = "signed_untrusted"   # signature + hashes intact, but the signer is not on a trust list

# Failure codes that only mean "we do not trust the signer". Anything else (hash mismatch,
# bad signature, malformed manifest) means the content or manifest was altered or is broken.
_UNTRUSTED_ONLY = {"signingCredential.untrusted"}


def _scan_both(path: str) -> bool:
    """True only if both markers appear somewhere in the file."""
    found = {b"c2pa": False, b"jumb": False}
    tail = b""
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(4 * 1024 * 1024)
            if not chunk:
                break
            window = tail + chunk
            for k in found:
                if not found[k] and k in window:
                    found[k] = True
            if all(found.values()):
                return True
            tail = window[-3:]
    return all(found.values())


def check(path: str) -> dict:
    tool = shutil.which("c2patool")
    if tool:
        try:
            proc = subprocess.run([tool, path], capture_output=True, text=True, timeout=60)
        except subprocess.SubprocessError:
            return _result(STATUS_UNVERIFIED, "c2patool", ["c2patool failed to run"])
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            if "no claim" in text.lower() or "no manifest" in text.lower() or "jumbf" in text.lower():
                return _result(STATUS_ABSENT, "c2patool", ["No C2PA manifest found in this file"])
            return _result(STATUS_INVALID, "c2patool", [text.strip()[:300] or "Manifest could not be validated"])
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return _result(STATUS_UNVERIFIED, "c2patool", ["Could not parse c2patool output"])
        failures = _find_failures(data)
        integrity = [f for f in failures if f not in _UNTRUSTED_ONLY]
        ai = "trainedalgorithmicmedia" in proc.stdout.lower()
        issuer = _find_issuer(data)
        if integrity:
            return _result(STATUS_INVALID, "c2patool", integrity[:5], ai_generated=ai, issuer=issuer)
        if failures:
            return _result(STATUS_UNTRUSTED, "c2patool",
                           ["Signature and content hashes are intact, but the signer is not on a trust list"],
                           ai_generated=ai, issuer=issuer)
        return _result(STATUS_VALID, "c2patool", ["Manifest signature and hashes validated"],
                       ai_generated=ai, issuer=issuer)
    if _scan_both(path):
        return _result(
            STATUS_UNVERIFIED, "marker-scan",
            ["C2PA markers found, but signature NOT verified (install c2patool for real validation)"],
        )
    return _result(STATUS_ABSENT, "marker-scan", ["No C2PA manifest markers found in this file"])


def _find_failures(data) -> list:
    out = []

    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key == "validation_status" and isinstance(val, list):
                    out.extend(str(v.get("code", v)) if isinstance(v, dict) else str(v) for v in val)
                else:
                    walk(val)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return out


def _find_issuer(data):
    found = []

    def walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                if key in ("issuer", "claim_generator") and isinstance(val, str):
                    found.append(val)
                else:
                    walk(val)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return found[0] if found else None


def _result(status, verifier, notes, ai_generated=False, issuer=None) -> dict:
    return {"status": status, "verifier": verifier, "notes": notes,
            "declared_ai_generated": ai_generated, "issuer": issuer}
