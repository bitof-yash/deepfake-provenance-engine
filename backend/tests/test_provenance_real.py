"""Provenance tests built from REAL c2patool 0.26.1 output (captured while testing), plus live
tests against tiny signed/tampered fixtures that run only when c2patool is installed.

Fixtures were signed with the public C2PA *test* certificate, so their signer is (correctly)
reported as untrusted. tampered.mp4 is signed.mp4 with 32 bytes of media data flipped.
"""
import json
import pathlib
import shutil
import subprocess

import pytest

from app import provenance, routing

FIX = pathlib.Path(__file__).parent / "fixtures" / "c2pa"
UNTRUSTED = {"code": "signingCredential.untrusted", "explanation": "signing certificate untrusted"}
MISMATCH = {"code": "assertion.bmffHash.mismatch", "explanation": "asset hash error"}


def _manifest(failures, ai=False):
    actions = {"actions": [{"action": "c2pa.created", **(
        {"digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"}
        if ai else {})}]}
    d = {"active_manifest": "urn:c2pa:x",
         "manifests": {"urn:c2pa:x": {"assertions": [{"label": "c2pa.actions.v2", "data": actions}],
                                      "signature_info": {"alg": "Es256", "issuer": "Test Issuer"}}},
         "validation_state": "Invalid" if failures else "Valid"}
    if failures:
        d["validation_status"] = failures
    return json.dumps(d, indent=2)


def _patch(monkeypatch, stdout="", stderr="", code=0):
    monkeypatch.setattr(provenance.shutil, "which", lambda _: "/x/c2patool")
    monkeypatch.setattr(provenance.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, code, stdout, stderr))


def test_valid_and_trusted(monkeypatch):
    _patch(monkeypatch, _manifest([]))
    r = provenance.check("x.mp4")
    assert r["status"] == provenance.STATUS_VALID and r["issuer"] == "Test Issuer"


def test_untrusted_signer_only_is_not_invalid(monkeypatch):
    _patch(monkeypatch, _manifest([UNTRUSTED], ai=True))
    r = provenance.check("x.mp4")
    assert r["status"] == provenance.STATUS_UNTRUSTED
    assert r["declared_ai_generated"] is True


def test_hash_mismatch_is_invalid_even_with_untrusted_signer(monkeypatch):
    _patch(monkeypatch, _manifest([UNTRUSTED, MISMATCH]))
    r = provenance.check("x.mp4")
    assert r["status"] == provenance.STATUS_INVALID
    assert r["notes"] == ["assertion.bmffHash.mismatch"]


def test_no_manifest_is_absent(monkeypatch):
    _patch(monkeypatch, stderr="Error: No claim found", code=1)
    assert provenance.check("x.mp4")["status"] == provenance.STATUS_ABSENT


def test_routing_untrusted_gets_no_validity_bonus():
    prov = {"status": provenance.STATUS_UNTRUSTED, "declared_ai_generated": False}
    clean = routing.route(0.0, prov)
    assert clean["decision"] == "ACCEPT_UNVERIFIED" and clean["trust_score"] == 100.0
    fake = routing.route(1.0, prov)
    assert fake["decision"] == "REJECT"
    valid = routing.route(0.3, {"status": provenance.STATUS_VALID})
    untrusted = routing.route(0.3, prov)
    assert valid["trust_score"] > untrusted["trust_score"]


@pytest.mark.skipif(shutil.which("c2patool") is None, reason="c2patool not installed")
@pytest.mark.parametrize("name,expected", [
    ("clean.mp4", provenance.STATUS_ABSENT),
    ("signed.mp4", provenance.STATUS_UNTRUSTED),
    ("tampered.mp4", provenance.STATUS_INVALID),
])
def test_live_c2patool_on_fixtures(name, expected):
    assert provenance.check(str(FIX / name))["status"] == expected
