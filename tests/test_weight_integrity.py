"""Weight-pin integrity: memoization must never become a blind spot.

The cache exists because hashing 2.4 GB costs ~18 s per run. These tests pin
down the two properties that matter: a cache hit is only allowed when the
file is byte-identical in size and mtime, and any real change is caught on the
next pass even with the cache enabled.
"""

from __future__ import annotations

import json
from pathlib import Path

PIN_A = "a" * 64
PIN_B = "b" * 64


def _registry(tmp: Path, model: Path, pin: str, *, external: Path | None = None) -> Path:
    doc = {
        "models": [
            {
                "name": "unit",
                "path": str(model),
                "sha256": pin,
                **(
                    {"external_data": {"path": str(external), "sha256": pin}}
                    if external is not None
                    else {}
                ),
            }
        ],
        "whisper": {},
    }
    reg = tmp / "registry.json"
    reg.write_text(json.dumps(doc), encoding="utf-8")
    return reg


def _digest(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def test_passing_pin_verifies_and_populates_the_cache(tmp_path):
    from src.weight_integrity import verify_registry

    model = tmp_path / "m.onnx"
    model.write_bytes(b"weights")
    reg = _registry(tmp_path, model, _digest(b"weights"))
    cache = tmp_path / "cache.json"

    first = verify_registry(reg, cache_path=cache)
    assert first["ok"] is True
    assert first["hashed"] == 1 and first["cached"] == 0
    assert cache.is_file()

    second = verify_registry(reg, cache_path=cache)
    assert second["ok"] is True
    assert second["hashed"] == 0 and second["cached"] == 1


def test_tampered_weights_fail_even_with_a_warm_cache(tmp_path):
    """The whole point: a cache hit must not survive an actual edit."""
    from src.weight_integrity import verify_registry

    model = tmp_path / "m.onnx"
    model.write_bytes(b"weights")
    reg = _registry(tmp_path, model, _digest(b"weights"))
    cache = tmp_path / "cache.json"
    assert verify_registry(reg, cache_path=cache)["ok"] is True

    model.write_bytes(b"tampered")
    out = verify_registry(reg, cache_path=cache)
    assert out["ok"] is False
    assert out["mismatches"][0]["name"] == "unit"


def test_external_weight_blob_is_pinned_separately(tmp_path):
    """The graph alone is inert; the 2.4 GB blob must be covered too."""
    from src.weight_integrity import verify_registry

    model = tmp_path / "m.onnx"
    blob = tmp_path / "m.onnx.data"
    model.write_bytes(b"graph")
    blob.write_bytes(b"tensors")
    reg = _registry(tmp_path, model, _digest(b"graph"), external=blob)
    reg.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "name": "unit",
                        "path": str(model),
                        "sha256": _digest(b"graph"),
                        "external_data": {"path": str(blob), "sha256": PIN_B},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = verify_registry(reg, cache_path=None, use_cache=False)
    assert out["ok"] is False
    assert any("weights" in m["name"] for m in out["mismatches"])


def test_unreadable_registry_is_not_ok(tmp_path):
    from src.weight_integrity import verify_registry

    bad = tmp_path / "registry.json"
    bad.write_text("{not json", encoding="utf-8")
    out = verify_registry(bad, cache_path=None, use_cache=False)
    assert out["ok"] is False
    assert "unreadable registry" in out["error"]


def test_absent_and_unpinned_slots_are_skipped_not_errors(tmp_path):
    from src.weight_integrity import verify_registry

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    reg = tmp_path / "registry.json"
    reg.write_text(
        json.dumps(
            {
                "models": [
                    {"name": "byo", "path": str(tmp_path / "absent.onnx"), "sha256": PIN_A},
                    {"name": "unpinned", "path": str(model), "sha256": ""},
                ]
            }
        ),
        encoding="utf-8",
    )
    out = verify_registry(reg, cache_path=None, use_cache=False)
    assert out["ok"] is True
    assert out["checked"] == 0
    assert any("absent" in s for s in out["skipped"])
    assert any("unpinned" in s for s in out["skipped"])


def test_corrupt_cache_is_treated_as_cold_not_as_a_pass(tmp_path):
    from src.weight_integrity import verify_registry

    model = tmp_path / "m.onnx"
    model.write_bytes(b"weights")
    reg = _registry(tmp_path, model, _digest(b"weights"))
    cache = tmp_path / "cache.json"
    cache.write_text("{{{ not json", encoding="utf-8")
    out = verify_registry(reg, cache_path=cache)
    assert out["ok"] is True
    assert out["hashed"] == 1


def test_mismatched_pin_is_reported_with_both_digests(tmp_path):
    from src.weight_integrity import verify_registry

    model = tmp_path / "m.onnx"
    model.write_bytes(b"weights")
    reg = _registry(tmp_path, model, PIN_B)
    out = verify_registry(reg, cache_path=None, use_cache=False)
    assert out["ok"] is False
    assert out["mismatches"][0]["expected"] == PIN_B
    assert out["mismatches"][0]["actual"] == _digest(b"weights")


def test_scheduled_task_audits_a_mismatch(tmp_path):
    """A detected tamper must leave a custody record, not just a log line."""
    import src.weight_integrity as wi
    import worker
    from src.database import StdlibSqliteRepo

    repo = StdlibSqliteRepo(tmp_path / "integrity.db")
    model = tmp_path / "m.onnx"
    model.write_bytes(b"weights")
    _registry(tmp_path, model, PIN_B)  # wrong pin on purpose

    original = wi.verify_registry
    wi.verify_registry = lambda *a, **k: {  # type: ignore[assignment]
        "ok": False,
        "checked": 1,
        "cached": 0,
        "hashed": 1,
        "mismatches": [{"name": "unit", "reason": "sha256 mismatch"}],
        "skipped": [],
        "elapsed_s": 0.1,
    }
    try:
        out = worker.verify_model_integrity(repo=repo)
    finally:
        wi.verify_registry = original

    assert out["ok"] is False
    actions = [a["action"] for a in repo.audit_for("")]  # case_uuid-less events
    assert "models.pin_mismatch" in actions


def test_scheduled_task_does_not_raise_on_a_broken_check():
    """A failed integrity check must never take the worker down with it."""
    import src.weight_integrity as wi
    import worker

    original = wi.verify_registry
    wi.verify_registry = lambda *a, **k: {  # type: ignore[assignment]
        "ok": False,
        "error": "unreadable registry",
        "checked": 0,
        "mismatches": [],
        "skipped": [],
        "cached": 0,
        "elapsed_s": 0.0,
    }
    try:
        out = worker.verify_model_integrity(repo=None)
    finally:
        wi.verify_registry = original
    assert out["ok"] is False
