"""ImageBind joint-space tier: shape, gating, coercion, fusion.

No model inference here beyond a marked integration check: the 2.4 GB ONNX is
provisioned separately and skipped when absent, so the suite stays fast and
air-gap friendly. What is always asserted is the FAIL-CLOSED plumbing — a
wrong-dimension vector must never reach the store, and an unsigned or
tampered model must never load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.db.store import (
    HybridQuery,
    InMemoryEventStore,
    ProvenanceEvent,
    _coerce_event,
    cosine,
)

IB_DIM = 1024
CLIP_DIM = 512


def _unit_vec(dim: int, seed: int = 1) -> list[float]:
    v = np.random.default_rng(seed).random(dim).astype(float)
    return (v / np.linalg.norm(v)).tolist()


# ------------------------------------------------------------- vector coercion


def test_coerce_rejects_wrong_dimension():
    """A 512-d vector in the ImageBind slot is a bug, not a match: drop it."""
    assert (
        _coerce_event({"event_id": "a", "ib_embedding": _unit_vec(CLIP_DIM)}).ib_embedding is None
    )
    assert (
        _coerce_event({"event_id": "b", "clip_embedding": _unit_vec(IB_DIM)}).clip_embedding is None
    )


def test_coerce_rejects_non_finite_and_junk():
    """NaN/inf would poison every cosine score: drop the whole vector."""
    assert (
        _coerce_event({"event_id": "n", "ib_embedding": [float("nan")] * IB_DIM}).ib_embedding
        is None
    )
    assert (
        _coerce_event({"event_id": "i", "ib_embedding": [float("inf")] * IB_DIM}).ib_embedding
        is None
    )
    assert _coerce_event({"event_id": "d", "ib_embedding": ["x"] * IB_DIM}).ib_embedding is None
    assert _coerce_event({"event_id": "b2", "ib_embedding": [True] * IB_DIM}).ib_embedding is None
    assert _coerce_event({"event_id": "e"}).ib_embedding is None
    assert _coerce_event({"event_id": "f", "ib_embedding": {}}).ib_embedding is None


# ------------------------------------------------------------- search fusion


def test_ib_tier_matches_without_hash_overlap():
    """Hash tiers disagree, ImageBind agrees: the semantic tier must surface it."""
    store = InMemoryEventStore()
    vec = _unit_vec(IB_DIM, seed=3)
    store.add_event(
        ProvenanceEvent(
            event_id="e1",
            description="irrelevant wording",
            phash_hex="0000000000000000",
            ib_embedding=vec,
        )
    )
    hits = store.search_hybrid(
        HybridQuery(phash_hex="ffffffffffffffff", ib_embedding=vec, clip_threshold=0.90)
    )
    assert [h.event.event_id for h in hits] == ["e1"]
    assert hits[0].match_type == "semantic"
    assert hits[0].clip_similarity == pytest.approx(1.0, abs=1e-3)


def test_ib_tier_respects_threshold():
    store = InMemoryEventStore()
    store.add_event(
        ProvenanceEvent(event_id="far", description="d", ib_embedding=_unit_vec(IB_DIM, seed=4))
    )
    hits = store.search_hybrid(
        HybridQuery(ib_embedding=_unit_vec(IB_DIM, seed=5), clip_threshold=0.999)
    )
    assert hits == []


def test_best_tier_wins_and_clip_is_independent():
    """Each event carries both vectors; a match on only one still surfaces."""
    store = InMemoryEventStore()
    clip_v = _unit_vec(CLIP_DIM, seed=6)
    ib_v = _unit_vec(IB_DIM, seed=6)
    store.add_event(ProvenanceEvent(event_id="both", description="d", clip_embedding=clip_v))
    store.add_event(ProvenanceEvent(event_id="ibonly", description="d", ib_embedding=ib_v))
    hits = store.search_hybrid(
        HybridQuery(embedding=clip_v, ib_embedding=ib_v, clip_threshold=0.999)
    )
    found = {h.event.event_id for h in hits}
    assert found == {"both", "ibonly"}


def test_hash_tier_is_not_overwritten_by_weaker_semantic():
    """An exact pHash match outranks a merely similar semantic hit."""
    store = InMemoryEventStore()
    store.add_event(
        ProvenanceEvent(event_id="exact", description="d", phash_hex="abcdef0123456789")
    )
    hits = store.search_hybrid(
        HybridQuery(
            phash_hex="abcdef0123456789", phash_threshold=4, ib_embedding=_unit_vec(IB_DIM, 8)
        )
    )
    assert hits[0].match_type == "phash"
    assert hits[0].confidence == pytest.approx(1.0)


def test_hash_only_events_never_need_the_ib_slot():
    store = InMemoryEventStore()
    store.add_event(ProvenanceEvent(event_id="h", description="d", phash_hex="aabbccddeeff0011"))
    hits = store.search_hybrid(HybridQuery(phash_hex="aabbccddeeff0011", phash_threshold=2))
    assert [h.event.event_id for h in hits] == ["h"]
    assert hits[0].event.ib_embedding is None


# ------------------------------------------------------------- model gating


def test_unsigned_model_is_refused(tmp_path, monkeypatch):
    """Fails closed: a model blob without a .sig must not produce embeddings."""
    from src import embeddings

    fake = tmp_path / "imagebind_vision.onnx"
    fake.write_bytes(b"not-a-real-model")
    embeddings._IB_SESSION = None
    embeddings._IB_LOADED = False
    monkeypatch.setenv("IMAGEBIND_ENABLED", "1")
    monkeypatch.setenv("MODEL_SIG_ENFORCE", "1")
    monkeypatch.setenv("MODEL_ED25519_PUBKEY", "00" * 32)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    from PIL import Image

    assert embeddings.imagebind_image_embedding(Image.fromarray(img), str(fake)) is None


def test_disabled_by_env_returns_none_without_loading(tmp_path, monkeypatch):
    from PIL import Image

    from src import embeddings

    fake = tmp_path / "imagebind_vision.onnx"
    fake.write_bytes(b"not-a-real-model")
    embeddings._IB_SESSION = None
    embeddings._IB_LOADED = False
    monkeypatch.setenv("IMAGEBIND_ENABLED", "0")
    assert (
        embeddings.imagebind_image_embedding(
            Image.fromarray(np.zeros((8, 8, 3), np.uint8)), str(fake)
        )
        is None
    )
    assert embeddings._IB_LOADED is False  # never even attempted a load


def test_missing_model_file_is_not_fatal(tmp_path, monkeypatch):
    from PIL import Image

    from src import embeddings

    embeddings._IB_SESSION = None
    embeddings._IB_LOADED = False
    monkeypatch.setenv("IMAGEBIND_ENABLED", "1")
    monkeypatch.delenv("MODEL_SIG_ENFORCE", raising=False)
    monkeypatch.delenv("MODEL_ED25519_PUBKEY", raising=False)
    img = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
    assert embeddings.imagebind_image_embedding(img, str(tmp_path / "absent.onnx")) is None


# ------------------------------------------------------------- registry


def test_registry_pins_the_imagebind_model():
    reg = json.loads(Path("models/registry.json").read_text(encoding="utf-8"))
    rows = {m["name"]: m for m in reg["models"]}
    ib = rows.get("imagebind-vision")
    if ib is None:
        pytest.skip("ImageBind not provisioned in this checkout")
    assert ib["path"].endswith("imagebind_vision.onnx")
    assert len(ib["sha256"]) == 64


def test_cosine_of_identical_vectors_is_one():
    v = _unit_vec(IB_DIM, seed=11)
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------- integration


@pytest.mark.slow
def test_real_model_embedding_roundtrip(monkeypatch):
    """Integration: only runs where the signed 2.4 GB model is present."""
    from PIL import Image

    from src import embeddings

    pub = Path("config/model_signing/ed25519.pub.hex")
    if not pub.is_file() or not Path("models/imagebind_vision.onnx").is_file():
        pytest.skip("ImageBind ONNX not provisioned")
    monkeypatch.setenv("IMAGEBIND_ENABLED", "1")
    monkeypatch.setenv("MODEL_SIG_ENFORCE", "1")
    monkeypatch.setenv("MODEL_ED25519_PUBKEY", pub.read_text(encoding="utf-8").strip())
    embeddings._IB_SESSION = None
    embeddings._IB_LOADED = False
    img = Image.fromarray((np.random.default_rng(3).random((256, 256, 3)) * 255).astype(np.uint8))
    vec = embeddings.imagebind_image_embedding(img)
    assert vec is not None
    assert len(vec) == IB_DIM
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-3)
    again = embeddings.imagebind_image_embedding(img)
    assert cosine(vec, again) == pytest.approx(1.0, abs=1e-5)
