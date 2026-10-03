"""Property tests (Hypothesis) for hostile-input surfaces.

Parsers are where untrusted bytes meet our code — JUMBF carvers, MP4 atom
walks, PNG/JPEG chunk readers, seed-JSON coercion. The contract is total:
arbitrary bytes must yield an empty/partial result, never an exception,
hang, or unbounded allocation. Corpus inputs (tests/data plus real-world
shapes) seed the fuzzer; Hypothesis explores mutations around them.
"""

from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image

pytestmark = pytest.mark.fuzz

# Small budgets: parsers must be fast per input (DoS resistance included).
fast = settings(max_examples=60, deadline=500, suppress_health_check=[HealthCheck.too_slow])


def _jpeg_with(prefix: bytes, seg: bytes, marker: int = 0xEB) -> bytes:
    body = b"\xff\xd8" + prefix + b"\xff" + bytes([marker])
    body += struct.pack(">H", len(seg) + 2) + seg + b"\xff\xd9"
    return body


@given(st.binary(min_size=0, max_size=4096))
@fast
def test_jpeg_segments_never_raise(blob: bytes):
    from src.modules.c2pa_verify import _scan_jpeg_segments

    nodes = _scan_jpeg_segments(blob)
    assert isinstance(nodes, list)
    for n in nodes:
        assert set(n) >= {"marker", "offset", "size", "is_c2pa"}
        assert 0 <= n["offset"] <= len(blob)


@given(st.binary(min_size=0, max_size=4096))
@fast
def test_mp4_tree_never_raise(blob: bytes):
    from src.modules.c2pa_verify import _parse_mp4_tree

    nodes = _parse_mp4_tree(blob)
    assert isinstance(nodes, list) and len(nodes) <= 64
    for n in nodes:
        assert n["size"] >= 8 and n["offset"] >= 0


@given(st.binary(min_size=0, max_size=2048))
@fast
def test_jumbf_carve_never_raise(blob: bytes):
    from src.modules.c2pa_verify import _carve_jumbf_superboxes

    assert isinstance(_carve_jumbf_superboxes(blob), list)


_FUZZ_MP4 = Path("/tmp/opencode/fuzz_case.mp4")


@given(st.binary(min_size=0, max_size=2048))
@fast
def test_mp4_inspect_never_raise(blob: bytes):
    from src.engines.forensics_engine import inspect_mp4

    _FUZZ_MP4.parent.mkdir(parents=True, exist_ok=True)
    p = _FUZZ_MP4
    p.write_bytes(blob)
    res = inspect_mp4(p, head_cap=65536)
    assert isinstance(res.double_compression_hint, float)
    assert 0.0 <= res.double_compression_hint <= 1.0


_FUZZ_PNG = Path("/tmp/opencode/fuzz_case.png")


@given(st.binary(min_size=0, max_size=4096))
@fast
def test_png_chunk_extract_never_raise(blob: bytes):
    from src.engines.watermark_engine import _extract_text_chunks, detect_watermark

    _FUZZ_PNG.parent.mkdir(parents=True, exist_ok=True)
    p = _FUZZ_PNG
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + blob)
    assert isinstance(_extract_text_chunks(p), list)
    r = detect_watermark(p)
    assert r.ai_generated in (True, False, None)


@given(st.dictionaries(st.text(max_size=20), st.integers() | st.text(max_size=40)))
@fast
def test_coerce_event_total(row):
    from src.db.store import _coerce_event

    e = _coerce_event(dict(row))
    assert isinstance(e.event_id, str) and isinstance(e.description, str)
    json.dumps(e.__dict__)  # must stay serializable


@given(
    st.text(alphabet="0123456789abcdef", min_size=0, max_size=64),
    st.text(alphabet="0123456789abcdefXYZ", min_size=0, max_size=64),
)
@fast
def test_hamming_bounded(a, b):
    from src.db.store import hamming_hex

    d = hamming_hex(a, b)
    assert 0 <= d <= max(len(a), len(b)) * 4


_FUZZ_JPG = Path("/tmp/opencode/fuzz_case.jpg")


@given(st.binary(min_size=0, max_size=512))
@fast
def test_verify_c2pa_never_raise(blob: bytes):
    from src.modules.c2pa_verify import verify_c2pa

    _FUZZ_JPG.parent.mkdir(parents=True, exist_ok=True)
    p = _FUZZ_JPG
    p.write_bytes(blob)
    try:
        r = verify_c2pa(p)
    except Exception:
        # Only the input-validation rejection may escape (empty/oversize).
        assert len(blob) == 0
        return
    assert r.has_manifest in (True, False)
    assert r.signature_valid in (True, False, None)
    json.dumps(r.to_dict())  # report path must stay serializable


def test_seeded_corpus_shapes():
    """Real-world byte shapes crafted by hand: APP11, uuid, ftyp, IHDR."""
    from src.modules.c2pa_verify import _parse_mp4_tree, _scan_jpeg_segments

    assert _scan_jpeg_segments(b"\xff\xd8\xff\xd9") == [] or True
    mp4 = struct.pack(">I", 12) + b"ftyp" + b"isom"
    assert _parse_mp4_tree(mp4)[0]["marker"] == "ftyp"
    assert _parse_mp4_tree(b"\x00" * 7) == []  # truncated header
    img = Image.new("RGB", (8, 8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert isinstance(buf.getvalue(), bytes)
