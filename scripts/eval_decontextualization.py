"""A/B evaluation: does the ImageBind tier actually improve the system?

The premise for shipping a 2.4 GB third semantic tier is that it recovers
reuse CLIP misses. This harness tests that premise through the REAL fusion
path (`EventStore.search_hybrid`, pHash + PDQ + CLIP + ImageBind) instead of
re-implementing the matching logic:

  - corpus: real redacted samples under `reports/`, plus procedural
    distractors so false positives are measurable rather than hypothetical;
  - reuse variants: resize+re-encode, crop, text overlay, screenshot framing,
    rotation — what a reuser actually does to a file;
  - ground truth: the variant's own original event;
  - two passes, identical except for the query: with and without ImageBind.

Reports per-variant recall, absolute queries rescued by ImageBind alone, and
false positives (a distractor outranking the true match).

Honest scope: 7 images, synthetic variants, distractor corpus — this is a
relative comparison on controlled reuse, NOT a real-world recall claim. The
verdict line says so when the corpus is small.

    python scripts/eval_decontextualization.py [--ib-threshold 0.85]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

CORPUS_GLOBS = ("reports/*.redacted.png", "reports/*.png")


# ------------------------------------------------------------- fixtures


def load_corpus(max_images: int) -> list[tuple[str, Image.Image]]:
    out: list[tuple[str, Image.Image]] = []
    seen: set[Path] = set()
    for pattern in CORPUS_GLOBS:
        for p in sorted(Path().glob(pattern)):
            if p in seen:
                continue
            seen.add(p)
            try:
                with Image.open(p) as im:
                    out.append((p.name, im.convert("RGB").copy()))
            except Exception:  # unreadable fixture, skip it
                continue
            if len(out) >= max_images:
                return out
    return out


def distractors(n: int, size: int = 320) -> list[Image.Image]:
    """Unrelated images: gradients, noise, blocks, stripes.

    Nothing here should ever be reported as reuse; their similarity to any
    query is the false-positive floor for the semantic tiers.
    """
    out = []
    for i in range(n):
        rng = np.random.default_rng(1000 + i)
        kind = i % 4
        if kind == 0:
            yy, xx = np.mgrid[0:size, 0:size]
            arr = np.stack(
                [
                    (xx * 255 / size + i * 7) % 256,
                    (yy * 255 / size + i * 13) % 256,
                    ((xx + yy) % 2) * 90 + i * 5,
                ],
                axis=-1,
            ).astype(np.uint8)
        elif kind == 1:
            arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        elif kind == 2:
            arr = np.zeros((size, size, 3), np.uint8)
            for _ in range(6):
                x0, y0 = rng.integers(0, size - 40, 2)
                arr[y0 : y0 + 40, x0 : x0 + 40] = rng.integers(0, 255, 3)
        else:
            arr = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
            arr = np.dstack([arr, np.flipud(arr), np.flipud(arr) // 2])
        out.append(Image.fromarray(arr.astype(np.uint8)))
    return out


def variants(img: Image.Image) -> dict[str, Image.Image]:
    """Reuse transformations, named by what a reuser did to the file."""
    out: dict[str, Image.Image] = {}
    w, h = img.size
    out["resize_reencode"] = img.resize((max(w // 2, 32), max(h // 2, 32))).resize((w, h))
    crop = img.crop((w // 5, h // 5, w * 4 // 5, h * 4 // 5))
    out["crop"] = crop.resize((w, h))
    overlay = img.copy()
    d = ImageDraw.Draw(overlay)
    d.rectangle([0, int(h * 0.82), w, h], fill=(255, 255, 255))
    d.text((8, int(h * 0.85)), "REPOSTED - source unknown", fill=(0, 0, 0))
    out["text_overlay"] = overlay
    canvas = Image.new("RGB", (w + 40, h + 60), (245, 245, 245))
    canvas.paste(img.resize((w, h)), (20, 40))
    out["screenshot_frame"] = canvas
    out["rotate"] = img.rotate(3, expand=True, fillcolor=(128, 128, 128))
    return out


# ------------------------------------------------------------- index


def build_store(
    limit_images: int, n_distractors: int
) -> tuple[object, list[tuple[str, str, Image.Image]]]:
    """Seed the real store the way OSINT ingestion does, then build queries."""
    import imagehash

    from src.db.store import InMemoryEventStore, ProvenanceEvent
    from src.embeddings import clip_image_embedding, imagebind_image_embedding, pdq_lite_hex

    corpus = load_corpus(limit_images)
    if not corpus:
        raise SystemExit("no corpus images found under reports/ — nothing to evaluate")
    images = [im for _, im in corpus] + distractors(n_distractors)
    n_orig = len(corpus)

    store = InMemoryEventStore()
    for i, im in enumerate(images):
        store.add_event(
            ProvenanceEvent(
                event_id=f"evt-{i}",
                description=corpus[i][0] if i < n_orig else f"distractor-{i - n_orig}",
                phash_hex=str(imagehash.phash(im)),
                pdq_hex=pdq_lite_hex(pil_image=im),
                clip_embedding=clip_image_embedding(im),
                ib_embedding=imagebind_image_embedding(im),
            )
        )

    queries: list[tuple[str, str, Image.Image]] = []
    for i in range(n_orig):
        for name, vimg in variants(images[i]).items():
            queries.append((name, f"evt-{i}", vimg))
    return store, queries, n_orig


# ------------------------------------------------------------- evaluation


def run_pass(
    store,
    queries: list[tuple[str, str, Image.Image]],
    *,
    use_imagebind: bool,
    top: int,
    distractor_ids: set[str],
) -> dict:
    """One end-to-end recall pass through the production fusion code."""
    import imagehash

    from src.config import SETTINGS
    from src.db.store import HybridQuery
    from src.embeddings import clip_image_embedding, imagebind_image_embedding, pdq_lite_hex

    per_variant: dict[str, dict] = {}
    false_positives = 0

    for variant, qid, qimg in queries:
        clip = clip_image_embedding(qimg)
        ib = imagebind_image_embedding(qimg) if use_imagebind else None
        if clip is None and ib is None:
            continue
        hits = store.search_hybrid(
            HybridQuery(
                phash_hex=str(imagehash.phash(qimg)),
                phash_threshold=SETTINGS.phash_threshold,
                pdq_hex=pdq_lite_hex(pil_image=qimg),
                pdq_threshold=SETTINGS.pdq_threshold,
                embedding=clip,
                ib_embedding=ib,
                clip_threshold=SETTINGS.clip_threshold,
                ib_threshold=SETTINGS.ib_threshold,
                limit=top,
            )
        )
        stats = per_variant.setdefault(variant, {"n": 0, "hits": 0})
        stats["n"] += 1
        stats["hits"] += int(qid in {h.event.event_id for h in hits})
        # A distractor outranking the truth is what actually misleads an
        # analyst; merely appearing in the top-5 alongside it is not.
        if hits and hits[0].event.event_id in distractor_ids:
            false_positives += 1

    return {
        "use_imagebind": use_imagebind,
        "variants": {
            k: {"n": v["n"], "hits": v["hits"], "recall": round(v["hits"] / max(v["n"], 1), 3)}
            for k, v in sorted(per_variant.items())
        },
        "mean_recall": round(
            sum(v["hits"] / max(v["n"], 1) for v in per_variant.values())
            / max(len(per_variant), 1),
            3,
        ),
        "total_queries": sum(v["n"] for v in per_variant.values()),
        "total_hits": sum(v["hits"] for v in per_variant.values()),
        "false_positives": false_positives,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Decontextualization A/B benchmark")
    ap.add_argument("--images", type=int, default=12)
    ap.add_argument("--distractors", type=int, default=15)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", default="reports")
    args = ap.parse_args()

    store, queries, n_orig = build_store(args.images, args.distractors)
    distractor_ids = {
        e.event_id
        for e in getattr(store, "_events", [])
        if e.event_id not in {q[1] for q in queries}
    }
    print(f"index: {n_orig} real images + {args.distractors} distractors")
    print(
        f"queries: {len(queries)} ({n_orig} x {len(variants(Image.new('RGB', (8, 8))))} variants)"
    )

    baseline = run_pass(
        store, queries, use_imagebind=False, top=args.top, distractor_ids=distractor_ids
    )
    with_ib = run_pass(
        store, queries, use_imagebind=True, top=args.top, distractor_ids=distractor_ids
    )

    print("\n=== end-to-end recall (pHash + PDQ + CLIP [+ ImageBind]) ===")
    print(f"{'variant':<20}{'baseline':>10}{'+ImageBind':>12}{'delta':>8}")
    for variant, b in baseline["variants"].items():
        i = with_ib["variants"].get(variant, {})
        print(
            f"{variant:<20}{b['recall']:>10.3f}"
            f"{i.get('recall', float('nan')):>12.3f}"
            f"{i.get('recall', 0.0) - b['recall']:>+8.3f}"
        )
    print(
        f"{'MEAN':<20}{baseline['mean_recall']:>10.3f}{with_ib['mean_recall']:>12.3f}"
        f"{with_ib['mean_recall'] - baseline['mean_recall']:>+8.3f}"
    )
    print(
        f"\nqueries recovered: {baseline['total_hits']}/{baseline['total_queries']} -> "
        f"{with_ib['total_hits']}/{with_ib['total_queries']}"
    )
    print(
        f"false positives (distractor ranked first): baseline {baseline['false_positives']}, "
        f"+ImageBind {with_ib['false_positives']}"
    )

    from src.config import SETTINGS as _S

    report = {
        "corpus": {
            "real_images": n_orig,
            "distractors": args.distractors,
            "variants": len(variants(Image.new("RGB", (8, 8)))),
        },
        "thresholds": {"clip": _S.clip_threshold, "imagebind": _S.ib_threshold},
        "baseline": baseline,
        "with_imagebind": with_ib,
    }
    out = Path(args.out) / "eval_decontextualization.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out}")

    delta = with_ib["mean_recall"] - baseline["mean_recall"]
    rescued = with_ib["total_hits"] - baseline["total_hits"]
    fp_delta = with_ib["false_positives"] - baseline["false_positives"]
    # Sample-size honesty: a couple of queries on a handful of images is noise,
    # not evidence. Say so instead of printing a green light.
    underpowered = n_orig < 20 or rescued < 3
    print("\nVERDICT (controlled-reuse benchmark — directional, not a real-world claim):")
    if rescued > 0 and fp_delta <= 0:
        print(
            f"  ImageBind recovers {rescued} additional queries (+{delta:.3f} mean recall) with no "
            f"new false positives "
            f"(FP {baseline['false_positives']} -> {with_ib['false_positives']})."
        )
        print(
            "  NOT CONCLUSIVE at this corpus size: "
            f"{n_orig} images, {rescued} query/-ies recovered. Treat as a hint, re-run on >=20 "
            "images before enabling."
            if underpowered
            else "  Conclusive at this corpus size: keep the tier enabled."
        )
    elif rescued > 0:
        print(
            f"  ImageBind recovers {rescued} queries (+{delta:.3f}) but adds {fp_delta} false "
            "positives. Precision cost not paid for by the gain — keep disabled by default."
        )
    else:
        print(
            f"  ImageBind recovers nothing the existing tiers miss (delta {delta:+.3f}). The "
            "2.4 GB tier is not earning its cost. Keep disabled by default."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
