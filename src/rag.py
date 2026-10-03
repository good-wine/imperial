"""Minimal offline RAG: keyword retrieval over reference events + past reports.

No embeddings required (works fully air-gapped with zero models): scores
candidate snippets by token overlap with the case facts, returns top-k with
source labels for citation. When CLIP/LLM embeddings exist they only
re-rank; retrieval never depends on them.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _read_past_report(md: Path) -> tuple[str, str] | None:
    """Read one past-report snippet; None when unreadable (fail-open)."""
    try:
        txt = md.read_text(encoding="utf-8")[:1500]
        return (f"past-report:{md.stem}", txt)
    except Exception:
        return None


def retrieve_snippets(
    case_text: str,
    top_k: int = 3,
    reference_db: str = "data/reference_events.json",
    reports_dir: str = "reports",
) -> list[dict[str, str]]:
    """Return [{source, text}] grounded snippets for the LLM prompt."""
    query = _tokens(case_text)
    if not query:
        return []
    candidates: list[tuple[str, str]] = []
    try:
        raw = json.loads(Path(reference_db).read_text(encoding="utf-8"))
        candidates.extend(
            (
                f"reference:{row.get('event_id', '?')}",
                (
                    f"{row.get('event_id')} ({row.get('event_date', '?')}): "
                    f"{row.get('description', '')}"
                ),
            )
            for row in raw
            if isinstance(row, dict)
        )
    except Exception as exc:
        log.debug("rag: reference db unreadable (%s)", exc)
    try:
        for md in sorted(Path(reports_dir).glob("*.report.md"))[-20:]:
            snippet = _read_past_report(md)
            if snippet is not None:
                candidates.append(snippet)
    except Exception:
        pass
    scored = []
    for source, text in candidates:
        overlap = len(query & _tokens(text))
        if overlap:
            scored.append((overlap, source, text[:600]))
    scored.sort(reverse=True)
    return [{"source": s, "text": t} for _, s, t in scored[: max(top_k, 0)]]
