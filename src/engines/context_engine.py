"""Engine 4 — Context & decontextualization (production wrapper).

Wraps `src/modules/context_check.py` hybrid retrieval and adds the intel
summary: top-hit citation, tier breakdown (phash/pdq/semantic), and a
recaption-risk statement for STIX/API consumers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.modules.context_check import check_context
from src.schemas import ContextResult

log = logging.getLogger(__name__)


def run_context_engine(
    file_path: str | Path, threshold: int | None = None, store: Any | None = None
) -> tuple[ContextResult, dict[str, Any]]:
    result = check_context(file_path, threshold=threshold, store=store)
    tiers: dict[str, int] = {}
    for m in result.matches:
        tiers[getattr(m, "match_type", "phash")] = (
            tiers.get(getattr(m, "match_type", "phash"), 0) + 1
        )
    summary = {
        "decontextualization": (
            "LIKELY"
            if result.is_decontextualized and result.matches and result.matches[0].confidence > 0.8
            else "POSSIBLE"
            if result.is_decontextualized
            else "NOT_OBSERVED"
        ),
        "top_hit": (result.matches[0].event_id if result.matches else None),
        "tiers": tiers,
        "semantic_tier": "active" if result.clip_available else "offline-hash-only",
    }
    log.info("context-engine: %s %s", Path(str(file_path)).name, summary["decontextualization"])
    return result, summary
