"""OSINT ingestion pipeline: feeds/channels -> cases -> analysis queue.

`run_once()` polls RSS (+ Telegram Bot API), downloads unseen media, creates
TLP-tagged cases (default TLP:AMBER for unvetted OSINT — never CLEAR), and
enqueues analysis without blocking the caller. Designed to run as a Celery
beat task (`worker.osint_poll`) or a standalone asyncio loop in the field.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _download(url: str, dest: Path, timeout_s: int = 60) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        from src.security import download_to_file

        size = download_to_file(
            url,
            dest,
            headers={"User-Agent": "ProvenanceEngine-OSINT/3.0"},
            timeout_s=timeout_s,
        )
        return size > 0
    except Exception as exc:
        log.warning("osint: download failed %s (%s)", url[:120], exc)
        return False


async def run_once(owner: str = "osint", tlp: str = "TLP:AMBER") -> list[dict[str, Any]]:
    """One poll cycle. Returns created case summaries."""
    from ingestion.rss import poll_feeds
    from ingestion.telegram import download_media, poll_bot_api
    from src.database import get_repository
    from src.security import guess_mime, sha256_of
    from worker import enqueue_analysis

    created: list[dict[str, Any]] = []
    repo = get_repository()

    feeds = [u.strip() for u in os.getenv("OSINT_RSS_FEEDS", "").split(",") if u.strip()]
    if feeds:
        for item in await poll_feeds(feeds):
            for url in item.media_urls[:3]:
                dest = Path("data/osint") / Path(url.split("?")[0]).name.replace("/", "_")
                if not await asyncio.to_thread(_download, url, dest):
                    continue
                case = repo.create_case(
                    filename=dest.name,
                    sha256=sha256_of(dest),
                    file_size=dest.stat().st_size,
                    mime_type=guess_mime(dest),
                    tlp=tlp,
                    owner=owner,
                    storage_ref=f"osint|rss|{item.source}|{item.external_id}|{url}",
                )
                task = enqueue_analysis(case["case_uuid"], str(dest))
                repo.set_status(case["case_uuid"], "queued", task_id=task)
                created.append(
                    {"case_uuid": case["case_uuid"], "via": "rss", "title": item.title[:120]}
                )

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chats = [c.strip() for c in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if c.strip()]
    if token:
        for tg in await poll_bot_api(token, chats):
            tg_dest = await download_media(token, tg, "data/osint")
            if tg_dest is None:
                continue
            case = repo.create_case(
                filename=tg_dest.name,
                sha256=sha256_of(tg_dest),
                file_size=tg_dest.stat().st_size,
                mime_type=guess_mime(tg_dest),
                tlp=tlp,
                owner=owner,
                storage_ref=f"osint|telegram|{tg.source}|{tg.external_id}",
            )
            task = enqueue_analysis(case["case_uuid"], str(tg_dest))
            repo.set_status(case["case_uuid"], "queued", task_id=task)
            created.append(
                {"case_uuid": case["case_uuid"], "via": "telegram", "title": tg.caption[:120]}
            )
    log.info("osint: cycle created %d case(s)", len(created))
    return created
