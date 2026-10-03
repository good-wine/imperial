"""RSS/Atom crisis-feed connector (stdlib-only, no feedparser needed).

Polls a list of feed URLs (`OSINT_RSS_FEEDS`, comma-separated), extracts
items with media enclosures (`enclosure`, `media:content`, or image links in
`description`), and yields normalized records. Deduplication by GUID/link via
a tiny on-disk seen-set (`data/osint_seen.json`). Async-friendly: blocking
HTTP runs in `asyncio.to_thread`.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

log = logging.getLogger(__name__)

_IMG_RE = re.compile(r'<img[^>]+src=["\']([^"\']+)["\']', re.IGNORECASE)
_LINK_MEDIA_RE = re.compile(r"https?://\S+\.(?:jpg|jpeg|png|mp4|mov|webp)(?:\?\S*)?", re.IGNORECASE)


@dataclass
class FeedItem:
    source: str
    external_id: str
    title: str
    link: str
    published: str
    media_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fetch(url: str, timeout_s: int = 20) -> bytes:
    from src.security import fetch_url

    return fetch_url(
        url,
        headers={"User-Agent": "ProvenanceEngine-OSINT/3.0"},
        timeout_s=timeout_s,
    )


def parse_feed(raw: bytes, source: str) -> list[FeedItem]:
    """Parse RSS 2.0 / Atom. Never raises on malformed XML (skips bad items)."""
    try:
        root = ET.fromstring(raw)
    except (ET.ParseError, DefusedXmlException) as exc:
        log.warning("rss: unparsable feed %s (%s)", source, exc)
        return []
    items: list[FeedItem] = []
    ns = {"media": "http://search.yahoo.com/mrss/", "atom": "http://www.w3.org/2005/Atom"}
    for entry in list(root.iter("item")) + list(root.iter(f"{{{ns['atom']}}}entry")):
        try:
            title = (entry.findtext("title") or entry.findtext(f"{{{ns['atom']}}}title") or "")[
                :300
            ]
            link = (entry.findtext("link") or "")[:500]
            for link_el in entry.findall(f"{{{ns['atom']}}}link"):
                if link_el.get("href"):
                    link = link_el.get("href", "")[:500]
            guid = (entry.findtext("guid") or entry.findtext(f"{{{ns['atom']}}}id") or link)[:500]
            pub = (entry.findtext("pubDate") or entry.findtext(f"{{{ns['atom']}}}updated") or "")[
                :64
            ]
            desc = (entry.findtext("description") or "") + (entry.findtext("summary") or "")
            media: list[str] = [
                enc.get("url", "")[:500] for enc in entry.findall("enclosure") if enc.get("url")
            ]
            media.extend(
                mc.get("url", "")[:500]
                for mc in entry.findall("media:content", ns) + entry.findall("media:thumbnail", ns)
                if mc.get("url")
            )
            media += _IMG_RE.findall(desc)[:4] + _LINK_MEDIA_RE.findall(html.unescape(desc))[:4]
            items.append(
                FeedItem(
                    source, guid or link, html.unescape(title), link, pub, sorted(set(media))[:6]
                )
            )
        except Exception as exc:  # noqa: PERF203 - per-item skip must stay in loop
            log.debug("rss: skipping malformed item (%s)", exc)
    return items


async def poll_feeds(
    feed_urls: list[str], seen_path: str = "data/osint_seen.json", timeout_s: int = 20
) -> list[FeedItem]:
    """Fetch all feeds concurrently; return only unseen items with media."""
    seen: set[str] = set()
    sp = Path(seen_path)
    try:
        if await asyncio.to_thread(sp.is_file):
            seen = set(json.loads(await asyncio.to_thread(sp.read_text, encoding="utf-8")))
    except Exception:
        pass
    raw_feeds = await asyncio.gather(*[_fetch_async(u, timeout_s) for u in feed_urls])
    fresh: list[FeedItem] = []
    for url, raw in zip(feed_urls, raw_feeds, strict=True):
        if not raw:
            continue
        for item in parse_feed(raw, source=url):
            key = f"{item.source}|{item.external_id}"
            if key not in seen and item.media_urls:
                seen.add(key)
                fresh.append(item)
    try:
        await asyncio.to_thread(sp.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(
            sp.write_text, json.dumps(sorted(seen)[-5000:], ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("rss: seen-set persist failed (%s)", exc)
    log.info("rss: %d fresh media item(s) from %d feed(s)", len(fresh), len(feed_urls))
    return fresh


async def _fetch_async(url: str, timeout_s: int) -> bytes:
    try:
        return await asyncio.to_thread(_fetch, url, timeout_s)
    except Exception as exc:
        log.warning("rss: fetch failed %s (%s)", url, exc)
        return b""
