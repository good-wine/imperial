"""Telegram crisis-channel connector (Bot API stdlib path + Telethon upgrade).

Two tiers:
  stdlib (default)  Bot API `getUpdates` long-poll on `TELEGRAM_BOT_TOKEN`
                    + `TELEGRAM_CHAT_IDS`: downloads photo/document media.
                    Zero extra dependencies, works air-gapped-adjacent.
  telethon (opt-in) `TELEGRAM_USE_TELETHON=1` with `telethon` installed:
                    full channel-history scraping via API ID/hash
                    (`TELEGRAM_API_ID`, `TELEGRAM_API_HASH`, session file).
                    Needs prior interactive login — documented, not automatic.

Every download is hashed immediately; records carry channel/message provenance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TgMedia:
    source: str  # channel/chat title or id
    external_id: str  # message id
    caption: str
    file_id: str
    mime_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _api(token: str, method: str, params: dict[str, Any] | None = None, timeout_s: int = 25) -> Any:
    """Bot API call. Returns the raw `result` payload (list for getUpdates,
    dict for getFile/getMe) — hence Any, narrowed by each caller."""
    data = urllib.parse.urlencode(params or {}).encode() if params else None
    from src.security import fetch_url

    raw = fetch_url(
        f"https://api.telegram.org/bot{token}/{method}",
        method="POST" if data else "GET",
        data=data,
        timeout_s=timeout_s,
    )
    body = json.loads(raw.decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(f"telegram api error: {body}")
    return body["result"]


async def poll_bot_api(token: str, chat_ids: list[str], limit: int = 50) -> list[TgMedia]:
    """Long-poll Bot API updates, keep photo/document messages for listed chats."""
    if not token:
        log.info("telegram: no bot token — connector idle")
        return []
    try:
        updates = await asyncio.to_thread(
            _api, token, "getUpdates", {"limit": limit, "timeout": 10}
        )
    except Exception as exc:
        log.warning("telegram: poll failed (%s)", exc)
        return []
    out: list[TgMedia] = []
    for upd in updates:
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        if (
            chat_ids
            and chat not in chat_ids
            and (msg.get("chat") or {}).get("title", "") not in chat_ids
        ):
            continue
        file_id, mime = "", ""
        if msg.get("photo"):
            file_id = msg["photo"][-1].get("file_id", "")  # largest variant
        elif isinstance(msg.get("document"), dict):
            doc = msg["document"]
            if (doc.get("mime_type") or "").startswith(("image/", "video/")):
                file_id, mime = doc.get("file_id", ""), doc.get("mime_type", "")
        if file_id:
            out.append(
                TgMedia(
                    source=str((msg.get("chat") or {}).get("title", chat)),
                    external_id=f"msg-{msg.get('message_id')}",
                    caption=str(msg.get("caption", ""))[:500],
                    file_id=file_id,
                    mime_hint=mime,
                )
            )
    log.info("telegram: %d media message(s)", len(out))
    return out


async def download_media(token: str, item: TgMedia, dest_dir: str | Path) -> Path | None:
    """Resolve file_path via Bot API and download. Returns local path or None."""
    try:
        info = await asyncio.to_thread(_api, token, "getFile", {"file_id": item.file_id})
        fpath = info.get("file_path", "")
        url = f"https://api.telegram.org/file/bot{token}/{fpath}"
        dest = Path(dest_dir) / f"tg_{item.external_id}_{Path(fpath).name}".replace("/", "_")
        dest.parent.mkdir(parents=True, exist_ok=True)

        def _dl() -> None:
            from src.security import download_to_file

            download_to_file(
                url,
                dest,
                headers={"User-Agent": "ProvenanceEngine-OSINT/3.0"},
                timeout_s=60,
            )

        await asyncio.to_thread(_dl)
        return dest
    except Exception as exc:
        log.warning("telegram: download failed %s (%s)", item.external_id, exc)
        return None


async def scrape_telethon(channels: list[str]) -> list[TgMedia]:
    """Opt-in Telethon history scrape. Raises a clear error when unconfigured."""
    import os

    if os.getenv("TELEGRAM_USE_TELETHON", "0") != "1":
        raise RuntimeError("telethon path disabled (TELEGRAM_USE_TELETHON!=1)")
    try:
        from telethon import TelegramClient
    except ImportError as exc:
        raise RuntimeError("telethon not installed (pip install telethon)") from exc
    api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
    api_hash = os.getenv("TELEGRAM_API_HASH", "")
    if not (api_id and api_hash):
        raise RuntimeError("TELEGRAM_API_ID/HASH required for telethon")
    out: list[TgMedia] = []
    async with TelegramClient("data/tg_session", api_id, api_hash) as client:
        for ch in channels:
            # NOTE: explicit loop (not extend/genexpr) — async generators
            # are not valid arguments to list.extend (TypeError at runtime).
            async for msg in client.iter_messages(ch, limit=100):
                if not (
                    msg.photo
                    or (
                        msg.document
                        and (msg.document.mime_type or "").startswith(("image/", "video/"))
                    )
                ):
                    continue
                out.append(
                    TgMedia(
                        source=str(ch),
                        external_id=f"msg-{msg.id}",
                        caption=str(msg.text or "")[:500],
                        file_id="",
                        mime_hint=str(getattr(msg.document, "mime_type", "")),
                    )
                )  # NOTE: explicit loop — async generators are not valid list.extend() args
    return out
