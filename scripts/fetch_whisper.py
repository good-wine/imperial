"""Provision Whisper weights on the CONNECTED station for air-gapped use.

Usage:
    python scripts/fetch_whisper.py --model tiny    # ~75 MB, field default
    python scripts/fetch_whisper.py --model small   # ~244 MB, better dialects
    # then copy models/whisper/ to the air-gapped host (WHISPER_ROOT).

Without provisioned weights the speech leg reports SKIPPED offline — never
a fabricated transcript.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:  # type: ignore[no-untyped-def]
    ap = argparse.ArgumentParser(description="Fetch Whisper weights for air-gap")
    ap.add_argument(
        "--model", default="tiny", choices=["tiny", "base", "small", "medium", "large-v3", "turbo"]
    )
    ap.add_argument("--root", default="models/whisper")
    args = ap.parse_args()
    try:
        import whisper
    except ImportError:
        print("openai-whisper not installed (pip install openai-whisper)", file=sys.stderr)
        return 1
    Path(args.root).mkdir(parents=True, exist_ok=True)
    whisper.load_model(args.model, download_root=args.root)
    print(f"whisper '{args.model}' provisioned under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
