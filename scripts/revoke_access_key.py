#!/usr/bin/env python3
"""Revoke an MCP access key by public key_id (the segment after rrk_ in the token)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.auth.access_keys import revoke_access_key_sync  # noqa: E402
from src.config.settings import get_settings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Revoke an access key by key_id.")
    p.add_argument("key_id", help="Public key id (not the secret segment)")
    p.add_argument(
        "--db",
        default=None,
        help="SQLite path (default: ACCESS_KEYS_DB_PATH from settings / .env)",
    )
    args = p.parse_args()
    settings = get_settings()
    db_path = args.db or settings.access_keys_db_path
    if revoke_access_key_sync(db_path, args.key_id):
        print("Revoked.", file=sys.stderr)
    else:
        print("No active key found for that key_id.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
