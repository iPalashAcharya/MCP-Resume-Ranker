#!/usr/bin/env python3
"""Create a new MCP access key and print it once (stdout). Requires PYTHONPATH=project root."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root on sys.path when run as python scripts/issue_access_key.py
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.auth.access_keys import issue_access_key_sync  # noqa: E402
from src.config.settings import get_settings  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Issue an MCP HTTP access key (stored hashed in SQLite).")
    p.add_argument("--label", default=None, help="Optional human-readable label")
    p.add_argument(
        "--db",
        default=None,
        help="SQLite path (default: ACCESS_KEYS_DB_PATH from settings / .env)",
    )
    args = p.parse_args()
    settings = get_settings()
    db_path = args.db or settings.access_keys_db_path
    key_id, token = issue_access_key_sync(db_path, label=args.label)
    print(f"key_id: {key_id}")
    print(f"token:  {token}")
    print("\nStore the token securely; it cannot be retrieved again.", file=sys.stderr)


if __name__ == "__main__":
    main()
