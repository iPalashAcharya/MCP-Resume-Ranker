"""
SQLite storage for MCP access keys. Secrets are hashed with Argon2id (passlib); only
`key_id` (public lookup segment) and `secret_hash` are persisted.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_urlsafe

import aiosqlite
from passlib.context import CryptContext

TOKEN_PREFIX = "rrk_"
MCP_INVOKE_SCOPE = "mcp:invoke"

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
_dummy_hash: str | None = None


def _timing_dummy_hash() -> str:
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = _pwd_context.hash("__access_key_timing_dummy__")
    return _dummy_hash


def _ensure_parent_dir(db_path: str) -> None:
    Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def parse_access_token(raw: str) -> tuple[str, str] | None:
    """Split bearer material into (key_id, secret). Returns None if malformed."""
    s = raw.strip()
    if not s.startswith(TOKEN_PREFIX):
        return None
    body = s[len(TOKEN_PREFIX) :]
    dot = body.find(".")
    if dot <= 0 or dot == len(body) - 1:
        return None
    key_id, secret = body[:dot], body[dot + 1 :]
    if not key_id or not secret:
        return None
    return key_id, secret


def format_full_token(key_id: str, secret: str) -> str:
    return f"{TOKEN_PREFIX}{key_id}.{secret}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS access_keys (
    key_id TEXT PRIMARY KEY,
    secret_hash TEXT NOT NULL,
    label TEXT,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
"""


async def init_access_keys_db(db_path: str) -> None:
    """Create database file, apply schema, enable WAL."""
    _ensure_parent_dir(db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA synchronous=NORMAL")
        await db.executescript(SCHEMA)
        await db.commit()


def init_access_keys_db_sync(db_path: str) -> None:
    _ensure_parent_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(SCHEMA)
        conn.commit()


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def issue_access_key_sync(db_path: str, label: str | None = None) -> tuple[str, str]:
    """
    Insert a new key. Returns (key_id, full_token). The full token is shown once to the operator.
    """
    init_access_keys_db_sync(db_path)
    key_id = token_urlsafe(12)
    secret = token_urlsafe(32)
    secret_hash = _pwd_context.hash(secret)
    created = _utc_now_iso()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO access_keys (key_id, secret_hash, label, created_at, revoked_at) "
            "VALUES (?, ?, ?, ?, NULL)",
            (key_id, secret_hash, label, created),
        )
        conn.commit()
    return key_id, format_full_token(key_id, secret)


async def issue_access_key(db_path: str, label: str | None = None) -> tuple[str, str]:
    await init_access_keys_db(db_path)
    key_id = token_urlsafe(12)
    secret = token_urlsafe(32)
    secret_hash = _pwd_context.hash(secret)
    created = _utc_now_iso()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "INSERT INTO access_keys (key_id, secret_hash, label, created_at, revoked_at) "
            "VALUES (?, ?, ?, ?, NULL)",
            (key_id, secret_hash, label, created),
        )
        await db.commit()
    return key_id, format_full_token(key_id, secret)


def revoke_access_key_sync(db_path: str, key_id: str) -> bool:
    """Set revoked_at. Returns True if a row was updated."""
    init_access_keys_db_sync(db_path)
    now = _utc_now_iso()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE access_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
            (now, key_id),
        )
        conn.commit()
        return cur.rowcount > 0


async def verify_access_key_async(db_path: str, token: str) -> tuple[bool, str | None]:
    """
    Validate token. Returns (ok, key_id) where key_id is safe to log; secret never returned.
    On failure, key_id is None. Runs a dummy passlib verify when the key is missing/revoked
    to reduce timing skew vs a wrong password for an existing key.
    """
    parsed = parse_access_token(token)
    if not parsed:
        _pwd_context.verify("invalid", _timing_dummy_hash())
        return False, None

    key_id, secret = parsed
    row: dict[str, str | None] | None = None
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT key_id, secret_hash, revoked_at FROM access_keys WHERE key_id = ?",
            (key_id,),
        ) as cur:
            r = await cur.fetchone()
            if r is not None:
                row = dict(r)

    active = row is not None and row.get("revoked_at") is None
    hash_to_check = row["secret_hash"] if active else _timing_dummy_hash()
    secret_to_check = secret if active else "invalid"
    ok = _pwd_context.verify(secret_to_check, hash_to_check)
    if not active or not ok:
        return False, None
    return True, key_id
