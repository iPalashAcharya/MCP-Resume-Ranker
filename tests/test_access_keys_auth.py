"""Tests for SQLite access keys, verifier, and admin issuance route."""
from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from src.auth.access_keys import (
    init_access_keys_db_sync,
    issue_access_key_sync,
    parse_access_token,
    revoke_access_key_sync,
    verify_access_key_async,
)
from src.auth.admin_routes import AdminRateLimiter, make_admin_issue_handler
from src.auth.verifier import SQLiteAccessKeyVerifier


def test_parse_access_token_roundtrip(tmp_path) -> None:
    db = str(tmp_path / "keys.db")
    init_access_keys_db_sync(db)
    key_id, token = issue_access_key_sync(db, label="t")
    parsed = parse_access_token(token)
    assert parsed is not None
    assert parsed[0] == key_id
    assert token.startswith(f"rrk_{key_id}.")


@pytest.mark.asyncio
async def test_verify_access_key_async(tmp_path) -> None:
    db = str(tmp_path / "keys.db")
    init_access_keys_db_sync(db)
    _, token = issue_access_key_sync(db)
    ok, kid = await verify_access_key_async(db, token)
    assert ok and kid
    ok_bad, _ = await verify_access_key_async(db, "rrk_nope.xxx")
    assert not ok_bad
    ok_wrong, _ = await verify_access_key_async(db, token + "x")
    assert not ok_wrong


@pytest.mark.asyncio
async def test_verify_revoked_key(tmp_path) -> None:
    db = str(tmp_path / "keys.db")
    init_access_keys_db_sync(db)
    key_id, token = issue_access_key_sync(db)
    assert revoke_access_key_sync(db, key_id)
    ok, _ = await verify_access_key_async(db, token)
    assert not ok


@pytest.mark.asyncio
async def test_sqlite_access_key_verifier(tmp_path) -> None:
    db = str(tmp_path / "keys.db")
    init_access_keys_db_sync(db)
    _, token = issue_access_key_sync(db)
    v = SQLiteAccessKeyVerifier(db)
    at = await v.verify_token(token)
    assert at is not None
    assert at.client_id
    assert "mcp:invoke" in at.scopes
    assert await v.verify_token("bad") is None


def test_admin_issue_route_auth(tmp_path) -> None:
    db = str(tmp_path / "keys.db")
    init_access_keys_db_sync(db)
    secret = "bootstrap-secret-test-value"
    limiter = AdminRateLimiter(30)
    handler = make_admin_issue_handler(db, secret, limiter)
    app = Starlette(routes=[Route("/admin/access-keys", endpoint=handler, methods=["POST"])])

    with TestClient(app) as client:
        assert client.post("/admin/access-keys", json={}).status_code == 401
        assert (
            client.post(
                "/admin/access-keys",
                headers={"Authorization": "Bearer wrong"},
                json={},
            ).status_code
            == 401
        )
        r = client.post(
            "/admin/access-keys",
            headers={"Authorization": f"Bearer {secret}"},
            json={"label": "from-test"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "key_id" in data and "token" in data
        assert data["token"].startswith("rrk_")


def test_admin_issue_route_rate_limit(tmp_path) -> None:
    db = str(tmp_path / "keys.db")
    init_access_keys_db_sync(db)
    secret = "rl-secret"
    limiter = AdminRateLimiter(2)
    handler = make_admin_issue_handler(db, secret, limiter)
    app = Starlette(routes=[Route("/admin/access-keys", endpoint=handler, methods=["POST"])])

    headers = {"Authorization": f"Bearer {secret}"}
    with TestClient(app) as client:
        assert client.post("/admin/access-keys", headers=headers, json={}).status_code == 200
        assert client.post("/admin/access-keys", headers=headers, json={}).status_code == 200
        assert client.post("/admin/access-keys", headers=headers, json={}).status_code == 429
