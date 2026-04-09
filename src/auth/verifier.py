"""FastMCP TokenVerifier backed by SQLite + passlib."""
from __future__ import annotations

from fastmcp.server.auth import TokenVerifier
from fastmcp.server.auth.auth import AccessToken

from src.auth.access_keys import MCP_INVOKE_SCOPE, verify_access_key_async
from src.config import get_logger

logger = get_logger(__name__)

_DEFAULT_SCOPES = [MCP_INVOKE_SCOPE]


class SQLiteAccessKeyVerifier(TokenVerifier):
    def __init__(self, db_path: str, required_scopes: list[str] | None = None) -> None:
        rs = _DEFAULT_SCOPES if required_scopes is None else required_scopes
        super().__init__(base_url=None, required_scopes=rs)
        self._db_path = db_path

    async def verify_token(self, token: str) -> AccessToken | None:
        if not token or not token.strip():
            return None
        ok, key_id = await verify_access_key_async(self._db_path, token)
        if not ok or not key_id:
            logger.debug("access_key.verify_failed")
            return None
        logger.debug("access_key.verify_ok", key_id=key_id)
        return AccessToken(
            token=token,
            client_id=key_id,
            scopes=list(self.required_scopes),
            expires_at=None,
            claims={},
        )
