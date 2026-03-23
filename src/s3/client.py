"""
s3/client.py — Async-capable AWS S3 client.
Provides download, metadata, and presigned-URL helpers.
Uses aioboto3 for non-blocking I/O.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

import aioboto3
from botocore.exceptions import ClientError

from src.config import get_logger, settings

logger = get_logger(__name__)

# ─── Session factory (shared across the process) ─────────────────────────────

_session = aioboto3.Session(
    aws_access_key_id=settings.aws.access_key_id,
    aws_secret_access_key=settings.aws.secret_access_key,
    region_name=settings.aws.region,
)


@asynccontextmanager
async def _s3_client() -> AsyncGenerator:
    async with _session.client("s3") as client:
        yield client


# ─── Public helpers ──────────────────────────────────────────────────────────

class S3Client:
    """Thin async wrapper around S3 operations needed by the MCP server."""

    # ── Download ──────────────────────────────────────────────────────────────

    async def download_bytes(self, bucket: str, key: str) -> bytes:
        """Download an S3 object and return raw bytes."""
        logger.info("s3.download_bytes", bucket=bucket, key=key)
        buf = io.BytesIO()
        async with _s3_client() as s3:
            try:
                await s3.download_fileobj(bucket, key, buf)
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                logger.error("s3.download_failed", bucket=bucket, key=key, error_code=code)
                raise FileNotFoundError(
                    f"S3 object not found: s3://{bucket}/{key} [{code}]"
                ) from exc
        return buf.getvalue()

    async def download_to_path(self, bucket: str, key: str, local_path: Path) -> Path:
        """Download an S3 object to a local file path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        raw = await self.download_bytes(bucket, key)
        local_path.write_bytes(raw)
        logger.info("s3.download_to_path", dest=str(local_path), size_bytes=len(raw))
        return local_path

    # ── Metadata ──────────────────────────────────────────────────────────────

    async def head_object(self, bucket: str, key: str) -> dict:
        """Return object metadata without downloading content."""
        async with _s3_client() as s3:
            try:
                resp = await s3.head_object(Bucket=bucket, Key=key)
                return {
                    "key": key,
                    "bucket": bucket,
                    "size": resp.get("ContentLength", 0),
                    "last_modified": resp.get("LastModified"),
                    "content_type": resp.get("ContentType", "application/octet-stream"),
                    "etag": resp.get("ETag", "").strip('"'),
                    "metadata": resp.get("Metadata", {}),
                }
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                raise FileNotFoundError(
                    f"S3 head_object failed: s3://{bucket}/{key} [{code}]"
                ) from exc

    async def object_exists(self, bucket: str, key: str) -> bool:
        """Return True if the object exists."""
        try:
            await self.head_object(bucket, key)
            return True
        except FileNotFoundError:
            return False

    # ── List ─────────────────────────────────────────────────────────────────

    async def list_objects(
        self,
        bucket: str,
        prefix: str = "",
        max_keys: int = 1000,
    ) -> list[dict]:
        """List objects in a bucket/prefix. Returns list of {key, size, etag}."""
        results: list[dict] = []
        async with _s3_client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=bucket, Prefix=prefix, PaginationConfig={"MaxItems": max_keys}
            ):
                for obj in page.get("Contents", []):
                    results.append(
                        {
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "etag": obj["ETag"].strip('"'),
                            "last_modified": obj["LastModified"],
                        }
                    )
        return results

    # ── Presigned URLs ────────────────────────────────────────────────────────

    async def presigned_url(
        self, bucket: str, key: str, expiry_seconds: int = 3600
    ) -> str:
        async with _s3_client() as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expiry_seconds,
            )
        return url

    # ── Upload ────────────────────────────────────────────────────────────────

    async def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload raw bytes to S3. Returns the S3 URI."""
        if content_type is None:
            content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
        async with _s3_client() as s3:
            await s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        uri = f"s3://{bucket}/{key}"
        logger.info("s3.upload_bytes", uri=uri, size_bytes=len(data))
        return uri

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def infer_file_extension(key: str) -> str:
        return Path(key).suffix.lower().lstrip(".")

    @staticmethod
    def etag_as_id(etag: str) -> str:
        """Derive a deterministic short ID from an S3 ETag."""
        return hashlib.sha256(etag.encode()).hexdigest()[:16]

    async def validate_file(self, bucket: str, key: str) -> dict:
        """Validate that a file is accessible and within size limits."""
        meta = await self.head_object(bucket, key)
        ext = self.infer_file_extension(key)
        max_bytes = settings.max_resume_size_mb * 1024 * 1024

        if ext not in settings.supported_formats:
            raise ValueError(
                f"Unsupported file format '{ext}'. Allowed: {settings.supported_formats}"
            )
        if meta["size"] > max_bytes:
            raise ValueError(
                f"File too large: {meta['size'] / 1024 / 1024:.1f} MB "
                f"(max {settings.max_resume_size_mb} MB)"
            )
        return meta


# ─── Module-level singleton ───────────────────────────────────────────────────

s3_client = S3Client()