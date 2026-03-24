"""
settings.py — Centralised Pydantic-v2 settings with full validation.
All environment variables are read once at startup; app code imports
`settings` directly.
"""
from __future__ import annotations

import json
from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _nested_settings_config(env_prefix: str) -> SettingsConfigDict:
    """Nested BaseSettings do not inherit the parent's env_file; load the same .env here."""
    return SettingsConfigDict(
        env_prefix=env_prefix,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ─── Enums ───────────────────────────────────────────────────────────────────

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class Transport(str, Enum):
    SSE = "sse"
    STDIO = "stdio"


class LogFormat(str, Enum):
    JSON = "json"
    TEXT = "text"


# ─── Sub-settings groups ──────────────────────────────────────────────────────

class AWSSettings(BaseSettings):
    model_config = _nested_settings_config("AWS_")

    access_key_id: Optional[str] = Field(
        default=None,
        description="Optional static key; if omitted, uses ~/.aws/credentials, IAM role, etc.",
    )
    secret_access_key: Optional[str] = Field(
        default=None,
        description="Optional static secret; must be set together with access_key_id.",
    )
    profile: Optional[str] = Field(
        default=None,
        description="Named profile from ~/.aws/config (env: AWS_PROFILE).",
    )
    region: str = Field("ap-south-1")
    s3_resume_bucket: str = Field(..., description="S3 bucket for resumes")
    s3_jd_bucket: str = Field(..., description="S3 bucket for JDs")
    resume_key_prefix: str = Field(
        "development/resumes/",
        description="Allowed S3 key prefix for resume ingestion (guard rail).",
    )
    jd_key_prefix: str = Field(
        "development/jd/",
        description="Allowed S3 key prefix for JD ingestion/ranking (guard rail).",
    )


class MilvusSettings(BaseSettings):
    model_config = _nested_settings_config("MILVUS_")

    host: str = Field("localhost")
    port: int = Field(19530)
    user: str = Field("root")
    password: str = Field("Milvus")
    db_name: str = Field("resume_ranker")
    collection_name: str = Field("resume_embeddings")
    jd_collection_name: str = Field("jd_embeddings")
    index_type: str = Field("IVF_FLAT")
    metric_type: str = Field("COSINE")
    nlist: int = Field(128)


class EmbeddingSettings(BaseSettings):
    model_config = _nested_settings_config("EMBEDDING_")

    model_name: str = Field("Qwen/Qwen2.5-0.5B-Instruct")
    device: str = Field(
        "mps",
        description="cpu | mps (Apple Silicon) | cuda | auto (cuda → mps → cpu)",
    )
    batch_size: int = Field(8)
    max_length: int = Field(512)
    dimension: int = Field(896)


class LLMSettings(BaseSettings):
    model_config = _nested_settings_config("LLM_")

    provider: LLMProvider = Field(LLMProvider.OPENAI)
    model_name: str = Field("qwen/qwen-2.5-72b-instruct")
    api_key: Optional[str] = Field(None)
    base_url: Optional[str] = Field(None)
    max_tokens: int = Field(2048)
    temperature: float = Field(0.0)


class MCPSettings(BaseSettings):
    model_config = _nested_settings_config("MCP_SERVER_")

    name: str = Field("resume-ranker-mcp")
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    transport: Transport = Field(Transport.SSE)


class RAGSettings(BaseSettings):
    model_config = _nested_settings_config("RAG_")

    top_k: int = Field(20)
    final_rank_k: int = Field(10)
    score_threshold: float = Field(0.3)
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(64)


class RedisSettings(BaseSettings):
    model_config = _nested_settings_config("REDIS_")

    host: str = Field("localhost")
    port: int = Field(6379)
    db: int = Field(0)
    ttl_seconds: int = Field(3600)
    enabled: bool = Field(True)


class NodeJSSettings(BaseSettings):
    model_config = _nested_settings_config("NODEJS_")

    backend_url: str = Field("http://localhost:3000")
    api_key: Optional[str] = Field(None)
    webhook_secret: Optional[str] = Field(None)


def _csv_or_json_list_str(v: object, *, empty_fallback: str) -> str:
    """Accept comma-separated text or a JSON array string; normalise to comma-separated."""
    if v is None:
        return empty_fallback
    if isinstance(v, list):
        return ",".join(str(x).strip() for x in v if str(x).strip())
    s = str(v).strip()
    if not s:
        return empty_fallback
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return ",".join(str(x).strip() for x in parsed if str(x).strip())
        except json.JSONDecodeError:
            pass
    return s


# ─── Root Settings ────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Sub-groups (composed inline)
    aws: AWSSettings = Field(default_factory=AWSSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    nodejs: NodeJSSettings = Field(default_factory=NodeJSSettings)

    # Security
    api_secret_key: str = Field("change_me_in_production_min_32_chars")
    allowed_origins_csv: str = Field(
        default="http://localhost:3000",
        validation_alias="ALLOWED_ORIGINS",
        description="Comma-separated origins, or a JSON array string.",
    )

    # Logging
    log_level: str = Field("INFO")
    log_format: LogFormat = Field(LogFormat.JSON)
    log_file: str = Field("logs/mcp_server.log")

    # Performance
    max_workers: int = Field(4)
    request_timeout: int = Field(60)
    max_resume_size_mb: int = Field(10)
    supported_formats_csv: str = Field(
        default="pdf,docx,doc,txt",
        validation_alias="SUPPORTED_FORMATS",
        description="Comma-separated extensions, or a JSON array string.",
    )

    @field_validator("allowed_origins_csv", mode="before")
    @classmethod
    def _normalize_allowed_origins_csv(cls, v: object) -> str:
        return _csv_or_json_list_str(v, empty_fallback="http://localhost:3000")

    @field_validator("supported_formats_csv", mode="before")
    @classmethod
    def _normalize_supported_formats_csv(cls, v: object) -> str:
        return _csv_or_json_list_str(v, empty_fallback="pdf,docx,doc,txt")

    @computed_field
    @property
    def allowed_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins_csv.split(",") if o.strip()]

    @computed_field
    @property
    def supported_formats(self) -> list[str]:
        return [f.strip().lower() for f in self.supported_formats_csv.split(",") if f.strip()]

    @model_validator(mode="after")
    def validate_secret_key(self) -> "Settings":
        if self.api_secret_key == "change_me_in_production_min_32_chars":
            import warnings
            warnings.warn("API_SECRET_KEY is using the default value. Set it in .env!", stacklevel=2)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Convenience singleton
settings = get_settings()
