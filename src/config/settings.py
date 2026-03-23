"""
settings.py — Centralised Pydantic-v2 settings with full validation.
All environment variables are read once at startup; app code imports
`settings` directly.
"""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    model_config = SettingsConfigDict(env_prefix="AWS_", extra="ignore")

    access_key_id: str = Field(..., description="AWS Access Key ID")
    secret_access_key: str = Field(..., description="AWS Secret Access Key")
    region: str = Field("us-east-1")
    s3_resume_bucket: str = Field(..., description="S3 bucket for resumes")
    s3_jd_bucket: str = Field(..., description="S3 bucket for JDs")


class MilvusSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MILVUS_", extra="ignore")

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
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    model_name: str = Field("Qwen/Qwen2.5-0.5B-Instruct")
    device: str = Field("cpu")
    batch_size: int = Field(8)
    max_length: int = Field(512)
    dimension: int = Field(896)


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: LLMProvider = Field(LLMProvider.OPENAI)
    model_name: str = Field("qwen/qwen-2.5-72b-instruct")
    api_key: Optional[str] = Field(None)
    base_url: Optional[str] = Field(None)
    max_tokens: int = Field(2048)
    temperature: float = Field(0.1)


class MCPSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_SERVER_", extra="ignore")

    name: str = Field("resume-ranker-mcp")
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    transport: Transport = Field(Transport.SSE)


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    top_k: int = Field(20)
    final_rank_k: int = Field(10)
    score_threshold: float = Field(0.3)
    chunk_size: int = Field(512)
    chunk_overlap: int = Field(64)


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", extra="ignore")

    host: str = Field("localhost")
    port: int = Field(6379)
    db: int = Field(0)
    ttl_seconds: int = Field(3600)
    enabled: bool = Field(True)


class NodeJSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NODEJS_", extra="ignore")

    backend_url: str = Field("http://localhost:3000")
    api_key: Optional[str] = Field(None)
    webhook_secret: Optional[str] = Field(None)


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
    allowed_origins: List[str] = Field(default=["http://localhost:3000"])

    # Logging
    log_level: str = Field("INFO")
    log_format: LogFormat = Field(LogFormat.JSON)
    log_file: str = Field("logs/mcp_server.log")

    # Performance
    max_workers: int = Field(4)
    request_timeout: int = Field(60)
    max_resume_size_mb: int = Field(10)
    supported_formats: List[str] = Field(default=["pdf", "docx", "doc", "txt"])

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",")]
        return v

    @field_validator("supported_formats", mode="before")
    @classmethod
    def parse_formats(cls, v):
        if isinstance(v, str):
            return [f.strip().lower() for f in v.split(",")]
        return v

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