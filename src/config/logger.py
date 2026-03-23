"""
logger.py — Structured JSON logging setup using structlog + stdlib logging.
Import `get_logger` everywhere instead of using `logging` directly.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any

import structlog

from src.config.settings import settings


def _configure_stdlib_logging() -> None:
    """Wire stdlib logging into structlog's output pipeline."""
    log_dir = Path(settings.log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        settings.log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        handlers=handlers,
        format="%(message)s",  # structlog owns formatting
    )

    # Silence noisy third-party libs
    for noisy in ("boto3", "botocore", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _build_processors() -> list[Any]:
    """Build the structlog processor chain."""
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    if settings.log_format.value == "json":
        shared.append(structlog.processors.JSONRenderer())
    else:
        shared.append(structlog.dev.ConsoleRenderer(colors=True))

    return shared


def configure_logging() -> None:
    """Call once at application startup."""
    _configure_stdlib_logging()
    structlog.configure(
        processors=_build_processors(),
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for the given module name."""
    return structlog.get_logger(name)
