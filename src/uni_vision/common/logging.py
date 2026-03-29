"""Structured logging setup using structlog — spec §12.2.

All log entries are JSON-formatted with mandatory fields:
  timestamp, level, event, camera_id (where applicable).

Usage::

    from uni_vision.common.logging import get_logger

    log = get_logger()
    log.info("ocr_complete", camera_id="cam_01", plate_text="MH12AB1234")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import structlog


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    *,
    log_file: Optional[str] = None,
) -> None:
    """Initialise the structlog + stdlib logging pipeline.

    Call this **once** at application startup, before any logger is used.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: ``"json"`` for production JSON lines, ``"console"`` for
             human-readable coloured output (development).
        log_file: Optional file path.  When set, a ``FileHandler`` is
                  attached to the stdlib root logger in addition to stderr.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Choose renderer
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    # Stderr handler (always present)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stderr_handler)
    root.setLevel(log_level)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(**initial_binds: object) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger with optional initial context.

    Args:
        **initial_binds: Key-value pairs bound to every log entry emitted
            by the returned logger (e.g., ``camera_id="cam_01"``).
    """
    return structlog.get_logger(**initial_binds)
