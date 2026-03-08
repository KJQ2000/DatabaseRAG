"""
utils/logging_utils.py
----------------------
Centralised logging configuration for the Agentic Database RAG project.
Logs are written to logs/app.log (rotating) and echoed to stdout.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path


_CONFIGURED = False


def configure_logging(log_level: str | None = None) -> None:
    """Set up root logging with a rotating file handler and a stream handler.

    Safe to call multiple times — subsequent calls are no-ops.

    Parameters
    ----------
    log_level:
        Override log level (e.g. "DEBUG", "INFO"). Defaults to the
        ``LOG_LEVEL`` environment variable, falling back to "INFO".
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    level_str = (log_level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    # Ensure the logs directory exists
    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Rotating file handler — 5 MB per file, keep last 3
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Stream (console) handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring logging is configured first.

    Parameters
    ----------
    name:
        Logger name — typically ``__name__`` from the calling module.

    Returns
    -------
    ``logging.Logger`` instance.
    """
    configure_logging()
    return logging.getLogger(name)
