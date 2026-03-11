"""
Structured Logging Configuration
Provides JSON-formatted logging for production and human-readable logging for development.

Usage:
    from src.monitoring.logging_config import setup_logging

    setup_logging()  # Auto-detects mode from LOG_FORMAT env var
    setup_logging(json_format=True)  # Force JSON
    setup_logging(json_format=False)  # Force human-readable
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON for structured log aggregation.

    Output fields:
        timestamp, level, logger, message, module, function, line,
        + any extra fields added via logger.info("msg", extra={...})
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields (e.g., request_id, species, latency)
        # Skip standard LogRecord attributes
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "relativeCreated",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "levelno",
            "levelname",
            "pathname",
            "filename",
            "module",
            "thread",
            "threadName",
            "process",
            "processName",
            "message",
            "msecs",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """
    Human-readable formatter with color support for development.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        if self.use_color:
            color = self.COLORS.get(level, "")
            level_str = f"{color}{level:<8}{self.RESET}"
        else:
            level_str = f"{level:<8}"

        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        name = record.name
        if len(name) > 25:
            name = "..." + name[-22:]

        message = record.getMessage()

        # Add extra fields inline
        extras = []
        standard_attrs = {
            "name",
            "msg",
            "args",
            "created",
            "relativeCreated",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "levelno",
            "levelname",
            "pathname",
            "filename",
            "module",
            "thread",
            "threadName",
            "process",
            "processName",
            "message",
            "msecs",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs and not key.startswith("_"):
                extras.append(f"{key}={value}")

        extra_str = f" [{', '.join(extras)}]" if extras else ""

        line = f"{timestamp} {level_str} {name:<25} {message}{extra_str}"

        if record.exc_info and record.exc_info[0] is not None:
            line += "\n" + self.formatException(record.exc_info)

        return line


def setup_logging(
    json_format: bool | None = None,
    level: str = None,
    log_file: str | None = None,
):
    """
    Configure logging for the application.

    Args:
        json_format: True=JSON, False=human, None=auto-detect from LOG_FORMAT env
        level: Log level (DEBUG/INFO/WARNING/ERROR). Auto-detects from LOG_LEVEL env.
        log_file: Optional path to write logs to a file as well.
    """
    # Auto-detect format
    if json_format is None:
        env_format = os.getenv("LOG_FORMAT", "human").lower()
        json_format = env_format == "json"

    # Auto-detect level
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    log_level = getattr(logging, level, logging.INFO)

    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = HumanFormatter()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        from pathlib import Path

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: format={'JSON' if json_format else 'human'}, "
        f"level={level}" + (f", file={log_file}" if log_file else "")
    )


# ─── Structured Log Helpers ──────────────────────────────────────


def log_pipeline_event(
    logger: logging.Logger,
    event: str,
    request_id: str = "",
    **kwargs,
):
    """
    Log a structured pipeline event with consistent extra fields.

    Usage:
        log_pipeline_event(logger, "search_complete",
            request_id="abc123", results=15, latency_ms=120)
    """
    logger.info(
        f"[{request_id}] {event}",
        extra={
            "event": event,
            "request_id": request_id,
            **kwargs,
        },
    )
