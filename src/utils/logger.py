"""
Structured logging module for RakshaSite.

Provides pre-configured loggers for system modules:
- YOLO: Computer vision detection and filtering
- AGENT: Reasoning, coordination, and orchestration
- RAG: Retrieval-Augmented Generation / knowledge retrieval
- REPORT: Safety report generation and compliance auditing
- API: REST/WebSocket endpoints and external interfaces
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional, Union

# Supported module names
SUPPORTED_MODULES = ["YOLO", "AGENT", "RAG", "REPORT", "API"]

# Standard log format: timestamp | module | level | message
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cache for initialized loggers to prevent duplicate handler creation
_loggers: Dict[str, logging.Logger] = {}


def _get_default_log_dir() -> Path:
    """Determine the default log directory (rakshasite/logs)."""
    # rakshasite/src/utils/logger.py -> parents[2] is rakshasite/
    base_dir = Path(__file__).resolve().parents[2]
    log_dir = base_dir / "logs"
    return log_dir


def setup_logger(
    name: str,
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure and return a structured logger with console and rotating file handlers.

    Args:
        name: Logger name (e.g. 'YOLO', 'AGENT', 'RAG', 'REPORT', 'API').
        log_dir: Directory to store log files. Defaults to rakshasite/logs/.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        console_output: Whether to attach a StreamHandler (stdout).
        file_output: Whether to attach a RotatingFileHandler.
        max_bytes: Maximum size of a log file before rotation.
        backup_count: Number of rotated log backup files to keep.

    Returns:
        Configured logging.Logger instance.
    """
    logger_name = name.strip().upper() if name.strip().upper() in SUPPORTED_MODULES else name.strip()

    if logger_name in _loggers:
        return _loggers[logger_name]

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Avoid duplicate handlers if logger was previously touched
    if not logger.handlers:
        # Console Handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # File Handler
        if file_output:
            target_log_dir = Path(log_dir) if log_dir else _get_default_log_dir()
            try:
                target_log_dir.mkdir(parents=True, exist_ok=True)
                log_file = target_log_dir / f"{logger_name.lower()}.log"
                file_handler = RotatingFileHandler(
                    filename=str(log_file),
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                # If file creation fails (e.g. permission issues), log warning to console
                logger.warning(
                    f"Failed to initialize file logging in {target_log_dir}: {e}"
                )

    _loggers[logger_name] = logger
    return logger


def get_logger(name: str = "YOLO", level: int = logging.INFO) -> logging.Logger:
    """
    Retrieve or create a configured structured logger by module name.

    Args:
        name: Name of the module ('YOLO', 'AGENT', 'RAG', 'REPORT', 'API' or custom).
        level: Logging level (default: logging.INFO).

    Returns:
        Configured logging.Logger instance.
    """
    normalized_name = name.strip().upper() if name.strip().upper() in SUPPORTED_MODULES else name.strip()
    return setup_logger(normalized_name, level=level)


# Pre-instantiated module loggers for direct import
yolo_logger = get_logger("YOLO")
agent_logger = get_logger("AGENT")
rag_logger = get_logger("RAG")
report_logger = get_logger("REPORT")
api_logger = get_logger("API")
