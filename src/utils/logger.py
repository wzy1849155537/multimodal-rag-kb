"""Structured logging setup using loguru."""

import sys
from pathlib import Path
from loguru import logger as _logger


def setup_logger(
    log_dir: str = "logs",
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """Configure loguru logger with console + file sinks."""
    _logger.remove()  # Remove default handler

    _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    _logger.add(
        log_path / "rag_kb_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    )


def get_logger(name: str = __name__):
    """Get a logger instance with a bound module name."""
    return _logger.bind(name=name)
