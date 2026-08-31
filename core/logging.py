"""
Structured logging setup shared across the pipeline.

Uses stdlib logging with a consistent format so log lines are greppable
and, later, forwardable to a log aggregator without rework.
"""

import logging
import sys

from core.config import settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
