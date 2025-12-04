from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from typing import Any

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def set_request_id(request_id: str) -> None:
    request_id_var.set(request_id)


def get_request_id() -> str:
    return request_id_var.get()


def set_correlation_id(correlation_id: str) -> None:
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str:
    return correlation_id_var.get()


def add_context(logger: Any, method_name: str, event_dict: dict) -> dict:
    request_id = request_id_var.get()
    correlation_id = correlation_id_var.get()
    
    if request_id:
        event_dict["request_id"] = request_id
    if correlation_id:
        event_dict["correlation_id"] = correlation_id
    
    return event_dict


def configure_logging(
    json_format: bool = False,
    log_level: str = "INFO",
    service_name: str = "churn-radar",
) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_format:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logging.getLogger(logger_name).handlers.clear()
        logging.getLogger(logger_name).propagate = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


configure_logging(
    json_format=os.getenv("LOG_JSON", "0") == "1",
    log_level=os.getenv("LOG_LEVEL", "INFO"),
)
