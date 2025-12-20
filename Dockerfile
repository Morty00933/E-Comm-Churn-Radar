# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Builder — устанавливаем зависимости через UV
# gcc не нужен: scikit-learn, LightGBM, XGBoost, SHAP поставляются
# как pre-built manylinux-колёса и не требуют компиляции
# =============================================================================
FROM ghcr.io/astral-sh/uv:0.6-python3.11-bookworm-slim AS builder

WORKDIR /app

COPY requirements.txt ./

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# BuildKit cache mount — колёса кешируются между сборками
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r requirements.txt

# =============================================================================
# Stage 2: Production image
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="Churn Radar Team" \
      version="1.0.0" \
      description="Customer Churn Prediction API"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH"

# BuildKit apt cache — пакеты не скачиваются заново при повторной сборке
# libgomp1 нужен в рантайме для LightGBM / XGBoost
# curl нужен для HEALTHCHECK
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Создаём непривилегированного пользователя
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Копируем venv из builder-стадии
COPY --from=builder /opt/venv /opt/venv

# Копируем исходный код
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup configs/ ./configs/

RUN mkdir -p /app/data /app/models /app/logs && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
