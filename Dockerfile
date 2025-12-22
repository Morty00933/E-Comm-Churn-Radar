FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install all dependencies (including dev for testing/linting)
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Create directories
RUN mkdir -p /app/data /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
