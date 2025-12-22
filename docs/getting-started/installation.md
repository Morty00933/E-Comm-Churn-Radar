# Installation

## Requirements

- Python 3.11+
- Docker & Docker Compose (for full stack)
- 4GB+ RAM recommended

## Quick Install

```bash
# Clone repository
git clone https://github.com/yourname/churn-radar.git
cd churn-radar

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Development Install

```bash
# Install with dev dependencies
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

## Docker Setup

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### Services

| Service | Port | URL |
|---------|------|-----|
| API | 8000 | http://localhost:8000 |
| MLflow | 5000 | http://localhost:5000 |
| Airflow | 8080 | http://localhost:8080 |
| Redis | 6379 | - |

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | development | Runtime environment |
| `MLFLOW_TRACKING_URI` | http://localhost:5000 | MLflow server |
| `REDIS_URL` | redis://localhost:6379/0 | Redis connection |
| `API_KEY` | dev-api-key | API authentication |

## Verify Installation

```bash
# Run tests
make test

# Start API
make api

# Check health
curl http://localhost:8000/health
```

## Troubleshooting

### Port conflicts

```bash
# Check what's using a port
lsof -i :8000

# Use different port
uvicorn src.api.app:app --port 8001
```

### Docker issues

```bash
# Reset containers
docker compose down -v
docker compose up -d --build
```

### Missing dependencies

```bash
# Reinstall all
pip install --force-reinstall -r requirements.txt
```
