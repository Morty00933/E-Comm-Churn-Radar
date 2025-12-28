# Architecture Overview

## System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        A[Web Client]
        B[Mobile App]
        C[Internal Tools]
    end
    
    subgraph "API Layer"
        D[FastAPI]
        E[Rate Limiter]
        F[Auth Middleware]
    end
    
    subgraph "ML Layer"
        G[Model Service]
        H[SHAP Explainer]
        I[Feature Store]
    end
    
    subgraph "Data Layer"
        J[Redis Cache]
        K[MLflow Registry]
        L[File Storage]
    end
    
    subgraph "Orchestration"
        M[Airflow]
        N[Training Pipeline]
        O[HPO Pipeline]
    end
    
    A --> D
    B --> D
    C --> D
    D --> E --> F --> G
    G --> H
    G --> I
    I --> J
    G --> K
    M --> N --> K
    M --> O --> K
```

## Components

### API Layer

**FastAPI Application** (`src/api/`)

- REST endpoints for predictions and explanations
- Pydantic schemas for validation
- Rate limiting (Redis-backed)
- API key authentication
- Prometheus metrics

### ML Layer

**Model Service** (`src/models/`)

- Model loading from MLflow or local files
- Prediction with confidence scores
- SHAP-based explanations
- Batch processing

**Feature Engineering** (`src/features/`)

- Feature preparation and transformation
- Feature store with Redis caching
- Data validation with Pandera

### Data Layer

**Data Processing** (`src/data/`)

- Raw data loading (CSV)
- Event aggregation
- Train/test splitting
- Churn label generation

### Orchestration

**Airflow DAGs** (`dags/`)

- `etl_pipeline` - Daily data processing
- `training_pipeline` - Weekly model training
- `hpo_pipeline` - Hyperparameter optimization
- `batch_inference` - Daily batch predictions

## Data Flow

### Training Flow

```mermaid
sequenceDiagram
    participant Airflow
    participant DataPipeline
    participant FeatureEng
    participant Trainer
    participant MLflow
    
    Airflow->>DataPipeline: Trigger ETL
    DataPipeline->>DataPipeline: Load raw data
    DataPipeline->>DataPipeline: Clean & validate
    DataPipeline->>FeatureEng: Process features
    FeatureEng->>Trainer: Train model
    Trainer->>MLflow: Log metrics & model
    MLflow->>MLflow: Register model
```

### Inference Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant FeatureStore
    participant Model
    participant Explainer
    
    Client->>API: POST /predict
    API->>API: Validate & auth
    API->>FeatureStore: Get/compute features
    FeatureStore->>Model: Predict
    Model->>API: Probabilities
    API->>Explainer: Get SHAP values
    Explainer->>API: Explanations
    API->>Client: Response
```

## Deployment Architecture

### Development

```
┌─────────────────────────────────────────┐
│             Docker Compose              │
├─────────┬─────────┬─────────┬──────────┤
│   API   │ MLflow  │ Airflow │  Redis   │
│  :8000  │  :5000  │  :8080  │  :6379   │
└─────────┴─────────┴─────────┴──────────┘
```

### Production

```
┌─────────────────────────────────────────────────────┐
│                   Load Balancer                     │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────┐
│                     │                               │
│  ┌──────────┐  ┌────┴─────┐  ┌──────────┐          │
│  │ API Pod  │  │ API Pod  │  │ API Pod  │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │             │             │                 │
│       └─────────────┼─────────────┘                 │
│                     │                               │
│  ┌──────────────────┼──────────────────────────┐   │
│  │                  │      Shared Services      │   │
│  │   ┌──────────┐   │   ┌──────────┐           │   │
│  │   │  Redis   │   │   │  MLflow  │           │   │
│  │   │ Cluster  │   │   │  Server  │           │   │
│  │   └──────────┘   │   └──────────┘           │   │
│  │                  │                           │   │
│  └──────────────────┴───────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic, uvicorn |
| ML | scikit-learn, LightGBM, XGBoost, SHAP |
| MLOps | MLflow, Airflow, DVC |
| Cache | Redis |
| Monitoring | Prometheus, Grafana |
| Logging | structlog |
| Container | Docker |
