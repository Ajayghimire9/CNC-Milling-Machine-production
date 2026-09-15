# ForgePulse — Industrial ML & MLOps Platform for CNC Manufacturing

ForgePulse is a production-oriented industrial ML system for CNC manufacturing intelligence. It turns job-level manufacturing records into **performance forecasts, uncertainty estimates, anomaly signals, monitored API predictions, and reproducible ML lifecycle artifacts**.

The project intentionally goes beyond a notebook: data validation, feature engineering, model training, experiment tracking, artifact integrity, streaming integration, batch scoring, observability, orchestration, containerization, Kubernetes deployment, and warehouse modelling are separated into explicit engineering layers.

## Architecture

```text
 CNC machine / ERP / MES telemetry
              │
              ▼
       Kafka telemetry bus
              │
              ▼
    Validation + feature layer
              │
       ┌──────┴───────┐
       ▼              ▼
 ExtraTrees         Isolation
 regression         Forest
       │              │
       └──────┬───────┘
              ▼
   Evaluation + uncertainty
              │
              ▼
      MLflow experiment log
              │
              ▼
   Versioned model + SHA-256
              │
       ┌──────┴───────┐
       ▼              ▼
 FastAPI online     Batch Parquet
 inference          inference
       │              │
       └──────┬───────┘
              ▼
 Prometheus → alerts → operations
              │
              ▼
     Kubernetes / HPA

 Airflow → validation → train → evaluate
 dbt → analytical warehouse models
 DVC → reproducible pipeline stages
 Terraform → infrastructure boundary
 GitHub Actions → lint → tests → image build
```

## Implemented engineering capabilities

### ML engineering
- Multi-target ExtraTrees regression for processing time and power consumption.
- Isolation Forest for unsupervised workload anomaly detection.
- Ensemble-based prediction uncertainty using tree dispersion.
- Deterministic train/test split and fixed model seeds.
- Reusable evaluation module with MAE, RMSE, and R².
- Model artifact manifest with SHA-256 integrity metadata.

### MLOps
- MLflow experiment tracking for parameters, metrics, artifacts, and model packaging.
- DVC pipeline definition for reproducible training dependencies and outputs.
- Explicit model lifecycle vocabulary: candidate → validated → production.
- Externalized configuration in `params.yaml`.
- Batch inference to Parquet for offline scoring.

### Data engineering
- Kafka producer/consumer adapter for CNC telemetry events.
- dbt staging and fact models with schema tests.
- Airflow DAG for validation → training → evaluation orchestration.
- Parquet as an analytics-friendly batch format.

### Production serving
- FastAPI typed request/response contracts.
- `/health` and `/ready` probes for container orchestration.
- Prometheus request counters and latency histograms.
- Risk-level routing based on anomaly score.
- Docker multi-stage build and non-root runtime user.
- Kubernetes deployment with resource requests/limits and HPA.

### Platform / DevOps
- Docker Compose development stack with Kafka, MLflow, Prometheus, and API.
- Terraform infrastructure boundary with externalized environment configuration.
- GitHub Actions quality gate: Ruff, Pytest, and container build.
- No credentials or cloud-specific secrets are committed to the repository.

## Project structure

```text
src/forgepulse/
├── api.py           # online inference + health/metrics
├── batch.py         # offline CSV → Parquet scoring
├── drift.py         # population stability utilities
├── evaluate.py      # evaluation CLI
├── evaluation.py    # reusable metrics/reporting
├── features.py      # validation + feature engineering
├── model.py         # regression + anomaly models
├── registry.py      # artifact integrity/lifecycle metadata
├── schema.py        # Pydantic contracts
├── streaming.py     # Kafka telemetry adapter
├── train.py         # training + MLflow tracking
└── validate.py      # data validation CLI

airflow/dags/        # orchestration
models/              # dbt warehouse models
infra/terraform/     # IaC boundary
k8s/                 # Kubernetes deployment + HPA
monitoring/          # Prometheus alert rules
deploy/              # Prometheus configuration
tests/               # automated tests
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

ruff check src tests
pytest -q
```

Train and track an experiment locally:

```bash
python -m forgepulse.train
mlflow ui --backend-store-uri ./mlruns
```

Serve the trained artifact:

```bash
uvicorn forgepulse.api:app --host 0.0.0.0 --port 8000
```

Operational endpoints:

```text
GET  /health
GET  /ready
GET  /metrics
POST /v1/predict
```

## Reproducible pipeline

```bash
python -m forgepulse.validate
python -m forgepulse.train
python -m forgepulse.evaluate
```

Or use the DVC stage definition:

```bash
dvc repro
```

For scheduled orchestration, the Airflow DAG is at `airflow/dags/forgepulse_pipeline.py`.

## Local platform stack

```bash
docker compose up --build
```

Services include the inference API, Kafka, MLflow, and Prometheus. Production Kubernetes manifests live under `k8s/`.

## Important portfolio note

This repository uses a real CNC production dataset and demonstrates industrial ML engineering patterns. It does **not** claim that the anomaly model is a safety-certified machine protection system. A real factory deployment would require validated telemetry contracts, plant-specific failure labels, historical backtesting, human approval policies, security controls, SLOs, and integration testing with the target MES/SCADA environment.

## Why this project matters for a Data/ML Engineer role

ForgePulse demonstrates the complete path from **industrial data → streaming/batch ingestion → validated features → model training → experiment tracking → model integrity → online/batch inference → monitoring → orchestration → container deployment → Kubernetes**.

It is designed to be interviewable: each technology has a concrete responsibility rather than being included only as a keyword.
