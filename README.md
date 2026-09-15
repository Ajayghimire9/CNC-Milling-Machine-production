# ForgePulse — CNC Manufacturing Intelligence

ForgePulse is a production-oriented machine-learning platform for CNC manufacturing data.

The original repository started as an exploratory analysis of CNC production records. I rebuilt it around a more realistic engineering problem: **how can a manufacturing team estimate job performance, identify unusual machine workloads, and expose those predictions through a service that can be monitored and tested?**

The result is a small but complete ML system rather than a notebook with a model attached to it.

## What it does

ForgePulse currently provides three capabilities:

- **Production forecasting** — estimates processing time and average power consumption from job-level manufacturing characteristics.
- **Anomaly detection** — flags unusual combinations of manufacturing features using an unsupervised Isolation Forest model.
- **Model serving** — exposes predictions through a typed FastAPI service with health checks and Prometheus-compatible metrics.

The training pipeline also evaluates the model on a held-out test set and stores the trained artifact together with a machine-readable metrics manifest.

## Architecture

```text
                 CNC production records
                          │
                          ▼
                 Data validation layer
                          │
                          ▼
              Feature engineering pipeline
             ┌────────────┴────────────┐
             ▼                         ▼
      Performance model          Anomaly detector
      ExtraTrees regression       Isolation Forest
             │                         │
             └────────────┬────────────┘
                          ▼
                   Model artifact
                          │
                          ▼
                    FastAPI service
                     │           │
                     ▼           ▼
                Predictions   /metrics
                                  │
                                  ▼
                         Prometheus / ops
```

This separation is intentional. Data validation, feature construction, training, inference, and monitoring are independent concerns, which makes the system easier to test and replace.

## Why these models?

The dataset is relatively small and tabular. A large neural network would add complexity without solving a real problem here.

ForgePulse therefore uses **ExtraTrees regression** for the production targets. It handles nonlinear relationships and feature interactions well on structured data while remaining inexpensive to train. An **Isolation Forest** is used separately because anomaly detection is a different problem from supervised prediction: unusual machine/job combinations do not require a labelled failure dataset.

That distinction is important in manufacturing systems, where labelled failure events are often much rarer than normal production records.

## Engineering features

- Typed request and response contracts with Pydantic
- Explicit input validation and finite-value handling
- Reusable feature engineering instead of notebook-only transformations
- Multi-target regression for processing time and power consumption
- Unsupervised anomaly detection
- Reproducible train/test split and fixed model seeds
- Versioned model artifact format
- JSON evaluation manifest
- Population Stability Index utility for feature drift checks
- FastAPI inference endpoint
- Prometheus request metrics
- Dockerized serving
- Ruff + Pytest in GitHub Actions
- `src/` package layout with installable Python project

## Project structure

```text
.
├── CNC_Milling_Machine/
│   └── Datasets/
│       └── CNC-Milling Machine_Production data.xlsx
├── src/
│   └── forgepulse/
│       ├── api.py          # HTTP inference service
│       ├── drift.py        # feature drift utilities
│       ├── features.py     # validation + feature engineering
│       ├── model.py        # predictive + anomaly models
│       ├── schema.py       # API/data contracts
│       └── train.py        # training + evaluation entry point
├── tests/
│   ├── test_drift.py
│   └── test_features.py
├── Dockerfile
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Run it locally

Python 3.11+ is recommended.

```bash
git clone https://github.com/Ajayghimire9/CNC-Milling-Machine-production.git
cd CNC-Milling-Machine-production

python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run quality checks:

```bash
ruff check src tests
pytest -q
```

Train the models:

```bash
python -m forgepulse.train
```

The command creates:

```text
artifacts/
├── forgepulse.joblib
└── forgepulse.json
```

The JSON file contains the model version and held-out evaluation metrics.

## Serve predictions

After training:

```bash
uvicorn forgepulse.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Example prediction:

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "raw_volume": 120.0,
    "number_of_lines_of_code": 850,
    "number_tool_changes": 12,
    "number_of_travels_to_machine_zero_point_in_rapid_traverse": 7,
    "number_axis_rotations": 45,
    "weighted_tool_diameter": 4.2,
    "weighted_cutting_length": 180.0,
    "weighted_number_of_cutting_edges": 8
  }'
```

Metrics are available at:

```text
http://localhost:8000/metrics
```

## Docker

Build and run the API after creating the model artifact locally:

```bash
docker build -t forgepulse .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" forgepulse
```

## Data and modelling notes

The source dataset contains CNC job-level production characteristics, including geometry-related measures, tool changes, axis rotations, cutting dimensions, processing time, and average power consumption.

The feature layer derives additional operational signals such as:

- cutting intensity
- tool-change density
- axis-rotation density
- a simple job-complexity index

The pipeline deliberately keeps these transformations in Python code rather than embedding them in a notebook so that training and serving use the same feature logic.

## What I would add for a real factory deployment

The repository is intentionally honest about its current boundary. The next engineering layer would be:

1. Kafka or MQTT ingestion from machine telemetry.
2. A durable lakehouse layer using object storage and an open table format.
3. MLflow for experiment tracking and model promotion.
4. Online feature and prediction monitoring.
5. Alerting for sustained drift or anomaly-rate changes.
6. Kubernetes deployment with resource limits and autoscaling.
7. CI/CD image publishing and environment promotion.
8. Model comparison against simple production baselines before deployment.

Those components are not presented as implemented until they actually exist in the repository.

## Portfolio positioning

ForgePulse demonstrates **ML engineering for industrial data** rather than only model training. The interesting part is the path from a raw manufacturing record to a validated feature set, two different modelling objectives, a persisted artifact, an API contract, operational metrics, tests, and CI.

That makes the project useful as a portfolio example for **Machine Learning Engineer, MLOps Engineer, Data Scientist, and Industrial AI** roles.

## License

No license has been added yet. Add one before distributing the repository as an open-source project.
