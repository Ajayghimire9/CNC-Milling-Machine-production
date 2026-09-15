# ForgePulse

Manufacturing forecasts and anomaly analysis.

ForgePulse turns job-level CNC records into processing-time and power-consumption estimates. An independent anomaly detector highlights records that differ from the training population.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -e ".[dev]"
python -m forgepulse.train
uvicorn forgepulse.api:app --host 127.0.0.1 --port 8000
```

## Design decisions

ExtraTrees predicts two targets; Isolation Forest adds an unsupervised anomaly score.

Tree dispersion is computed with the same scaled features used during training. Invalid rows are rejected instead of silently disappearing from predictions.

Online and batch paths share the fitted model. Readiness loads the artifact instead of reporting success solely because a path exists.

Kafka adapters, Airflow stages, dbt models and Kubernetes manifests provide integration points. Local Compose includes MLflow and Prometheus.

## Technology

Python, scikit-learn, MLflow, FastAPI, Kafka, Airflow, dbt, Parquet, Docker, Prometheus.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

Tree dispersion is not a calibrated confidence interval, and anomaly scores are not verified fault labels. The infrastructure directory is a boundary for future cloud wiring, not evidence of a deployed cloud platform. The model must be tested on plant-specific data before operational use.
