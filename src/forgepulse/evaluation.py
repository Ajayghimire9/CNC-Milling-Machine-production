from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import TARGETS


def evaluate(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for target in TARGETS:
        metrics[f"{target}_mae"] = float(mean_absolute_error(y_true[target], y_pred[target]))
        metrics[f"{target}_rmse"] = float(mean_squared_error(y_true[target], y_pred[target]) ** 0.5)
        metrics[f"{target}_r2"] = float(r2_score(y_true[target], y_pred[target]))
    return metrics


def write_report(metrics: dict[str, float], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
