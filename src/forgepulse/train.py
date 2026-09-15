from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .features import TARGETS
from .model import ManufacturingModels

DATA_PATH = Path("CNC_Milling_Machine/Datasets/CNC-Milling Machine_Production data.xlsx")
MODEL_PATH = Path("artifacts/forgepulse.joblib")


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict[str, float]:
    frame = pd.read_excel(data_path)
    train_df, test_df = train_test_split(frame, test_size=0.2, random_state=42)
    models = ManufacturingModels().fit(train_df)
    predictions, _, _ = models.predict(test_df)

    metrics: dict[str, float] = {}
    for target in TARGETS:
        metrics[f"{target}_mae"] = float(mean_absolute_error(test_df[target], predictions[target]))
        metrics[f"{target}_rmse"] = float(mean_squared_error(test_df[target], predictions[target]) ** 0.5)

    models.save(model_path)
    model_path.with_suffix(".json").write_text(
        json.dumps({"model_version": models.version, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ForgePulse manufacturing models")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.output), indent=2))


if __name__ == "__main__":
    main()
