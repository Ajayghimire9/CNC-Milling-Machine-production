from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .model import ManufacturingModels


def run(input_path: Path, model_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(input_path)
    model = ManufacturingModels.load(model_path)
    prediction, uncertainty, score, flag = model.predict_with_uncertainty(frame)
    result = frame.copy()
    result["processing_time_pred"] = prediction["processing_time"]
    result["processing_time_std"] = uncertainty["processing_time_std"]
    result["power_pred"] = prediction["average_power_consumption"]
    result["power_std"] = uncertainty["average_power_consumption_std"]
    result["anomaly_score"] = score
    result["anomaly"] = flag
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline ForgePulse batch inference")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=Path("artifacts/forgepulse.joblib"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.parquet"))
    args = parser.parse_args()
    run(args.input, args.model, args.output)


if __name__ == "__main__":
    main()
