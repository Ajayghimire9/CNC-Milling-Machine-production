from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .evaluation import evaluate, write_report
from .features import TARGETS
from .model import ManufacturingModels


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a ForgePulse artifact")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("CNC_Milling_Machine/Datasets/CNC-Milling Machine_Production data.xlsx"),
    )
    parser.add_argument("--model", type=Path, default=Path("artifacts/forgepulse.joblib"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    frame = pd.read_excel(args.data)
    _, test_df = train_test_split(frame, test_size=0.2, random_state=42)
    model = ManufacturingModels.load(args.model)
    prediction, _, _ = model.predict(test_df)
    metrics = evaluate(test_df[TARGETS], prediction)
    write_report(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()
