from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import TARGETS, build_features


def validate(path: Path) -> dict[str, int]:
    frame = pd.read_excel(path)
    features = build_features(frame)
    if (
        features.isna().any().any()
        or not features.applymap(lambda value: pd.notna(value) and pd.api.types.is_number(value))
        .all()
        .all()
    ):
        raise ValueError("Input contains invalid feature values")
    missing_targets = [target for target in TARGETS if target not in frame.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    return {"rows": len(frame), "features": features.shape[1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("CNC_Milling_Machine/Datasets/CNC-Milling Machine_Production data.xlsx"),
    )
    args = parser.parse_args()
    print(validate(args.data))


if __name__ == "__main__":
    main()
