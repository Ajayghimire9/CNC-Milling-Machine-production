from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "raw_volume",
    "number_of_lines_of_code",
    "number_tool_changes",
    "number_of_travels_to_machine_zero_point_in_rapid_traverse",
    "number_axis_rotations",
    "weighted_tool_diameter",
    "weighted_cutting_length",
    "weighted_number_of_cutting_edges",
]

TARGETS = ["processing_time", "average_power_consumption"]


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    clean = frame.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    if (clean[FEATURES] < 0).any().any():
        raise ValueError("Manufacturing features must be non-negative")
    return clean


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    clean = validate_frame(frame)
    x = clean[FEATURES].copy()
    x["cutting_intensity"] = x["weighted_cutting_length"] / x["raw_volume"].clip(lower=1e-6)
    x["tool_change_density"] = x["number_tool_changes"] / (x["number_of_lines_of_code"] + 1.0)
    x["axis_rotation_density"] = x["number_axis_rotations"] / (x["raw_volume"] + 1.0)
    x["complexity_index"] = np.log1p(x["number_of_lines_of_code"] + x["number_axis_rotations"])
    return x
