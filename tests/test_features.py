import pandas as pd
import pytest

from forgepulse.features import build_features


def sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_volume": [10.0, 20.0],
            "number_of_lines_of_code": [100, 150],
            "number_tool_changes": [3, 4],
            "number_of_travels_to_machine_zero_point_in_rapid_traverse": [2, 5],
            "number_axis_rotations": [10, 20],
            "weighted_tool_diameter": [2.0, 2.5],
            "weighted_cutting_length": [15.0, 30.0],
            "weighted_number_of_cutting_edges": [4, 6],
        }
    )


def test_feature_builder_adds_operational_features():
    result = build_features(sample())
    assert "cutting_intensity" in result
    assert "complexity_index" in result
    assert result.shape[0] == 2


def test_negative_values_are_rejected():
    frame = sample()
    frame.loc[0, "raw_volume"] = -1
    with pytest.raises(ValueError):
        build_features(frame)
