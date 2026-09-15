from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MachineRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_volume: float = Field(gt=0)
    number_of_lines_of_code: float = Field(ge=0)
    number_tool_changes: float = Field(ge=0)
    number_of_travels_to_machine_zero_point_in_rapid_traverse: float = Field(ge=0)
    number_axis_rotations: float = Field(ge=0)
    weighted_tool_diameter: float = Field(gt=0)
    weighted_cutting_length: float = Field(ge=0)
    weighted_number_of_cutting_edges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    processing_time: float
    processing_time_std: float
    average_power_consumption: float
    average_power_consumption_std: float
    anomaly_score: float
    anomaly: bool
    risk_level: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    checks: dict[str, Any]
