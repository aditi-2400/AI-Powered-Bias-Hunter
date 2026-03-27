from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")


class RunListItem(BaseModel):
    run_id: str = Field(..., example="20260327_154500")
    path: str = Field(
        ...,
        example="outputs/runs/20260327_154500",
    )
    has_metrics: bool = Field(default=False, example=True)
    has_fairness_report: bool = Field(default=False, example=True)
    has_agent_report: bool = Field(default=False, example=True)


class RunListResponse(BaseModel):
    runs: List[RunListItem]


class RunSummaryResponse(BaseModel):
    run_id: str = Field(..., example="20260327_154500")
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"accuracy": 0.724, "n_train": 750, "n_test": 250},
    )
    fairness_report: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"sex": {"difference": {"selection_rate": 0.1036}}},
    )
    diagnostics_run_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"executed": [{"tool": "run_slice_scan", "args": {}}], "errors": []},
    )
    agent_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"requested_diagnostics": [{"tool": "run_slice_scan", "args": {}}]},
    )
    agent_report: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"summary": "Fairness Audit Report", "recommended_tests": []},
    )


class CreateRunResponse(BaseModel):
    run_id: str = Field(..., example="20260327_154500")
    run_dir: str = Field(..., example="outputs/runs/20260327_154500")
    effective_config_path: str = Field(
        ...,
        example="/tmp/bias_hunter_api_uploads/20260327_154500/effective_config.yaml",
    )
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"accuracy": 0.724, "n_train": 750, "n_test": 250},
    )
    fairness_report: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"sex": {"difference": {"selection_rate": 0.1036}}},
    )
    diagnostics_run_summary: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"executed": [{"tool": "run_slice_scan", "args": {}}], "errors": []},
    )
    agent_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"requested_diagnostics": [{"tool": "run_slice_scan", "args": {}}]},
    )
    agent_report: Optional[Dict[str, Any]] = Field(
        default=None,
        example={"summary": "Fairness Audit Report", "recommended_tests": []},
    )


class ErrorResponse(BaseModel):
    detail: str = Field(
        ...,
        description="Human-readable error message.",
        example="Run not found: 20260327_154500",
    )
