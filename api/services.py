from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.agent_common import write_json
from src.agent_plan import run_agent_plan
from src.agent_report import run_agent_report
from src.evaluate_fairness_metrics import evaluate_fairness
from src.run_diagnostics import run_diagnostics
from src.train import run_training

RUNS_ROOT = Path("outputs/runs")
UPLOAD_ROOT = Path("/tmp/bias_hunter_api_uploads")


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def ensure_roots() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def list_runs() -> list[Dict[str, Any]]:
    ensure_roots()
    runs: list[Dict[str, Any]] = []

    for path in sorted(RUNS_ROOT.iterdir(), reverse=True):
        if not path.is_dir() or path.name == "latest":
            continue
        runs.append(
            {
                "run_id": path.name,
                "path": str(path),
                "has_metrics": (path / "metrics.json").exists(),
                "has_fairness_report": (path / "fairness_report.json").exists(),
                "has_agent_report": (path / "agent_report.json").exists(),
            }
        )
    return runs


def get_run_dir(run_id: str) -> Path:
    if run_id == "latest":
        return RUNS_ROOT / "latest"
    return RUNS_ROOT / run_id


def get_run_summary(run_id: str) -> Dict[str, Any]:
    run_dir = get_run_dir(run_id)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")

    return {
        "run_id": run_dir.name,
        "metrics": load_json_if_exists(run_dir / "metrics.json"),
        "fairness_report": load_json_if_exists(run_dir / "fairness_report.json"),
        "diagnostics_run_summary": load_json_if_exists(run_dir / "diagnostics_run_summary.json"),
        "agent_plan": load_json_if_exists(run_dir / "agent_plan.json"),
        "agent_report": load_json_if_exists(run_dir / "agent_report.json"),
    }


def _timestamp_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = UPLOAD_ROOT / stamp
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = UPLOAD_ROOT / f"{stamp}_{suffix}"
    return candidate


def prepare_effective_config(
    config_bytes: bytes,
    config_filename: str,
    dataset_bytes: bytes | None = None,
    dataset_filename: str | None = None,
) -> Path:
    ensure_roots()
    target_dir = _timestamp_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    config_path = target_dir / config_filename
    with open(config_path, "wb") as f:
        f.write(config_bytes)

    config = load_yaml(config_path)

    if dataset_bytes is not None and dataset_filename is not None:
        dataset_path = target_dir / dataset_filename
        with open(dataset_path, "wb") as f:
            f.write(dataset_bytes)
        config.setdefault("dataset", {})
        config["dataset"]["path"] = str(dataset_path)

    effective_config_path = target_dir / "effective_config.yaml"
    write_yaml(effective_config_path, config)
    return effective_config_path


def run_pipeline(effective_config_path: Path) -> Dict[str, Any]:
    config = load_yaml(effective_config_path)

    train_result = run_training(config)
    run_dir = Path(train_result["out_dir"])

    fairness_report = evaluate_fairness(str(run_dir), config)

    plan = run_agent_plan(run_dir=str(run_dir), config_path=str(effective_config_path))
    write_json(str(run_dir / "agent_plan.json"), plan)

    diagnostics_summary = run_diagnostics(
        run_dir=str(run_dir),
        config_path=str(effective_config_path),
    )

    agent_report = run_agent_report(
        run_dir=str(run_dir),
        config_path=str(effective_config_path),
    )
    write_json(str(run_dir / "agent_report.json"), agent_report)

    latest_dir = RUNS_ROOT / "latest"
    if latest_dir.exists() and latest_dir.resolve() != run_dir.resolve():
        shutil.copy2(run_dir / "agent_plan.json", latest_dir / "agent_plan.json")
        shutil.copy2(run_dir / "agent_report.json", latest_dir / "agent_report.json")
        if (run_dir / "diagnostics_run_summary.json").exists():
            shutil.copy2(
                run_dir / "diagnostics_run_summary.json",
                latest_dir / "diagnostics_run_summary.json",
            )

    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "effective_config_path": str(effective_config_path),
        "metrics": train_result.get("results"),
        "fairness_report": fairness_report,
        "diagnostics_run_summary": diagnostics_summary,
        "agent_plan": plan,
        "agent_report": agent_report,
    }
