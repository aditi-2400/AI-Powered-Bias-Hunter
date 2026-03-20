from __future__ import annotations
import re
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_RUN_DIR = "outputs/runs/latest"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_CONFIG_PATH = "config/audit_config.yaml"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def call_ollama(model: str, prompt: str, timeout_s: int = 180) -> str:
    payload = {
        "model": model,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return ONE valid JSON object only. "
                    "Do not use markdown fences, backticks, or YAML."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")

    parsed = json.loads(raw)
    return parsed["message"]["content"]


def parse_model_json(raw: str) -> Dict[str, Any]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError("Model response was not valid JSON.") from e

    if not isinstance(obj, dict):
        raise ValueError("Model response JSON must be an object.")

    return obj


def load_evidence(run_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {}
    missing: List[str] = []

    required = ["fairness_report.json"]
    optional = [
        "metrics.json",
        "group_sizes.json",
        "diagnosis.json",
        "proxy_report.json",
        "distribution_report.json",
        "slice_report.json",
        "threshold_sensitivity.json",
        "agent_plan.json",
    ]

    for fname in required:
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)
        else:
            evidence[fname.replace(".json", "")] = load_json(path)

    for fname in optional:
        path = os.path.join(run_dir, fname)
        if os.path.exists(path):
            evidence[fname.replace(".json", "")] = load_json(path)

    diagnostics_dir = os.path.join(run_dir, "diagnostics")
    if os.path.isdir(diagnostics_dir):
        diagnostics = {}
        for fname in sorted(os.listdir(diagnostics_dir)):
            if fname.endswith(".json"):
                fpath = os.path.join(diagnostics_dir, fname)
                diagnostics[fname.replace(".json", "")] = load_json(fpath)
        if diagnostics:
            evidence["diagnostics"] = diagnostics

    evidence["audit_config"] = {
        "dataset": config.get("dataset", {}),
        "label_col": config.get("label_col"),
        "positive_label": config.get("positive_label"),
        "sensitive_cols": config.get("sensitive_cols", []),
        "min_group_size": config.get("min_group_size"),
        "fairness_threshold": config.get("fairness_threshold"),
        "model": config.get("model", {}),
    }

    evidence["_meta"] = {
        "run_dir": run_dir,
        "missing_required": missing,
        "loaded": sorted([k for k in evidence.keys() if not k.startswith("_")]),
    }
    return evidence