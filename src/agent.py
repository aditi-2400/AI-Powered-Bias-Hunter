from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_RUN_DIR = "outputs/runs/latest"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_CONFIG_PATH = "config/audit_config.yaml"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

REQUIRED_INPUTS = ["fairness_report.json"]

REQUIRED_OUTPUT_KEYS = [
    "summary",
    "detected_issues",
    "likely_causes",
    "recommended_tests",
    "mitigations",
    "limits",
    "narrative_markdown",
]


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


def load_evidence(run_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {}
    missing: List[str] = []

    for fname in REQUIRED_INPUTS:
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            missing.append(fname)
        else:
            evidence[fname.replace(".json", "")] = load_json(path)

    for opt in [
        "metrics.json",
        "diagnosis.json",
        "group_sizes.json",
        "proxy_report.json",
        "distribution_report.json",
        "slice_report.json",
    ]:
        path = os.path.join(run_dir, opt)
        if os.path.exists(path):
            evidence[opt.replace(".json", "")] = load_json(path)

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


def build_prompt(evidence: Dict[str, Any]) -> str:
    cfg = evidence.get("audit_config", {})
    threshold = cfg.get("fairness_threshold", 0.05)
    mild = 0.5 * threshold
    moderate = 1.0 * threshold
    severe = 2.0 * threshold

    evidence_str = json.dumps(evidence, indent=2, ensure_ascii=False)

    return f"""
You are an AI fairness auditor. You will receive JSON evidence produced by a deterministic pipeline.

STRICT RULES:
- Use ONLY the provided JSON evidence. Do NOT invent numbers.
- Do NOT compute new metrics. Only interpret what exists.
- If evidence is missing for a claim, write "insufficient evidence" in limits.
- Output MUST be valid JSON (one top-level object) and NOTHING else.
- narrative_markdown MUST be a valid JSON string (use \\n for new lines).
- Do NOT use triple quotes \"\"\".
- Do NOT use ``` fences.
- narrative_markdown MUST be a JSON string with escaped newlines (\\n).
- Do NOT use YAML block scalars like "|" or ">".
- Do NOT output any YAML.
- Every string in detected_issues[].evidence MUST cite exact JSON paths like:
  "fairness_report.<attribute>.difference.selection_rate = 0.1108"
  "fairness_report.<attribute>.by_group.true_positive_rate.<group> = 0.6364"
- metric must be exactly one of:
    selection_rate
    true_positive_rate
    false_positive_rate
- Severity MUST be based on fairness_report.<attribute>.difference.<metric>, not by_group values.
- If you cite by_group values, also cite the corresponding difference path for the same issue.
- You must compute severity numerically using the thresholds provided.
- Do not estimate severity heuristically.
- If group_sizes exists and any group count is below audit_config.min_group_size, include a limits entry noting that estimates for that group may be unstable due to small sample size.

ISSUE TYPE MAPPING:
- selection_rate -> demographic_disparity
- true_positive_rate -> unequal_opportunity
- false_positive_rate -> unequal_harm

PROJECT THRESHOLDS (from audit_config):
- fairness_threshold = {threshold}
- severity bands using "difference":
  - mild if difference >= {mild:.6f} and < {moderate:.6f}
  - moderate if difference >= {moderate:.6f} and < {severe:.6f}
  - severe if difference >= {severe:.6f}
  - if missing fields -> unknown

RECOMMENDED TESTS (choose based on flagged disparities):
- If selection_rate disparity exists: recommend ["run_feature_distribution_comparison", "run_proxy_detection"]
- If TPR/FPR disparity exists: recommend ["run_threshold_sensitivity", "run_slice_scan"]
- If group_sizes.json exists and any group < min_group_size: recommend ["check_group_sample_sizes"] and mention instability in limits.

OUTPUT JSON SCHEMA:
{{
  "summary": string,
  "detected_issues": [
    {{
      "attribute": string,
      "metric": string,
      "issue_type": string,
      "severity": "mild"|"moderate"|"severe"|"unknown",
      "evidence": [string, ...]
    }}
  ],
  "likely_causes": [string, ...],
  "recommended_tests": [string, ...],
  "mitigations": [string, ...],
  "limits": [string, ...],
  "narrative_markdown": string
}}

Narrative markdown must use headings:
## Fairness Audit Narrative
### What we found
### Evidence (numbers)
### Likely causes (hypotheses)
### Recommended next tests
### Mitigations to consider
### Limits / Unknowns

EVIDENCE JSON:
{evidence_str}
""".strip()


def call_ollama(model: str, prompt: str, timeout_s: int = 120) -> str:
    payload = {
        "model": model,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return ONE valid JSON object only. "
                    "narrative_markdown MUST be a JSON string with \\n. "
                    "Do NOT use backticks."
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


def extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def validate_agent_output(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    for k in REQUIRED_OUTPUT_KEYS:
        if k not in obj:
            errors.append(f"Missing key {k}")

    if "summary" in obj and not isinstance(obj["summary"], str):
        errors.append("summary must be a string")

    if "detected_issues" in obj and not isinstance(obj["detected_issues"], list):
        errors.append("detected_issues must be a list")

    if "narrative_markdown" in obj and not isinstance(obj["narrative_markdown"], str):
        errors.append("narrative_markdown must be a string")

    valid_metrics = {"selection_rate", "true_positive_rate", "false_positive_rate"}
    valid_severities = {"mild", "moderate", "severe", "unknown"}

    if "detected_issues" in obj and isinstance(obj["detected_issues"], list):
        for i, issue in enumerate(obj["detected_issues"]):
            if not isinstance(issue, dict):
                errors.append(f"detected_issues[{i}] must be an object")
                continue

            metric = issue.get("metric")
            severity = issue.get("severity")
            evidence = issue.get("evidence")

            if metric not in valid_metrics:
                errors.append(
                    f"detected_issues[{i}].metric must be one of {sorted(valid_metrics)}"
                )

            if severity not in valid_severities:
                errors.append(
                    f"detected_issues[{i}].severity must be one of {sorted(valid_severities)}"
                )

            if not isinstance(evidence, list):
                errors.append(f"detected_issues[{i}].evidence must be a list")
            else:
                if not all(isinstance(x, str) for x in evidence):
                    errors.append(f"detected_issues[{i}].evidence must contain only strings")

    for key in ["likely_causes", "recommended_tests", "mitigations", "limits"]:
        if key in obj and not isinstance(obj[key], list):
            errors.append(f"{key} must be a list")
        elif key in obj and isinstance(obj[key], list):
            if not all(isinstance(x, str) for x in obj[key]):
                errors.append(f"{key} must contain only strings")

    return (len(errors) == 0), errors


def run_agent(
    run_dir: str = DEFAULT_RUN_DIR,
    model: str = DEFAULT_MODEL,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        return {
            "summary": "Missing audit_config.yaml.",
            "detected_issues": [],
            "likely_causes": [],
            "recommended_tests": [],
            "mitigations": [],
            "limits": [f"Config not found: {config_path}"],
            "narrative_markdown": "## Fairness Audit Narrative\n\n### Limits / Unknowns\n- Missing audit_config.yaml\n",
        }

    config = load_yaml(config_path)
    evidence = load_evidence(run_dir, config)

    if evidence["_meta"]["missing_required"]:
        return {
            "summary": "Insufficient evidence: required input files missing.",
            "detected_issues": [],
            "likely_causes": [],
            "recommended_tests": [],
            "mitigations": [],
            "limits": [f"Missing required files: {', '.join(evidence['_meta']['missing_required'])}"],
            "narrative_markdown": "## Fairness Audit Narrative\n\n### Limits / Unknowns\n- Missing required evidence files.\n",
        }

    prompt = build_prompt(evidence)
    write_text(os.path.join(run_dir, "agent_prompt.txt"), prompt)

    raw = call_ollama(model, prompt)

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_json_object(raw)
        if extracted is None:
            write_text(os.path.join(run_dir, "agent_report_raw.txt"), raw)
            return {
                "summary": "Agent output was not valid JSON.",
                "detected_issues": [],
                "likely_causes": [],
                "recommended_tests": [],
                "mitigations": [],
                "limits": ["Model response did not contain a JSON object."],
                "narrative_markdown": "## Fairness Audit Narrative\n\n### Limits / Unknowns\n- Agent response was not JSON.\n",
            }
        try:
            obj = json.loads(extracted)
        except json.JSONDecodeError:
            write_text(os.path.join(run_dir, "agent_report_raw.txt"), raw)
            return {
                "summary": "Agent output could not be parsed as JSON.",
                "detected_issues": [],
                "likely_causes": [],
                "recommended_tests": [],
                "mitigations": [],
                "limits": ["Failed to parse extracted JSON block."],
                "narrative_markdown": "## Fairness Audit Narrative\n\n### Limits / Unknowns\n- Could not parse agent JSON.\n",
            }

    ok, errors = validate_agent_output(obj)
    if not ok:
        write_text(os.path.join(run_dir, "agent_report_raw.txt"), raw)
        return {
            "summary": "Agent output failed validation.",
            "detected_issues": [],
            "likely_causes": [],
            "recommended_tests": [],
            "mitigations": [],
            "limits": errors,
            "narrative_markdown": "## Fairness Audit Narrative\n\n### Limits / Unknowns\n- Agent output failed schema validation.\n",
        }

    return obj


def main():
    run_dir = os.environ.get("FAIRNESS_RUN_DIR", DEFAULT_RUN_DIR)
    model = os.environ.get("FAIRNESS_AGENT_MODEL", DEFAULT_MODEL)
    config_path = os.environ.get("FAIRNESS_CONFIG", DEFAULT_CONFIG_PATH)

    report = run_agent(run_dir=run_dir, model=model, config_path=config_path)
    out_path = os.path.join(run_dir, "agent_report.json")
    write_json(out_path, report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
