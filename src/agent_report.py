from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from src.agent_common import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL,
    DEFAULT_RUN_DIR,
    call_ollama,
    load_evidence,
    load_yaml,
    parse_model_json,
    write_json,
    write_text,
)

REQUIRED_OUTPUT_KEYS = [
    "summary",
    "detected_issues",
    "likely_causes",
    "recommended_tests",
    "mitigations",
    "limits",
    "narrative_markdown",
]


def classify_severity(difference: Any, threshold: float) -> str:
    try:
        value = float(difference)
    except (TypeError, ValueError):
        return "unknown"

    mild = 0.5 * threshold
    moderate = 1.0 * threshold
    severe = 2.0 * threshold

    if value >= severe:
        return "severe"
    if value >= moderate:
        return "moderate"
    if value >= mild:
        return "mild"
    return "unknown"


def normalize_issue_severities(obj: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    issues = obj.get("detected_issues")
    if not isinstance(issues, list):
        return obj

    fairness_report = evidence.get("fairness_report", {}) or {}
    cfg = evidence.get("audit_config", {}) or {}
    threshold = float(cfg.get("fairness_threshold", 0.05))

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        attribute = issue.get("attribute")
        metric = issue.get("metric")
        if not isinstance(attribute, str) or not isinstance(metric, str):
            issue["severity"] = "unknown"
            continue

        diff = (
            fairness_report.get(attribute, {})
            .get("difference", {})
            .get(metric)
        )
        issue["severity"] = classify_severity(diff, threshold)

    return obj


def _normalize_test_name(name: str) -> str:
    cleaned = name.strip()
    if cleaned.startswith("run_"):
        return cleaned[len("run_") :]
    return cleaned


def _completed_diagnostic_keys(evidence: Dict[str, Any]) -> set[str]:
    completed: set[str] = set()

    for key in (evidence.get("diagnostics") or {}).keys():
        completed.add(_normalize_test_name(str(key)))

    # Backward-compatible single-file artifacts.
    if "group_sizes" in evidence:
        completed.add("check_group_sample_sizes")
    if "threshold_sensitivity" in evidence:
        completed.add("threshold_sensitivity")
    if "slice_report" in evidence:
        completed.add("slice_scan")
    if "distribution_report" in evidence:
        completed.add("feature_distribution_comparison")
    if "proxy_report" in evidence:
        completed.add("proxy_detection")

    return completed


def _is_already_completed(test_name: str, completed: set[str]) -> bool:
    normalized = _normalize_test_name(test_name)
    if normalized in completed:
        return True

    # Example: recommended "run_threshold_sensitivity" but only
    # "threshold_sensitivity__<attribute>" exists in diagnostics.
    if "__" not in normalized:
        return any(c.startswith(f"{normalized}__") for c in completed)
    return False


def filter_recommended_tests(obj: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    tests = obj.get("recommended_tests")
    if not isinstance(tests, list):
        return obj

    completed = _completed_diagnostic_keys(evidence)
    filtered = [
        t for t in tests
        if isinstance(t, str) and not _is_already_completed(t, completed)
    ]
    obj["recommended_tests"] = filtered
    return obj


def _replace_recommended_section(markdown: str, replacement: str) -> str:
    pattern = (
        r"(?ms)^(#{2,3}\s+Recommended next tests\s*\n)"
        r"(.*?)(?=^#{2,3}\s+|\Z)"
    )

    def _repl(match: re.Match[str]) -> str:
        heading = match.group(1)
        return f"{heading}{replacement}\n"

    return re.sub(pattern, _repl, markdown, count=1)


def harmonize_narrative_markdown(obj: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    narrative = obj.get("narrative_markdown")
    tests = obj.get("recommended_tests")
    if not isinstance(narrative, str) or not isinstance(tests, list):
        return obj

    completed = _completed_diagnostic_keys(evidence)

    if tests:
        lines = [f"- {t}" for t in tests if isinstance(t, str)]
        replacement = "\n".join(lines) if lines else "- None"
    else:
        replacement = "- No additional diagnostics recommended (existing diagnostics already executed)."

    updated = _replace_recommended_section(narrative, replacement)

    # Remove stale instructions that ask to rerun already-completed diagnostics.
    if "check_group_sample_sizes" in completed:
        updated = re.sub(
            r"\s*Run check_group_sample_sizes to verify group sizes\.?",
            "",
            updated,
            flags=re.IGNORECASE,
        )

    obj["narrative_markdown"] = updated
    return obj


def build_report_prompt(evidence: Dict[str, Any]) -> str:
    cfg = evidence.get("audit_config", {})
    threshold = cfg.get("fairness_threshold", 0.05)
    mild = 0.5 * threshold
    moderate = 1.0 * threshold
    severe = 2.0 * threshold

    summary_evidence = {
    "fairness_report": evidence.get("fairness_report"),
    "metrics": evidence.get("metrics"),
    "group_sizes": evidence.get("group_sizes"),
    "agent_plan": evidence.get("agent_plan"),
    "diagnostics_keys": list(evidence.get("diagnostics", {}).keys()),
}

    evidence_str = json.dumps(summary_evidence, indent=2, ensure_ascii=False)

    return f"""
You are an AI fairness auditor. You will receive JSON evidence produced by a deterministic pipeline.

GOAL:
Write a final fairness report for a human reader using:
- fairness metrics
- group sizes
- any diagnostics that were already executed
- the planning context if agent_plan is present

STRICT RULES:
- Use ONLY the provided JSON evidence. Do NOT invent numbers.
- Do NOT compute new metrics. Only interpret what exists.
- If evidence is missing for a claim, write "insufficient evidence" in limits.
- Output MUST be valid JSON (one top-level object) and NOTHING else.
- narrative_markdown MUST be a valid JSON string (use \\n for new lines).
- Do NOT use triple quotes \"\"\".
- Do NOT use ``` fences.
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
- If agent_plan exists, use it only as planning context; do not claim a diagnostic was run unless its artifact is present in evidence.
- If diagnostics or diagnostic artifacts are present, use them when explaining likely causes, mitigations, and limits.
- Distinguish clearly between:
  - findings supported by fairness_report
  - hypotheses supported by diagnostics
  - tests that are still recommended next

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

GUIDANCE FOR RECOMMENDED_TESTS:
- recommended_tests should contain only tests that are still useful to run next.
- Do NOT include diagnostics that have already been run and whose artifacts are already present.
- If diagnostics already provide partial evidence but not a conclusion, you may recommend follow-up tests.
- If group_sizes.json already exists, do NOT recommend check_group_sample_sizes again.
- If diagnostics.feature_distribution__<attribute> exists, do NOT recommend run_feature_distribution_comparison again for that attribute.
- If diagnostics.proxy_detection__<attribute> exists, do NOT recommend run_proxy_detection again for that attribute.
- If diagnostics.threshold_sensitivity__<attribute> exists, do NOT recommend run_threshold_sensitivity again for that attribute.
- If diagnostics.slice_scan exists, do NOT recommend run_slice_scan again.

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


def validate_report_output(obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
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

            if "attribute" not in issue or not isinstance(issue.get("attribute"), str):
                errors.append(f"detected_issues[{i}].attribute must be a string")

            if issue.get("metric") not in valid_metrics:
                errors.append(
                    f"detected_issues[{i}].metric must be one of {sorted(valid_metrics)}"
                )

            if issue.get("severity") not in valid_severities:
                errors.append(
                    f"detected_issues[{i}].severity must be one of {sorted(valid_severities)}"
                )

            if "issue_type" not in issue or not isinstance(issue.get("issue_type"), str):
                errors.append(f"detected_issues[{i}].issue_type must be a string")

            evidence = issue.get("evidence")
            if not isinstance(evidence, list):
                errors.append(f"detected_issues[{i}].evidence must be a list")
            elif not all(isinstance(x, str) for x in evidence):
                errors.append(f"detected_issues[{i}].evidence must contain only strings")

    for key in ["likely_causes", "recommended_tests", "mitigations", "limits"]:
        if key in obj and not isinstance(obj[key], list):
            errors.append(f"{key} must be a list")
        elif key in obj and isinstance(obj[key], list):
            if not all(isinstance(x, str) for x in obj[key]):
                errors.append(f"{key} must contain only strings")

    return (len(errors) == 0), errors


def run_agent_report(
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

    prompt = build_report_prompt(evidence)
    write_text(os.path.join(run_dir, "agent_report_prompt.txt"), prompt)

    raw = call_ollama(model, prompt)
    try:
        obj = parse_model_json(raw)
    except ValueError as e:
        write_text(os.path.join(run_dir, "agent_report_raw.txt"), raw)
        return {
            "summary": "Agent output was not valid JSON.",
            "detected_issues": [],
            "likely_causes": [],
            "recommended_tests": [],
            "mitigations": [],
            "limits": [str(e)],
            "narrative_markdown": "## Fairness Audit Narrative\n\n### Limits / Unknowns\n- Agent response was not valid JSON.\n",
        }

    ok, errors = validate_report_output(obj)
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

    obj = normalize_issue_severities(obj, evidence)
    obj = filter_recommended_tests(obj, evidence)
    obj = harmonize_narrative_markdown(obj, evidence)
    return obj


def main():
    run_dir = os.environ.get("FAIRNESS_RUN_DIR", DEFAULT_RUN_DIR)
    model = os.environ.get("FAIRNESS_AGENT_MODEL", DEFAULT_MODEL)
    config_path = os.environ.get("FAIRNESS_CONFIG", DEFAULT_CONFIG_PATH)

    report = run_agent_report(run_dir=run_dir, model=model, config_path=config_path)
    out_path = os.path.join(run_dir, "agent_report.json")
    write_json(out_path, report)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
