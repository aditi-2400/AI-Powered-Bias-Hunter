import os, json, yaml, argparse
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.evaluate_fairness_metrics import get_latest_run_dir

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    with open(path, "r") as f:
        return json.load(f)
    
def fmt_float(x, ndigits=4):
    try:
        if x is None:
            return "NA"
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return "NA"


def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _resolve_artifact_path(latest_dir: str, output_file: str) -> str:
    if os.path.exists(output_file):
        return output_file
    # Handle run-summary paths like outputs/runs/latest/...
    if output_file.startswith("outputs/runs/latest/"):
        rel = output_file.replace("outputs/runs/latest/", "", 1)
        candidate = os.path.join(latest_dir, rel)
        if os.path.exists(candidate):
            return candidate
    return output_file


def _summarize_threshold_sensitivity(obj: Dict[str, Any]) -> List[str]:
    lines = []
    for attr, payload in (obj or {}).items():
        results = payload.get("threshold_results", [])
        if not results:
            continue
        best = min(
            results,
            key=lambda r: float(r.get("difference", {}).get("true_positive_rate", 1e9)),
        )
        tpr_diff = best.get("difference", {}).get("true_positive_rate")
        fpr_diff = best.get("difference", {}).get("false_positive_rate")
        lines.append(
            f"- `{attr}`: lowest TPR gap at threshold {best.get('threshold')} "
            f"(TPR diff={fmt_float(tpr_diff)}, FPR diff={fmt_float(fpr_diff)})"
        )
    return lines


def _summarize_slice_scan(obj: Dict[str, Any]) -> List[str]:
    top = (obj or {}).get("top_slices", [])[:5]
    lines = []
    for row in top:
        lines.append(
            f"- `{row.get('feature')}={row.get('value')}` (n={row.get('n')}): "
            f"selection={fmt_float(row.get('selection_rate'))}, "
            f"TPR={fmt_float(row.get('true_positive_rate'))}, "
            f"FPR={fmt_float(row.get('false_positive_rate'))}"
        )
    return lines

def render_section(title: str, section: dict) -> str:
    """
    section expected shape:
      {
        "by_group": {metric_name: {group: value, ...}, ...} OR {group: {metric: value}} depending on to_dict(),
        "difference": {metric_name: value, ...},
        "ratio": {metric_name: value, ...},
        "flags": {metric_name: true/false, ...}  (optional)
      }
    """
    lines = []
    lines.append(f"## {title}\n")

    # Differences & flags
    diffs = section.get("difference", {})
    ratios = section.get("ratio", {})
    flags = section.get("flags", {})

    if diffs:
        lines.append("### Disparity summary\n")
        lines.append("| Metric | Difference (max-min) | Ratio (min/max) | Flag |")
        lines.append("|---|---:|---:|---|")
        for metric_name in diffs.keys():
            diff_val = diffs.get(metric_name)
            ratio_val = ratios.get(metric_name)
            flag_val = flags.get(metric_name, False)
            lines.append(
                f"| {metric_name} | {fmt_float(diff_val)} | {fmt_float(ratio_val)} | {'YES' if flag_val else 'NO'} |"
            )
        lines.append("")

    # By-group table (best-effort)
    by_group = section.get("by_group", {})
    if by_group:
        lines.append("### By-group metrics\n")

        # Fairlearn's MetricFrame.by_group.to_dict() can be nested either way depending on pandas version
        # Try to normalize into: group -> metric -> value
        group_to_metrics = {}

        # Case A: {metric: {group: value}}
        if by_group and all(isinstance(v, dict) for v in by_group.values()):
            # Heuristic: if keys look like metric names (selection_rate, true_positive_rate, etc.)
            # then assume metric->group->value
            metric_names = list(by_group.keys())
            sample_inner = by_group[metric_names[0]]
            if isinstance(sample_inner, dict):
                for metric_name, group_map in by_group.items():
                    for group_name, val in group_map.items():
                        group_to_metrics.setdefault(str(group_name), {})[metric_name] = val

        # If normalization failed, fall back to string dump
        if not group_to_metrics:
            lines.append("Could not normalize by_group format; raw dump:\n")
            lines.append("```json")
            lines.append(json.dumps(by_group, indent=2, default=str))
            lines.append("```")
            lines.append("")
            return "\n".join(lines)

        metric_cols = sorted({m for gm in group_to_metrics.values() for m in gm.keys()})
        lines.append("| Group | " + " | ".join(metric_cols) + " |")
        lines.append("|---|" + "|".join(["---:"] * len(metric_cols)) + "|")
        for group_name, mvals in group_to_metrics.items():
            row = [group_name] + [fmt_float(mvals.get(m)) for m in metric_cols]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/audit_config.yaml")
    args = parser.parse_args()

    # config is optional here (used mainly for thresholds, labels, metadata)
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    latest_dir = get_latest_run_dir()

    metrics_path = os.path.join(latest_dir, "metrics.json")
    fairness_path = os.path.join(latest_dir, "fairness_report.json")

    metrics = load_json(metrics_path)
    fairness = load_json(fairness_path)

    threshold = config.get("fairness_threshold", None)
    min_group_size = config.get("min_group_size", None)

    lines = []
    lines.append("# Bias Audit Report\n")
    lines.append(f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC_\n")

    # Run summary
    lines.append("## Run summary\n")
    lines.append(f"- Accuracy: **{fmt_float(metrics.get('accuracy'))}**")
    lines.append(f"- Train size: **{metrics.get('n_train', 'NA')}**")
    lines.append(f"- Test size: **{metrics.get('n_test', 'NA')}**")
    if threshold is not None:
        lines.append(f"- Fairness threshold: **{threshold}**")
    if min_group_size is not None:
        lines.append(f"- Minimum group size: **{min_group_size}**")
    lines.append("")

    # Fairness sections
    sensitive_cols = config.get("sensitive_cols", []) or []
    attrs_to_render = [attr for attr in sensitive_cols if attr in fairness]
    if not attrs_to_render:
        attrs_to_render = list(fairness.keys())

    for attr in attrs_to_render:
        title_attr = str(attr).replace("_", " ")
        lines.append(render_section(f"Fairness by {title_attr}", fairness[attr]))

    # Simple “what to look at” section (no LLM, deterministic)
    lines.append("## Findings\n")
    flagged = []
    for attr in attrs_to_render:
        sec = fairness.get(attr, {})
        flags = sec.get("flags", {})
        for metric_name, is_flagged in (flags or {}).items():
            if is_flagged:
                flagged.append((attr, metric_name, sec.get("difference", {}).get(metric_name)))
    if flagged:
        lines.append("The following disparities exceeded the configured threshold:\n")
        for attr, metric_name, diff_val in flagged:
            lines.append(f"- **{attr}**: **{metric_name}** difference = **{fmt_float(diff_val)}**")
    else:
        lines.append("No disparities exceeded the configured threshold (based on current flags).")
    lines.append("")

    # Diagnostics section
    diag_summary = load_json_if_exists(os.path.join(latest_dir, "diagnostics_run_summary.json"))
    if diag_summary:
        lines.append("## Diagnostics run\n")
        executed = diag_summary.get("executed", []) or []
        errors = diag_summary.get("errors", []) or []
        if executed:
            lines.append("### Executed diagnostics\n")
            for item in executed:
                tool = item.get("tool", "unknown")
                args = item.get("args", {}) or {}
                lines.append(f"- `{tool}` args={json.dumps(args, ensure_ascii=False)}")

            lines.append("")
            lines.append("### Diagnostic highlights\n")
            for item in executed:
                tool = item.get("tool")
                artifact = _resolve_artifact_path(latest_dir, item.get("output_file", ""))
                payload = load_json_if_exists(artifact)
                if payload is None:
                    lines.append(f"- `{tool}`: output artifact not found")
                    continue

                if tool == "run_threshold_sensitivity":
                    summary_lines = _summarize_threshold_sensitivity(payload)
                    lines.append(f"- `{tool}`")
                    lines.extend(summary_lines or ["- no threshold results available"])
                elif tool == "run_slice_scan":
                    summary_lines = _summarize_slice_scan(payload)
                    lines.append(f"- `{tool}`")
                    lines.extend(summary_lines or ["- no slice results available"])
                else:
                    lines.append(f"- `{tool}`: artifact generated at `{artifact}`")
            lines.append("")

        if errors:
            lines.append("### Diagnostic errors\n")
            for err in errors:
                lines.append(
                    f"- `{err.get('tool', 'unknown')}` args={json.dumps(err.get('args', {}), ensure_ascii=False)}: "
                    f"{err.get('error', 'unknown error')}"
                )
            lines.append("")

    # Agent explanation section
    agent_report = load_json_if_exists(os.path.join(latest_dir, "agent_report.json"))
    if agent_report:
        lines.append("## Agent explanation\n")
        if agent_report.get("summary"):
            lines.append(f"- Summary: {agent_report.get('summary')}")
        likely_causes = agent_report.get("likely_causes", []) or []
        if likely_causes:
            for cause in likely_causes:
                lines.append(f"- Likely cause: {cause}")
        limits = agent_report.get("limits", []) or []
        if limits:
            for lim in limits:
                lines.append(f"- Limit: {lim}")
        lines.append("")

        narrative = agent_report.get("narrative_markdown")
        if isinstance(narrative, str) and narrative.strip():
            lines.append("### Narrative\n")
            lines.append(narrative)
            lines.append("")

    out_md = os.path.join(latest_dir, "report.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved report to: {out_md}")


if __name__ == "__main__":
    main()
