import os, json, yaml, argparse
from datetime import datetime
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
    if "sex" in fairness:
        lines.append(render_section("Fairness by sex", fairness["sex"]))
    if "age_group" in fairness:
        lines.append(render_section("Fairness by age group", fairness["age_group"]))

    # Simple “what to look at” section (no LLM, deterministic)
    lines.append("## Findings\n")
    flagged = []
    for attr in ["sex", "age_group"]:
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

    out_md = os.path.join(latest_dir, "report.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved report to: {out_md}")


if __name__ == "__main__":
    main()
