import json, os
from datetime import datetime
import argparse
import pandas as pd
import yaml
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate

def get_latest_run_dir():
    latest_dir = os.path.join("outputs","runs","latest")

    if not os.path.exists(latest_dir):
        raise FileNotFoundError("No latest run directory found. Run training first")
    
    return latest_dir

def load_predictions(latest_dir: str) -> pd.DataFrame:
    path = os.path.join(latest_dir, "predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("predictions.csv not found. Run training first")
    return pd.read_csv(path)

def compute_metric_frame(metrics: dict, y_true: pd.Series, y_pred: pd.Series, 
                    sens_features: pd.Series):
    return MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sens_features
    )

def add_flags(section: dict, threshold: float) -> dict:
    diffs = section["difference"]
    section["flags"] = {
        k: (v is not None and float(v) > threshold)
        for k, v in diffs.items()
    }
    return section

def compute_group_sizes(preds: pd.DataFrame, sensitive_cols: list[str]) -> dict:
    out = {}
    for col in sensitive_cols:
        counts = preds[col].value_counts(dropna=False).to_dict()
        out[col] = {str(k): int(v) for k, v in counts.items()}
    return out

def evaluate_fairness(run_dir: str, config: dict) -> dict:
    preds = load_predictions(run_dir)

    positive_label = config.get("positive_label")
    sensitive_cols = config.get("sensitive_cols", []) or []
    threshold = float(config.get("fairness_threshold", 0.05))

    required_cols = ["y_true", "y_pred"] + sensitive_cols
    missing = [c for c in required_cols if c not in preds.columns]
    if missing:
        raise ValueError(
            f"predictions.csv missing required columns for fairness evaluation: {missing}"
        )
    
    group_sizes = compute_group_sizes(preds, sensitive_cols)
    y_true = (preds["y_true"] == positive_label).astype(int)
    y_pred = (preds["y_pred"] == positive_label).astype(int)

    metrics = {
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    }

    report = {}
    summary_rows = []

    for attr in sensitive_cols:
        mf = compute_metric_frame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sens_features=preds[attr],
        )

        section = {
            "by_group": mf.by_group.to_dict(),
            "difference": mf.difference().to_dict(),
            "ratio": mf.ratio().to_dict(),
        }
        section = add_flags(section, threshold)
        report[attr] = section

        by = mf.by_group
        for group_name, row in by.iterrows():
            summary_rows.append({
                "sensitive_attribute": attr,
                "group": str(group_name),
                **{k: float(row[k]) for k in row.index},
            })

    out_path = os.path.join(run_dir, "fairness_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    group_sizes_path = os.path.join(run_dir, "group_sizes.json")
    with open(group_sizes_path, "w", encoding="utf-8") as f:
        json.dump(group_sizes, f, indent=2, default=str)

    summary_path = os.path.join(run_dir, "fairness_by_group.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    # keep latest mirrored
    latest_dir = os.path.join("outputs", "runs", "latest")
    if os.path.abspath(run_dir) != os.path.abspath(latest_dir):
        os.makedirs(latest_dir, exist_ok=True)
        with open(os.path.join(latest_dir, "fairness_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(latest_dir, "fairness_by_group.csv"),
            index=False,
        )
    with open(os.path.join(latest_dir, "group_sizes.json"), "w", encoding="utf-8") as f:
            json.dump(group_sizes, f, indent=2, default=str)

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/audit_config.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_dir = args.run_dir or get_latest_run_dir()
    report = evaluate_fairness(run_dir, config)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()