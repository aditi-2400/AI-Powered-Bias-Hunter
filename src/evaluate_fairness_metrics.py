import json, os
from datetime import datetime
import argparse
import pandas as pd
import yaml
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
from data.dataset import load_raw_dataset, clean_dataset

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

def load_sensitive_features(row_index:pd.Series) -> pd.DataFrame:
    df = load_raw_dataset()
    df = clean_dataset(df)
    subset = df.loc[row_index]
    return subset[["sex", "age_group"]]

def compute_metrics(metrics: dict, y_true: pd.Series, y_pred: pd.Series, 
                    sens_features: pd.DataFrame, sens_cols: str):
    return MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sens_features[sens_cols]
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/audit_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    latest_dir = get_latest_run_dir()

    preds = load_predictions(latest_dir)
    sens = load_sensitive_features(preds["row_index"])

    positive_label = config.get("positive_label", "good")
    y_true = (preds["y_true"] == positive_label).astype(int)
    y_pred = (preds["y_pred"] == positive_label).astype(int)

    metrics = {
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    }

    mf_age = compute_metrics(metrics, y_true, y_pred, sens, "age_group")
    mf_sex = compute_metrics(metrics, y_true, y_pred, sens, "sex")

    report = {
        "sex": {
            "by_group": mf_sex.by_group.to_dict(),
            "difference": mf_sex.difference().to_dict(),
            "ratio": mf_sex.ratio().to_dict(),
        },
        "age_group": {
            "by_group": mf_age.by_group.to_dict(),
            "difference": mf_age.difference().to_dict(),
            "ratio": mf_age.ratio().to_dict(),
        },
    }

    threshold = float(config.get("fairness_threshold", "0.05"))
    def add_flags(section: dict) -> dict:
        diffs = section["difference"]
        section["flags"] = {k: (v is not None and float(v) > threshold) for k,v in diffs.items()}
        return section
    
    report["sex"] = add_flags(report["sex"])
    report["age_group"] = add_flags(report["age_group"])

    out_path = os.path.join(latest_dir, "fairness_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    summary_rows = []
    for attr, mf in [("sex", mf_sex), ("age_group", mf_age)]:
        by = mf.by_group
        for group_name, row in by.iterrows():
            summary_rows.append({
                "sensitive_attribute": attr,
                "group": str(group_name),
                **{k: float(row[k]) for k in row.index},
            })
    pd.DataFrame(summary_rows).to_csv(os.path.join(latest_dir, "fairness_by_group.csv"), index=False)

    print(f"Saved fairness report to: {out_path}")


if __name__ == "__main__":
    main()