from src.model import build_model
import argparse
import json, os
from datetime import datetime

import pandas as pd
import yaml
from data.dataset import load_prepared_data
from sklearn.metrics import accuracy_score

def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run_training(config: dict) -> dict:
    X_train, X_test, y_train, y_test, sens_train, sens_test = load_prepared_data(config)
    model_type = config.get("model", {}).get("type", "logistic_regression")
    model = build_model(X_train, model_type=model_type)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = None

    # Prefer probability for the configured positive class
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        classes = [str(c) for c in model.classes_]

        positive_label = str(config.get("positive_label"))
        if positive_label in classes:
            pos_idx = classes.index(positive_label)
            y_score = proba[:, pos_idx]
        elif proba.shape[1] == 2:
            # fallback: second column for binary classification
            y_score = proba[:, 1]

    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    sensitive_cols = config.get("sensitive_cols", []) or []
    preds_df = pd.DataFrame({
        "row_index": X_test.index,
        "y_true": y_test.values,
        "y_pred": y_pred,
    })
    if y_score is not None:
        preds_df["y_score"] = y_score

    for col in sensitive_cols:
        preds_df[col] = sens_test[col].values

    acc = accuracy_score(y_test, y_pred)

    results = {
        "accuracy": float(acc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_type": model_type,
        "label_col": config.get("label_col"),
        "positive_label": config.get("positive_label"),
        "sensitive_cols": sensitive_cols,
    }

    results["has_y_score"] = y_score is not None
    if y_score is not None and hasattr(model, "classes_"):
        results["model_classes"] = [str(c) for c in model.classes_]

    os.makedirs("outputs/runs", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs","runs",run_id)
    os.makedirs(out_dir, exist_ok=True)

    preds_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    X_test.to_csv(os.path.join(out_dir, "X_test_features.csv"), index=False)
    _write_json(os.path.join(out_dir, "metrics.json"), results)

    latest_dir = os.path.join("outputs", "runs", "latest")
    os.makedirs(latest_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(latest_dir, "predictions.csv"), index=False)
    X_test.to_csv(os.path.join(latest_dir, "X_test_features.csv"), index=False)
    _write_json(os.path.join(latest_dir, "metrics.json"), results)

    return {"out_dir": out_dir, "results": results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/audit_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    output = run_training(config)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
    
    

