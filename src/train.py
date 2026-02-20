from src.model import build_model
import argparse
import json, os
from datetime import datetime

import pandas as pd
import yaml
from data.dataset import load_prepared_data
from sklearn.metrics import accuracy_score

def run_training(config: dict) -> dict:
    X_train, X_test, y_train, y_test = load_prepared_data(config["test_size"], config["random_state"])
    model = build_model(X_train, model_type=config.get("model", {}).get("type","logistic_regression"))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    preds_df = pd.DataFrame({
        "row_index":X_test.index,
        "y_true": y_test.values,
        "y_pred": y_pred
    })

    acc = accuracy_score(y_test, y_pred)

    results = {
        "accuracy": acc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    os.makedirs("outputs/runs", exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("outputs","runs",run_id)
    os.makedirs(out_dir, exist_ok=True)

    preds_df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    latest_dir = os.path.join("outputs", "runs", "latest")
    os.makedirs(latest_dir, exist_ok=True)
    preds_df.to_csv(os.path.join(latest_dir, "predictions.csv"), index=False)
    with open(os.path.join(latest_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return {"out_dir": out_dir, "results": results}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/audit_config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    output = run_training(config)
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
    
    

