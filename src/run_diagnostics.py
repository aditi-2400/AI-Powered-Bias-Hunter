from __future__ import annotations

import json
import os
from typing import Any, Dict, Callable, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)

from src.agent_common import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_RUN_DIR,
    load_json,
    load_yaml,
    write_json,
)


def load_predictions(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing predictions.csv: {path}")
    return pd.read_csv(path)


def load_features(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "X_test_features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing X_test_features.csv: {path}. "
            "Update training to save test features for diagnostics."
        )
    return pd.read_csv(path)


def ensure_diagnostics_dir(run_dir: str) -> str:
    diagnostics_dir = os.path.join(run_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    return diagnostics_dir


def _build_generic_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )


def run_check_group_sample_sizes(
    run_dir: str,
    config: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    preds = load_predictions(run_dir)
    sensitive_cols = config.get("sensitive_cols", []) or []

    out = {}
    for col in sensitive_cols:
        if col not in preds.columns:
            continue
        counts = preds[col].value_counts(dropna=False).to_dict()
        out[col] = {str(k): int(v) for k, v in counts.items()}
    return out


def run_feature_distribution_comparison(
    run_dir: str,
    config: Dict[str, Any],
    attribute: str,
    **kwargs,
) -> Dict[str, Any]:
    preds = load_predictions(run_dir)
    X = load_features(run_dir)

    if attribute not in preds.columns:
        raise ValueError(f"Attribute '{attribute}' not found in predictions.csv")

    groups = preds[attribute].astype(str)
    counts = groups.value_counts()
    unique_groups = counts.index.tolist()

    if len(unique_groups) < 2:
        return {
            "attribute": attribute,
            "error": f"Need at least 2 groups for comparison, found {len(unique_groups)}"
        }

    g1, g2 = unique_groups[0], unique_groups[1]
    mask1 = groups == g1
    mask2 = groups == g2

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_summary = []
    for col in numeric_cols:
        s1 = pd.to_numeric(X.loc[mask1, col], errors="coerce")
        s2 = pd.to_numeric(X.loc[mask2, col], errors="coerce")

        m1 = float(s1.mean()) if not s1.dropna().empty else None
        m2 = float(s2.mean()) if not s2.dropna().empty else None
        diff = None if m1 is None or m2 is None else float(abs(m1 - m2))

        numeric_summary.append(
            {
                "feature": col,
                "group_a": str(g1),
                "group_b": str(g2),
                "mean_a": m1,
                "mean_b": m2,
                "abs_mean_diff": diff,
            }
        )

    numeric_summary = sorted(
        numeric_summary,
        key=lambda x: float("-inf") if x["abs_mean_diff"] is None else -x["abs_mean_diff"],
    )

    categorical_summary = []
    for col in categorical_cols:
        dist1 = X.loc[mask1, col].astype(str).value_counts(normalize=True).head(5).to_dict()
        dist2 = X.loc[mask2, col].astype(str).value_counts(normalize=True).head(5).to_dict()

        all_keys = set(dist1) | set(dist2)
        tvd = 0.5 * sum(abs(dist1.get(k, 0.0) - dist2.get(k, 0.0)) for k in all_keys)

        categorical_summary.append(
            {
                "feature": col,
                "group_a": str(g1),
                "group_b": str(g2),
                "top_dist_a": {str(k): float(v) for k, v in dist1.items()},
                "top_dist_b": {str(k): float(v) for k, v in dist2.items()},
                "total_variation_distance": float(tvd),
            }
        )

    categorical_summary = sorted(
        categorical_summary,
        key=lambda x: -x["total_variation_distance"],
    )

    return {
        "attribute": attribute,
        "compared_groups": [str(g1), str(g2)],
        "group_counts": {str(k): int(v) for k, v in counts.to_dict().items()},
        "top_numeric_shifts": numeric_summary[:10],
        "top_categorical_shifts": categorical_summary[:10],
    }


def run_proxy_detection(
    run_dir: str,
    config: Dict[str, Any],
    attribute: str,
    **kwargs,
) -> Dict[str, Any]:
    preds = load_predictions(run_dir)
    X = load_features(run_dir)

    if attribute not in preds.columns:
        raise ValueError(f"Attribute '{attribute}' not found in predictions.csv")

    y = preds[attribute].astype(str)
    classes = sorted(y.dropna().unique().tolist())

    if len(classes) != 2:
        return {
            "attribute": attribute,
            "error": f"Proxy detection currently supports binary attribute only; found {len(classes)} groups."
        }

    y_bin = (y == classes[1]).astype(int)

    pre = _build_generic_preprocessor(X)
    clf = Pipeline(
        steps=[
            ("preprocess", pre),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    clf.fit(X, y_bin)
    y_pred = clf.predict(X)

    result = {
        "attribute": attribute,
        "classes": classes,
        "accuracy": float(accuracy_score(y_bin, y_pred)),
    }

    if hasattr(clf.named_steps["classifier"], "predict_proba"):
        y_score = clf.predict_proba(X)[:, 1]
        result["auc"] = float(roc_auc_score(y_bin, y_score))
        auc = result["auc"]
        if auc >= 0.8:
            risk = "high"
        elif auc >= 0.65:
            risk = "moderate"
        else:
            risk = "low"
        result["risk_level"] = risk

    return result


def run_slice_scan(
    run_dir: str,
    config: Dict[str, Any],
    **kwargs,
) -> Dict[str, Any]:
    preds = load_predictions(run_dir)
    X = load_features(run_dir)

    # Use a small number of low-cardinality columns to form slices
    candidate_cols: List[str] = []
    for col in X.columns:
        nunique = X[col].nunique(dropna=False)
        if nunique <= 8:
            candidate_cols.append(col)

    candidate_cols = candidate_cols[:5]

    if not candidate_cols:
        return {
            "error": "No low-cardinality feature columns available for slice scan."
        }

    positive_label = config.get("positive_label")
    y_true = (preds["y_true"] == positive_label).astype(int)
    y_pred = (preds["y_pred"] == positive_label).astype(int)

    rows = []
    for col in candidate_cols:
        for value, idx in X.groupby(col).groups.items():
            idx = list(idx)
            if len(idx) < 10:
                continue

            yt = y_true.iloc[idx]
            yp = y_pred.iloc[idx]

            selection_rate = float(yp.mean()) if len(yp) else None
            positives = yt.sum()
            negatives = len(yt) - positives

            tpr = None
            if positives > 0:
                tpr = float(((yp == 1) & (yt == 1)).sum() / positives)

            fpr = None
            if negatives > 0:
                fpr = float(((yp == 1) & (yt == 0)).sum() / negatives)

            rows.append(
                {
                    "feature": col,
                    "value": str(value),
                    "n": int(len(idx)),
                    "selection_rate": selection_rate,
                    "true_positive_rate": tpr,
                    "false_positive_rate": fpr,
                }
            )

    rows = sorted(
        rows,
        key=lambda r: (
            -(r["selection_rate"] if r["selection_rate"] is not None else -1),
            -r["n"],
        ),
    )

    return {
        "candidate_features": candidate_cols,
        "top_slices": rows[:20],
    }


def run_threshold_sensitivity(
    run_dir: str,
    config: Dict[str, Any],
    attribute: str | None = None,
    **kwargs,
) -> Dict[str, Any]:
    preds = load_predictions(run_dir)

    if "y_score" not in preds.columns:
        return {
            "error": "predictions.csv does not contain y_score. Save model probabilities or scores during training."
        }

    positive_label = config.get("positive_label")
    sensitive_cols = config.get("sensitive_cols", []) or []
    target_attributes = [attribute] if attribute else sensitive_cols

    y_true = (preds["y_true"] == positive_label).astype(int)
    y_score = pd.to_numeric(preds["y_score"], errors="coerce")

    metrics = {
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    }

    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}

    for attr in target_attributes:
        if attr not in preds.columns:
            continue

        attr_results = []
        for thr in thresholds:
            y_pred_thr = (y_score >= thr).astype(int)

            mf = MetricFrame(
                metrics=metrics,
                y_true=y_true,
                y_pred=y_pred_thr,
                sensitive_features=preds[attr],
            )

            attr_results.append(
                {
                    "threshold": float(thr),
                    "difference": {
                        k: float(v) if v is not None else None
                        for k, v in mf.difference().to_dict().items()
                    },
                    "ratio": {
                        k: float(v) if v is not None else None
                        for k, v in mf.ratio().to_dict().items()
                    },
                }
            )

        results[attr] = {
            "attribute": attr,
            "threshold_results": attr_results,
        }

    return results


DIAGNOSTIC_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "check_group_sample_sizes": run_check_group_sample_sizes,
    "run_feature_distribution_comparison": run_feature_distribution_comparison,
    "run_proxy_detection": run_proxy_detection,
    "run_slice_scan": run_slice_scan,
    "run_threshold_sensitivity": run_threshold_sensitivity,
}


def run_diagnostics(
    run_dir: str = DEFAULT_RUN_DIR,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    config = load_yaml(config_path)
    plan_path = os.path.join(run_dir, "agent_plan.json")
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"Missing agent_plan.json: {plan_path}")

    plan = load_json(plan_path)
    requests = plan.get("requested_diagnostics", [])

    diagnostics_dir = ensure_diagnostics_dir(run_dir)
    results = {"executed": [], "errors": []}

    for item in requests:
        tool = item.get("tool")
        args = item.get("args", {}) or {}

        if tool not in DIAGNOSTIC_REGISTRY:
            results["errors"].append(
                {
                    "tool": tool,
                    "error": f"Unsupported diagnostic tool: {tool}",
                }
            )
            continue

        try:
            output = DIAGNOSTIC_REGISTRY[tool](run_dir=run_dir, config=config, **args)

            if tool == "check_group_sample_sizes":
                out_path = os.path.join(run_dir, "group_sizes.json")
            elif tool == "run_feature_distribution_comparison":
                attr = args.get("attribute", "unknown")
                out_path = os.path.join(diagnostics_dir, f"feature_distribution__{attr}.json")
            elif tool == "run_proxy_detection":
                attr = args.get("attribute", "unknown")
                out_path = os.path.join(diagnostics_dir, f"proxy_detection__{attr}.json")
            elif tool == "run_slice_scan":
                out_path = os.path.join(diagnostics_dir, "slice_scan.json")
            elif tool == "run_threshold_sensitivity":
                attr = args.get("attribute")
                if attr:
                    out_path = os.path.join(diagnostics_dir, f"threshold_sensitivity__{attr}.json")
                else:
                    out_path = os.path.join(diagnostics_dir, "threshold_sensitivity.json")
            else:
                out_path = os.path.join(diagnostics_dir, f"{tool}.json")

            write_json(out_path, output)
            results["executed"].append(
                {
                    "tool": tool,
                    "args": args,
                    "output_file": out_path,
                }
            )

        except Exception as e:
            err_obj = {
                "tool": tool,
                "args": args,
                "error": str(e),
            }
            results["errors"].append(err_obj)

            safe_name = tool
            if args.get("attribute"):
                safe_name = f"{tool}__{args.get('attribute')}"
            err_path = os.path.join(diagnostics_dir, f"{safe_name}__error.json")
            write_json(err_path, err_obj)

    write_json(os.path.join(run_dir, "diagnostics_run_summary.json"), results)
    return results


def main():
    run_dir = os.environ.get("FAIRNESS_RUN_DIR", DEFAULT_RUN_DIR)
    config_path = os.environ.get("FAIRNESS_CONFIG", DEFAULT_CONFIG_PATH)

    results = run_diagnostics(run_dir=run_dir, config_path=config_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()