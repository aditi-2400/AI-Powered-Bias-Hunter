from __future__ import annotations

import pandas as pd


def apply_preprocessing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    label_col = config.get("label_col")
    label_map = config.get("label_map")
    if label_col and label_map:
        normalized_map = {str(k): v for k, v in label_map.items()}
        df[label_col] = df[label_col].astype(str).map(normalized_map)

    for new_col, spec in (config.get("derived_columns") or {}).items():
        col_type = spec.get("type")
        source = spec.get("source")

        if source not in df.columns:
            raise ValueError(f"Missing source column for derived column '{new_col}': {source}")

        if col_type == "map":
            mapping = {str(k): v for k, v in (spec.get("mapping") or {}).items()}
            df[new_col] = df[source].astype(str).map(mapping)

        elif col_type == "bin":
            bins = spec.get("bins")
            labels = spec.get("labels")
            if bins is None or labels is None:
                raise ValueError(f"Derived column '{new_col}' of type 'bin' requires bins and labels")
            df[new_col] = pd.cut(df[source], bins=bins, labels=labels)

        else:
            raise ValueError(f"Unsupported derived column type for '{new_col}': {col_type}")

    return df