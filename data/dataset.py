from __future__ import annotations
import pandas as pd
from src.preprocessing import apply_preprocessing
import json, os
from sklearn.model_selection import train_test_split

def _load_schema_columns(schema_path: str):
    with open(schema_path, 'r') as f:
        obj = json.load(f)
    cols = obj.get("columns")
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        raise ValueError(f"Invalid schema file: {schema_path}. Expected {{'columns': [..]}}")
    return cols

def load_dataset_from_config(config: dict, dataset_path: str | None = None) -> pd.DataFrame:
    """
    Dataset-agnostic loader.

    Supports:
      - CSV/TSV/delimited text via pandas.read_csv
      - UCI-style '.data' (space/regex delimited) by specifying sep + header + schema_path/columns.

    Required config keys:
      config['dataset']['path'] (unless dataset_path override provided)
      config['label_col']
      (optional) config['dataset'] parsing options: format, sep, header, columns, schema_path, na_values, encoding
    """

    ds = config.get("dataset",{})
    path = dataset_path or ds.get('path')
    if not path:
        raise ValueError("Missing dataset path. Provide config['dataset']['path'] or dataset_path override.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    fmt = ds.get("format").lower()
    encoding = ds.get("encoding", "utf-8")
    na_values = ds.get("na_values", None)
    ext = os.path.splitext(path)[1].lower()
    if not fmt:
        if ext in [".csv"]:
            fmt = "csv"
        elif ext in [".tsv"]:
            fmt = "tsv"
        elif ext in [".data",".txt"]:
            fmt = "delimited"
        else:
            fmt = "csv"

    if fmt == "tsv":
        sep = "\t"
        header = True
    elif fmt == "delimited":
        sep = ds.get("sep", r"\s+")
        header = bool(ds.get("header", False))
    else:
        sep = ds.get("sep", ",")
        header = bool(ds.get("header", True))
    
    columns = ds.get("columns")
    schema_path = ds.get("schema_path")
    if not header:
        if schema_path:
            columns = _load_schema_columns(schema_path)
        if not columns:
            raise ValueError(
                "Dataset has no header. Provide dataset.columns or dataset.schema_path in config."
            )
        
    df = pd.read_csv(
        path, 
        sep=sep, 
        header=header if header else None,
        names = columns if not header else None,
        na_values=na_values,
        encoding=encoding,
        engine="python",
        )

    return df

def validate_dataset(df: pd.DataFrame, label_col: str, sensitive_cols: list[str]):
    sensitive_cols = sensitive_cols or []
    missing = [c for c in [label_col] + sensitive_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    uniq = df[label_col].dropna().unique()
    if len(uniq) != 2:
        raise ValueError(f"Binary-only: {label_col} has {len(uniq)} unique values: {uniq}")
    
def split_dataset(
        df: pd.DataFrame, 
        label_col: str, 
        sensitive_cols: list[str] | None = None,
        test_size: float = 0.25,
        random_state: float = 42,
        ):
    """
    Returns:
    X_train, X_test, y_train, y_test, sens_train, sens_test
    where X excludes label + sensitive cols.
    """
    sensitive_cols = sensitive_cols or []
    validate_dataset(df, label_col=label_col, sensitive_cols=sensitive_cols)
    y = df[label_col]
    sens = df[sensitive_cols].copy() if sensitive_cols else pd.DataFrame(index=df.index)
    X = df.drop(columns = ([label_col] + sensitive_cols))
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(X, y, sens, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test, sens_train, sens_test

def load_prepared_data(config: dict, dataset_path: str | None = None):
    """
    Backward-compatible wrapper used by the rest of the project:
      - loads dataset using config
      - validates binary task
      - splits using config split params
    """
    df = load_dataset_from_config(config=config, dataset_path=dataset_path)
    df = apply_preprocessing(df, config)

    label_col = config["label_col"]
    sensitive_cols = config.get("sensitive_cols", []) or []
    test_size = float(config.get("test_size", 0.25))
    random_state = int(config.get("random_state", 42))

    return split_dataset(
        df,
        label_col=label_col,
        sensitive_cols=sensitive_cols,
        test_size=test_size,
        random_state=random_state,
    )