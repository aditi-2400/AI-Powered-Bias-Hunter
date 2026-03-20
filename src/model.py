from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd

def build_preprocessor(X):
    # Robust dtype handling:
    # - numeric: int/float (and optionally bool)
    # - categorical: object/category/bool (if you prefer)
    # - exclude datetime columns from both (or coerce upstream)
    if not isinstance(X, pd.DataFrame):
        raise ValueError("build_preprocessor expects a pandas DataFrame")

    datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    X2 = X.drop(columns=datetime_cols, errors="ignore")

    # Treat bool as numeric (0/1). If you want bool categorical, move 'bool' into cat include.
    num_cols = X2.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X2.columns if c not in num_cols]
 
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        # LR benefits from scaling; with_mean=False keeps it compatible with sparse matrices
        ("scaler", StandardScaler(with_mean=False)),
    ])
 
    categorical = Pipeline(steps=[
         ("imputer", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ]
    )
    
def build_model(X, model_type: str = 'logistic_regression') -> Pipeline:
    pre = build_preprocessor(X)
    if model_type == 'logistic_regression':
        clf = LogisticRegression(max_iter=2000)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return Pipeline(steps=[
        ("preprocess", pre),
        ("classifier", clf)
    ])