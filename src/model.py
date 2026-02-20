from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
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