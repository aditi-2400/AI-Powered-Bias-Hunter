import pandas as pd
import urllib.request
from io import StringIO
from sklearn.model_selection import train_test_split

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

COLUMN_NAMES = [
    "checking_account_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_status",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"
]

def download_dataset() -> StringIO:
    with urllib.request.urlopen(URL) as response:
        data = response.read().decode("utf-8")
    return StringIO(data)

def load_raw_dataset() -> pd.DataFrame:
    data_stream = download_dataset()
    df = pd.read_csv(data_stream, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['credit_risk'] = df["credit_risk"].map({1: "good", 2: "bad"})

    df['sex'] = df["personal_status_sex"].apply(
        lambda x: "female" if x in ["A92", "A95"] else "male"
    )

    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 25, 40, 60, 120],
        labels=['young','adult','middle','senior']
    )
    return df

def validate_dataset(df: pd.DataFrame):
    if df.isnull().sum().sum() != 0:
        raise ValueError("Dataset contains missing values")
    
    if df.shape[0] < 500:
        raise ValueError("Dataset too small")
    
    if "credit_risk" not in df.columns:
        raise ValueError("Target column missing")

def split_dataset(df: pd.DataFrame, label_col: str = "credit_risk", test_size: float = 0.25, random_state: int = 42):
    X = df.drop(columns=[label_col])
    y = df[label_col]

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def load_prepared_data(test_size: float = 0.25, random_state: int = 42):
    df = load_raw_dataset()
    df = clean_dataset(df)
    validate_dataset(df)

    return split_dataset(
        df,
        test_size=test_size,
        random_state=random_state
    )
