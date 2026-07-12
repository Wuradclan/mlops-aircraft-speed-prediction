from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.config import file_path as default_file_path
except ModuleNotFoundError:
    from config import file_path as default_file_path

CATEGORICAL_COLUMNS = ["Company", "Engine Type"]
NUMERIC_COLUMNS = [
    "HP or lbs thr ea engine",
    "Max speed Knots",
    "Rcmnd cruise Knots",
    "Stall Knots dirty",
    "Fuel gal/lbs",
    "All eng service ceiling",
    "All eng rate of climb",
    "Takeoff over 50ft",
    "Takeoff ground run",
    "Landing over 50ft",
    "Landing ground roll",
    "Gross weight lbs",
    "Empty weight lbs",
    "Range N.M.",
]
PREDICTION_EXCLUDED_COLUMNS = [
    "Landing over 50ft",
    "Takeoff over 50ft",
    "Takeoff ground run",
    "Landing ground roll",
]
DATA_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
DEFAULT_TARGET_COLUMN = "Max speed Knots"


def resolve_dataset_path(csv_path=None) -> Path:
    dataset_path = Path(csv_path) if csv_path is not None else Path(default_file_path)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent.parent / dataset_path
    return dataset_path

def load_airplane_data(csv_path=None) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(csv_path)
    df = pd.read_csv(dataset_path)
    df = df[DATA_COLUMNS].copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown")

    return df

def clean_airplane_data(csv_path=None, target_column=DEFAULT_TARGET_COLUMN, test_size=0.2, random_state=42):
    df = load_airplane_data(csv_path)
    df = df.dropna(subset=[target_column]).copy()

    cleaned_df = df.copy()
    x = cleaned_df.drop(columns=[target_column])
    y = cleaned_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, cleaned_df