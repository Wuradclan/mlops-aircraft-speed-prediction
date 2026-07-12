from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src.preprocessing import (
        CATEGORICAL_COLUMNS,
        DEFAULT_TARGET_COLUMN,
        NUMERIC_COLUMNS,
        clean_airplane_data,
    )
except ModuleNotFoundError:
    from preprocessing import (
        CATEGORICAL_COLUMNS,
        DEFAULT_TARGET_COLUMN,
        NUMERIC_COLUMNS,
        clean_airplane_data,
    )

TARGET_COLUMN = DEFAULT_TARGET_COLUMN
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = "data/Aiplane_BlueBook.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model.pkl"

def build_preprocessor(feature_columns):
    numeric_features = [
        column for column in feature_columns
        if column in NUMERIC_COLUMNS
    ]

    categorical_features = [
        column for column in feature_columns
        if column in CATEGORICAL_COLUMNS
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    return preprocessor

def build_models(feature_columns):
    return {
        "Linear Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_columns),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(feature_columns),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def main():
    X_train, X_test, y_train, y_test, cleaned_df = clean_airplane_data(
        DATA_PATH,
        target_column=TARGET_COLUMN,
    )

    print("Data cleaned successfully")
    print("Target column:", TARGET_COLUMN)
    print("Cleaned data shape:", cleaned_df.shape)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("-" * 40)

    models = build_models(X_train.columns.tolist())

    best_model = None
    best_rmse = float("inf")
    best_model_name = None

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print("MAE:", metrics["mae"])
        print("MSE:", metrics["mse"])
        print("RMSE:", metrics["rmse"])
        print("R²:", metrics["r2"])
        print("-" * 40)

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model = model
            best_model_name = model_name

    if best_model is None:
        raise RuntimeError("No model was successfully trained.")

    MODEL_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_OUTPUT_PATH,
    )

    print("Training completed")
    print("Best model:", best_model_name)
    print("Best RMSE:", best_rmse)
    print(f"Model saved to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
