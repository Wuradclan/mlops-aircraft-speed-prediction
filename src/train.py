from pathlib import Path
import joblib
import numpy as np

# ✅ AJOUT MLOPS : Importation de la librairie mlflow obligatoire
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # Ajout de Ridge pour diversifier les runs
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
    # ✅ AJOUT : Sécurité pour s'assurer que la colonne cible (Target) n'est jamais dans les features
    numeric_features = [col for col in feature_columns if col in NUMERIC_COLUMNS and col != TARGET_COLUMN]
    categorical_features = [col for col in feature_columns if col in CATEGORICAL_COLUMNS]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ])

    return preprocessor


# 🚨 CORRECTION EXIGENCE PROJET :
# L'ancienne fonction "build_models" ne générait que 2 modèles.
# Le barème exige un minimum de 5 runs. Nous avons donc créé 5 configurations distinctes ici.
def get_experiment_models(feature_columns):
    preprocessor = build_preprocessor(feature_columns)

    return {
        "run_01_linear_regression_baseline": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),
        "run_02_ridge_alpha_0_1": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=0.1, random_state=42))
        ]),
        "run_03_ridge_alpha_1_0": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0, random_state=42))
        ]),
        "run_04_random_forest_50_trees": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
        ]),
        "run_05_random_forest_100_trees": Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
        ])
    }


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def main():
    X_train, X_test, y_train, y_test, cleaned_df = clean_airplane_data(
        DATA_PATH,
        target_column=TARGET_COLUMN,
    )

    print("Données nettoyées avec succès")
    print("Shape de X_train:", X_train.shape)
    print("-" * 40)

    # Récupération de nos 5 configurations pour les 5 runs obligatoires
    models = get_experiment_models(X_train.columns.tolist())

    best_model = None
    best_rmse = float("inf")
    best_run_name = None

    # ✅ AJOUT MLOPS : Initialisation de l'expérience globale dans MLflow
    mlflow.set_experiment("Prediction_Vitesse_Avion")

    for run_name, pipeline in models.items():
        print(f"Entraînement du run : {run_name} ...")

        # ✅ AJOUT MLOPS : On englobe chaque itération dans un "Run" MLflow
        with mlflow.start_run(run_name=run_name):

            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)

            # ✅ AJOUT MLOPS : Traçabilité des paramètres du modèle
            model_step = pipeline.named_steps["model"]
            mlflow.log_param("model_type", type(model_step).__name__)

            if isinstance(model_step, RandomForestRegressor):
                mlflow.log_param("n_estimators", model_step.n_estimators)
                mlflow.log_param("max_depth", model_step.max_depth)
            elif isinstance(model_step, Ridge):
                mlflow.log_param("alpha", model_step.alpha)

            # ✅ AJOUT MLOPS : Traçabilité des métriques de performance
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2", metrics["r2"])

            # ✅ AJOUT MLOPS : Sauvegarde du modèle sous forme d'artefact dans MLflow
            mlflow.sklearn.log_model(pipeline, "model")

            print(f"RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']:.2f}")
            print("-" * 40)

            # Logique pour conserver en mémoire le meilleur des 5 runs (basé sur le plus petit RMSE)
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = pipeline
                best_run_name = run_name

    if best_model is None:
        raise RuntimeError("Aucun modèle n'a été entraîné avec succès.")

    # Sauvegarde finale physique du meilleur modèle pour qu'il soit lu par l'API FastAPI
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)

    print(f"✅ Entraînement terminé ! Le meilleur modèle est issu du run : {best_run_name}")
    print(f"🚀 Il a été sauvegardé pour l'API dans : {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()