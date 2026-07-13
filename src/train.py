import argparse
from pathlib import Path
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor       # Nouvel import

# Gestion robuste des imports selon l'endroit d'où le script est lancé
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
    # Sécurité pour s'assurer que la colonne cible (Target) n'est jamais dans les features
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


# ==========================================
# 🏭 USINES À MODÈLES (FACTORY FUNCTIONS)
# Chaque fonction est isolée et gère ses propres hyperparamètres
# ==========================================

def build_linear_pipeline(preprocessor):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])


def build_ridge_pipeline(preprocessor, alpha):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=alpha, random_state=42))
    ])


def build_lasso_pipeline(preprocessor, alpha):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Lasso(alpha=alpha, random_state=42))
    ])


def build_rf_pipeline(preprocessor, n_estimators, max_depth):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        ))
    ])


def build_xgboost_pipeline(preprocessor, n_estimators, max_depth, learning_rate):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        ))
    ])


# ==========================================
# 📋 REGISTRE DES EXPÉRIMENTATIONS
# C'est ici qu'on orchestre nos 7 runs très proprement
# ==========================================

def get_experiment_models(feature_columns, args):
    preprocessor = build_preprocessor(feature_columns)

    return {
        # Modèles de base (Baselines avec valeurs fixes pour comparer)
        "run_01_linear_baseline": build_linear_pipeline(preprocessor),
        "run_02_ridge_baseline": build_ridge_pipeline(preprocessor, alpha=0.1),
        "run_04_rf_baseline": build_rf_pipeline(preprocessor, n_estimators=50, max_depth=5),

        # Modèles sur-mesure (Écoutent les commandes du terminal)
        "run_03_ridge_custom": build_ridge_pipeline(preprocessor, alpha=args.alpha),
        "run_05_rf_custom": build_rf_pipeline(preprocessor, n_estimators=args.n_estimators, max_depth=args.max_depth),
        "run_06_lasso_custom": build_lasso_pipeline(preprocessor, alpha=args.alpha),
        "run_07_xgboost_custom": build_xgboost_pipeline(preprocessor, n_estimators=args.n_estimators,
                                                        max_depth=args.max_depth, learning_rate=args.learning_rate)
    }

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def main():

    # Configuration des arguments de la ligne de commande
    parser = argparse.ArgumentParser(description="Entraînement des modèles d'avions avec hyperparamètres")
    #xgboost
    parser.add_argument("--model_type", type=str, default="random_forest",
                        choices=["linear", "ridge", "lasso", "random_forest", "xgboost"])

    parser.add_argument("--alpha", type=float, default=1.0, help="Paramètre alpha pour le modèle Ridge Custom")
    parser.add_argument("--n_estimators", type=int, default=100, help="Nombre d'arbres pour le RF Custom")
    parser.add_argument("--max_depth", type=int, default=10, help="Profondeur max pour le RF Custom")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Taux d'apprentissage pour XGBoost")
    args = parser.parse_args()

    # 1. Chargement et nettoyage des données
    X_train, X_test, y_train, y_test, cleaned_df = clean_airplane_data(
        DATA_PATH,
        target_column=TARGET_COLUMN,
    )
    print("--- COLONNES ATTENDUES PAR LE MODÈLE ---")
    print(X_train.columns.tolist())

    print("Données nettoyées avec succès")
    print(f"Shape de X_train: {X_train.shape}")
    print("-" * 40)

    # 2. Récupération des modèles incluant les arguments CLI
    models = get_experiment_models(X_train.columns.tolist(), args)

    best_model = None
    best_rmse = float("inf")
    best_run_name = None

    # 3. Configuration MLOps globale
    #mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # Si on est sur Mac (pas de variable définie), on utilise "http://localhost:5050"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
    mlflow.set_tracking_uri(tracking_uri)

    print(f"Connexion à MLflow sur : {mlflow.get_tracking_uri()}")  # Pour debug
    mlflow.set_experiment("Prediction_Vitesse_Avion")


    # 4. Boucle d'entraînement
    for run_name, pipeline in models.items():
        print(f"Entraînement du run : {run_name} ...")

        with mlflow.start_run(run_name=run_name):
            pipeline.fit(X_train, y_train)
            metrics = evaluate_model(pipeline, X_test, y_test)

            # Extraction dynamique du modèle final dans le pipeline pour les logs
            model_step = pipeline.named_steps["model"]
            mlflow.log_param("model_type", type(model_step).__name__)

            # Log des paramètres spécifiques selon le type de modèle
            if isinstance(model_step, RandomForestRegressor):
                mlflow.log_param("n_estimators", model_step.n_estimators)
                mlflow.log_param("max_depth", model_step.max_depth)
            elif isinstance(model_step, Ridge):
                mlflow.log_param("alpha", model_step.alpha)

            # Log des métriques
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2", metrics["r2"])

            # Sauvegarde du modèle (correction de sécurité cloudpickle et renommage explicite)
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

            print(f"RMSE: {metrics['rmse']:.2f} | R²: {metrics['r2']:.2f}")
            print("-" * 40)

            # Mise à jour du meilleur modèle
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model = pipeline
                best_run_name = run_name

    if best_model is None:
        raise RuntimeError("Aucun modèle n'a été entraîné avec succès.")

    # 5. Sauvegarde physique du grand gagnant pour FastAPI
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_OUTPUT_PATH)

    print(f"✅ Entraînement terminé !")
    print(f"🏆 Le meilleur modèle est issu du run : {best_run_name} (RMSE: {best_rmse:.2f})")
    print(f"💾 Il a été sauvegardé pour l'API dans : {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()