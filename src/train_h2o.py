import argparse
from pathlib import Path
import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.h2o  # <--- Important pour le log H2O
import numpy as np
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

# Gestion des imports
try:
    from src.preprocessing import CATEGORICAL_COLUMNS, DEFAULT_TARGET_COLUMN, NUMERIC_COLUMNS, clean_airplane_data
except ModuleNotFoundError:
    from preprocessing import CATEGORICAL_COLUMNS, DEFAULT_TARGET_COLUMN, NUMERIC_COLUMNS, clean_airplane_data

TARGET_COLUMN = DEFAULT_TARGET_COLUMN
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = "data/Aiplane_BlueBook.csv"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model.pkl"


# --- USINES À MODÈLES (Scikit-Learn) ---
def build_preprocessor(feature_columns):
    numeric_features = [col for col in feature_columns if col in NUMERIC_COLUMNS and col != TARGET_COLUMN]
    categorical_features = [col for col in feature_columns if col in CATEGORICAL_COLUMNS]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("categorical", Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])
    return preprocessor


def get_experiment_models(feature_columns, args):
    preprocessor = build_preprocessor(feature_columns)

    # 1. Définition des modèles de base pour le Stacking
    base_models = [
        # 1. Le Champion (Arbres boostés)
        ('xgb',
         XGBRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate)),

        # 2. L'Arbre alternatif (Bagging robuste)
        ('extra_trees', ExtraTreesRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)),

        # 3. Les Linéaires (Stabilité)
        ('ridge', Ridge(alpha=args.alpha)),
        ('lasso', Lasso(alpha=args.alpha)),

        # 4. La Distance pure
        ('knn', KNeighborsRegressor(n_neighbors=5)),

        # 5. La Distance à Noyau (Relations non-linéaires)
        ('svr', SVR(kernel='rbf', C=10.0, gamma='scale')),

        # 6. Le Réseau de Neurones (Vision radicalement différente)
        ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, early_stopping=True))
    ]

    # 2. Le Meta-modèle (Apprend à combiner les prédictions)
    meta_model = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42)

    # 3. Construction du Stacking
    #stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,  # Validation croisée à 5 plis pour générer des prédictions robustes
        n_jobs=-1  # Utilise tous les cœurs de ton processeur
    )

    return {
        "run_01_linear": Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())]),
        "run_02_ridge": Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=args.alpha))]),
        "run_05_lasso": Pipeline(steps=[("preprocessor", preprocessor), ("model", Lasso(alpha=args.alpha))]),
        "run_03_rf": Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth))]),
        "run_04_xgboost": Pipeline(steps=[("preprocessor", preprocessor), ("model", XGBRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate))]),
        "run_06_stacking": Pipeline(steps=[("preprocessor", preprocessor), ("model", stacking_model)])
    }

# --- LOGIQUE H2O AUTOML ---
def train_h2o_automl(df, target, max_runtime_secs=120):
    h2o.init(nthreads=-1, strict_version_check=False)  # Utilise tous les cœurs CPU
    hf = h2o.H2OFrame(df)
    train, test = hf.split_frame(ratios=[0.8], seed=42)

    aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=42, project_name="airplane_vitesse")
    aml.train(y=target, training_frame=train)
    return aml.leader


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="xgboost",
                        choices=["linear", "ridge", "lasso", "random_forest", "xgboost", "h2o", "stacking", "svr",
                                 "knn"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    args = parser.parse_args()

    # 1. Chargement
    X_train, X_test, y_train, y_test, cleaned_df = clean_airplane_data(DATA_PATH, target_column=TARGET_COLUMN)

    # 2. Configuration MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment("Prediction_Vitesse_Avion")

    # 3. Branche H2O
    if args.model_type == "h2o":
        print("🚀 Lancement H2O AutoML...")
        with mlflow.start_run(run_name="run_h2o_automl"):
            best_model = train_h2o_automl(cleaned_df, TARGET_COLUMN)
            # Log des métriques
            perf = best_model.model_performance(test_data=None)  # Si besoin de test data, l'extraire du split
            nom_algo = best_model.algo
            mlflow.log_param("model_type", f"H2O_{nom_algo.upper()}")

            mlflow.log_metric("rmse", perf.rmse())
            mlflow.log_metric("mae", perf.mae())
            mlflow.log_metric("r2", perf.r2())
            mlflow.log_metric("mse", perf.mse())
            mlflow.h2o.log_model(best_model, name="model")
            print(f"H2O terminé. Leader RMSE: {perf.rmse()}")

        # 4. Branche Classique
    else:
        models = get_experiment_models(X_train.columns.tolist(), args)

        # Mapping explicite et robuste entre l'argument du terminal et la clé du modèle
        run_name_mapping = {
            "linear": "run_01_linear",
            "ridge": "run_02_ridge",
            "lasso": "run_05_lasso",
            "random_forest": "run_03_rf",
            "xgboost": "run_04_xgboost",
            "stacking": "run_06_stacking"
        }

        run_name = run_name_mapping.get(args.model_type)
        pipeline = models.get(run_name)

        if pipeline is None:
            raise ValueError(f"Le modèle '{args.model_type}' n'est pas configuré correctement.")

        with mlflow.start_run(run_name=run_name):
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            # 1. Calcul des métriques
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            # 2. LOGGING AUTOMATIQUE (La méthode Pro)
            # Cela enregistre TOUS tes arguments (n_estimators, max_depth, etc.) d'un seul coup
            mlflow.log_params(vars(args))

            # 3. Log des métriques
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "mse": mse, "r2": r2})

            # 4. Log du modèle
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
            joblib.dump(pipeline, MODEL_OUTPUT_PATH)

            print(f"✅ Modèle {args.model_type} terminé.")
            print(f"📊 RMSE: {rmse:.2f} | R2: {r2:.2f}")


if __name__ == "__main__":
    main()