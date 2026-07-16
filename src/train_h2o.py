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
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

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

    # 1. On définit le transformer numérique (Imputer + Scaler)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. On l'utilise ICI dans le ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                        ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])

    return preprocessor


def get_experiment_models(feature_columns, args):
    preprocessor = build_preprocessor(feature_columns)

    # 1. Uniquement des modèles robustes et performants en base
    base_models = [
        ('xgb', XGBRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate)),
        ('extra_trees', ExtraTreesRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)),
        ('rf', RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)),
        ('ridge', Ridge(alpha=args.alpha))
    ]

    # 2. Un méta-modèle linéaire robuste qui cherche le meilleur compromis (pondération)
    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 10))

    # 3. Construction du Stacking
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    return {
        "run_01_linear": Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())]),
        "run_02_ridge": Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=args.alpha))]),
        "run_05_lasso": Pipeline(steps=[("preprocessor", preprocessor), ("model", Lasso(alpha=args.alpha))]),
        "run_03_rf": Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth))]),
        "run_04_xgboost": Pipeline(steps=[("preprocessor", preprocessor), ("model", XGBRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate))]),
        "run_07_extra_trees": Pipeline(steps=[("preprocessor", preprocessor), ("model", ExtraTreesRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth))]),
        "run_08_knn": Pipeline(steps=[("preprocessor", preprocessor), ("model", KNeighborsRegressor(n_neighbors=5))]),
        "run_09_svr": Pipeline(steps=[("preprocessor", preprocessor), ("model", SVR(kernel='rbf', C=10.0, gamma='scale'))]),
        "run_10_mlp": Pipeline(steps=[("preprocessor", preprocessor), ("model", MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=500, early_stopping=True))]),
        "run_06_stacking": Pipeline(steps=[("preprocessor", preprocessor), ("model", stacking_model)])
    }

# --- LOGIQUE H2O AUTOML ---
def train_h2o_automl(df, target, max_runtime_secs=120):
    h2o.init(nthreads=-1, strict_version_check=False)
    hf = h2o.H2OFrame(df)
    train, test = hf.split_frame(ratios=[0.8], seed=42)

    aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=42, project_name="airplane_vitesse")
    aml.train(y=target, training_frame=train)
    return aml.leader


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()

    # ⚠️ CORRECTION : Ajout de TOUS les choix dans argparse
    parser.add_argument("--model_type", type=str, default="xgboost",
                        choices=["linear", "ridge", "lasso", "random_forest", "xgboost",
                                 "extra_trees", "knn", "svr", "mlp", "h2o", "stacking"])

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
            perf = best_model.model_performance(test_data=None)
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

        # ⚠️ CORRECTION : Ajout de TOUS les modèles dans le dictionnaire de mapping
        run_name_mapping = {
            "linear": "run_01_linear",
            "ridge": "run_02_ridge",
            "lasso": "run_05_lasso",
            "random_forest": "run_03_rf",
            "xgboost": "run_04_xgboost",
            "extra_trees": "run_07_extra_trees",
            "knn": "run_08_knn",
            "svr": "run_09_svr",
            "mlp": "run_10_mlp",
            "stacking": "run_06_stacking"
        }

        run_name = run_name_mapping.get(args.model_type)
        pipeline = models.get(run_name)

        if pipeline is None:
            raise ValueError(f"Le modèle '{args.model_type}' n'est pas configuré correctement.")

        with mlflow.start_run(run_name=run_name):
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            # Calcul des métriques
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            # LOGGING AUTOMATIQUE
            mlflow.log_params(vars(args))
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "mse": mse, "r2": r2})
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
            joblib.dump(pipeline, MODEL_OUTPUT_PATH)

            print(f"✅ Modèle {args.model_type} terminé.")
            print(f"📊 RMSE: {rmse:.2f} | R2: {r2:.2f}")


if __name__ == "__main__":
    main()