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
import optuna
from sklearn.model_selection import KFold
from category_encoders import TargetEncoder # <-- Import nécessaire en haut de trai
from sklearn.model_selection import cross_val_predict

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
    # Utilisation du TargetEncoder au lieu du OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("target_encoder", TargetEncoder())
    ])

    # # 2. On l'utilise ICI dans le ColumnTransformer
    # preprocessor = ColumnTransformer(transformers=[
    #     ("numeric", numeric_transformer, numeric_features),
    #     ("categorical", Pipeline(steps=[("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    #                                     ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    # ])
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features),
    ])

    return preprocessor



# ⚠️ MODIFICATION : On passe les paramètres explicitement pour permettre à Optuna de les modifier
def get_experiment_models(feature_columns, n_estimators=100, max_depth=10, learning_rate=0.1, alpha=1.0):
    preprocessor = build_preprocessor(feature_columns)

    base_models = [
        ('xgb', XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)),
        ('extra_trees', ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth)),
        ('rf', RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)),
        ('ridge', Ridge(alpha=alpha))
    ]

    meta_model = RidgeCV(alphas=np.logspace(-3, 3, 10))

    # 2. On fixe l'aléatoire de la validation croisée du Stacking
    cv_fixed = KFold(n_splits=5, shuffle=True, random_state=42)

    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv_fixed,
        n_jobs=4
    )

    return {
        "run_01_linear": Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())]),
        "run_02_ridge": Pipeline(steps=[("preprocessor", preprocessor), ("model", Ridge(alpha=alpha))]),
        "run_05_lasso": Pipeline(steps=[("preprocessor", preprocessor), ("model", Lasso(alpha=alpha))]),
        "run_03_rf": Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth))]),
        "run_04_xgboost": Pipeline(steps=[("preprocessor", preprocessor), ("model", XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate))]),
        "run_07_extra_trees": Pipeline(steps=[("preprocessor", preprocessor), ("model", ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth))]),
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

    # 1. On récupère la liste exacte de toutes les colonnes SAUF la cible
    x_columns = [col for col in df.columns if col != target]

    aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=42, project_name="airplane_vitesse")
    #aml.train(y=target, training_frame=train)
    # 2. On ajoute explicitement le paramètre `x` pour bloquer toute fuite de données
    aml.train(x=x_columns, y=target, training_frame=train)
    # On retourne le modèle leader ET l'objet aml complet pour le leaderboard
    return aml.leader, aml, test

def calculate_metrics(y_true, y_pred):
    """Calcule le RMSE, le R2 et la MAE pour une paire de valeurs."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2, mae


def log_and_save_model(pipeline, metrics, model_path, model_type, params=None):
    """Centralise le logging MLflow, la sauvegarde locale et les prints."""
    if params:
        mlflow.log_params(params)

    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
    joblib.dump(pipeline, model_path)

    print(f"\n✅ Modèle {model_type} terminé.")
    print(f"🔴 RMSE Train: {metrics.get('rmse_train', 0):.2f}")
    print(f"🟡 RMSE CV:    {metrics.get('rmse_cv', 0):.2f}")
    print(f"🟢 RMSE Test:  {metrics.get('rmse_test', 0):.2f}")

def log_and_save_h2o(best_model, metrics, model_type, params=None):
    """Log et enregistre spécifiquement pour H2O."""
    if params:
        mlflow.log_params(params)

    mlflow.log_metrics(metrics)
    mlflow.h2o.log_model(best_model, name="model")

    print(f"\n✅ Modèle {model_type} terminé.")
    print(f"🔴 RMSE Train: {metrics.get('rmse_train', 0):.2f}")
    print(f"🟡 RMSE CV:    {metrics.get('rmse_cv', 0):.2f}")
    print(f"🟢 RMSE Test:  {metrics.get('rmse_test', 0):.2f}")


def train_evaluate_and_log(pipeline, X_train, y_train, X_test, y_test, model_path, model_type, params=None,
                           is_optimized=False):
    """Effectue l'entraînement complet, l'évaluation et le logging."""

    # 1. Validation croisée
    print(f"🔄 Calcul de la validation croisée (xval) pour {model_type}...")
    cv_preds = cross_val_predict(pipeline, X_train, y_train, cv=5, n_jobs=-1)
    rmse_cv, r2_cv, mae_cv = calculate_metrics(y_train, cv_preds)

    # 2. Entraînement final
    pipeline.fit(X_train, y_train)

    # 3. Prédictions Train
    preds_train = pipeline.predict(X_train)
    rmse_train, r2_train, mae_train = calculate_metrics(y_train, preds_train)

    # 4. Prédictions Test
    preds_test = pipeline.predict(X_test)
    rmse_test, r2_test, mae_test = calculate_metrics(y_test, preds_test)

    # Dictionnaire des métriques harmonisé
    metrics = {
        "rmse": rmse_cv, "r2": r2_cv, "mae": mae_cv,
        "rmse_train": rmse_train, "r2_train": r2_train, "mae_train": mae_train,
        "rmse_cv": rmse_cv, "r2_cv": r2_cv,
        "rmse_test": rmse_test, "r2_test": r2_test, "mae_test": mae_test
    }

    # On ajoute le flag optimized si nécessaire
    if is_optimized and params:
        params["optimized"] = True
    # Appel de la fonction de sauvegarde mutualisée
    log_and_save_model(pipeline, metrics, model_path, model_type, params=params)
    return metrics


def log_mlflow_data(params, metrics):
    """Log simple des paramètres et métriques (utilisé par Optuna ET le Champion)."""
    if params:
        mlflow.log_params(params)
    mlflow.log_metrics(metrics)

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
    #args = parser.parse_args()

    # ⚠️ NOUVEAUX ARGUMENTS POUR OPTUNA
    parser.add_argument("--tune", action="store_true", help="Lancer l'optimisation Optuna")
    parser.add_argument("--n_trials", type=int, default=20, help="Nombre d'essais Optuna")

    parser.add_argument("--n_est_min", type=int, default=100)
    parser.add_argument("--n_est_max", type=int, default=1000)
    parser.add_argument("--depth_min", type=int, default=3)
    parser.add_argument("--depth_max", type=int, default=15)

    args = parser.parse_args()

    # 1. Chargement
    X_train, X_test, y_train, y_test, cleaned_df = clean_airplane_data(DATA_PATH, target_column=TARGET_COLUMN)

    # 2. Configuration MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050"))
    mlflow.set_experiment("Prediction_Vitesse_Avion")
    # CORRECTION MAJEURE : LE DICTIONNAIRE EST DÉFINI ICI, ACCESSIBLE PAR TOUTES LES BRANCHES
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

    # 3. Branche H2O
    if args.model_type == "h2o":
        print("🚀 Lancement H2O AutoML...")
        with mlflow.start_run(run_name="run_h2o_automl"):
            best_model, aml_obj, test_frame = train_h2o_automl(cleaned_df, TARGET_COLUMN)

            # --- Évaluation ---
            perf_train = best_model.model_performance(train=True)
            perf_test = best_model.model_performance(test_data=test_frame)
            perf_cv = best_model.model_performance(xval=True)

            nom_algo = best_model.algo
            print("\n📊 === H2O LEADERBOARD ===")
            print(aml_obj.leaderboard.as_data_frame())

            # --- Feature Importance (On garde ton bloc intact) ---
            print("\n🧮 === FEATURE IMPORTANCE ===")
            try:
                varimp = best_model.varimp(use_pandas=True)
                if varimp is not None:
                    print(varimp.head(10))
                    varimp.to_csv("h2o_feature_importance.csv", index=False)
                    mlflow.log_artifact("h2o_feature_importance.csv")
                else:
                    print("Le leader est un StackedEnsemble. Regardons les poids :")
                    print(best_model.metalearner().coef())
            except Exception as e:
                print(f"Impossible d'extraire la feature importance : {e}")

            # --- Dictionnaire des métriques (Harmonisé) ---
            metrics = {
                "rmse": perf_cv.rmse(),  # Le score maître pour l'API (CV)
                "r2": perf_cv.r2(),
                "rmse_train": perf_train.rmse(),  # Détecteur de mémorisation
                "r2_train": perf_train.r2(),
                "rmse_cv": perf_cv.rmse(),  # Doublon visuel
                "rmse_test": perf_test.rmse(),  # Généralisation finale
                "r2_test": perf_test.r2()
            }

            # --- Appel de la fonction mutualisée spécialisée H2O ---
            log_and_save_h2o(
                best_model,
                metrics,
                f"H2O_{nom_algo.upper()}",
                params={"model_type": f"H2O_{nom_algo.upper()}"}
            )
    # 4. Branche OPTUNA
    elif args.tune:
        print(f"🎯 Démarrage de l'optimisation Optuna pour {args.model_type} ({args.n_trials} trials)...")

        # Le Run Parent qui contiendra toute l'étude
        with mlflow.start_run(run_name=f"Optuna_Study_{args.model_type}") as parent_run:

            def objective(trial):
                # 4.1 Définition dynamique des paramètres
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", args.n_est_min, args.n_est_max, step=50),
                    "max_depth": trial.suggest_int("max_depth", args.depth_min, args.depth_max),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True)
                }

                # 4.2 Génération du modèle
                trial_models = get_experiment_models(X_train.columns.tolist(), **params)
                trial_pipeline = trial_models.get(run_name_mapping.get(args.model_type))

                # 4.3 Sous-Run MLflow (Nested)
                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    # A. Calcul des métriques (Réutilisation de tes fonctions)

                    # 1. Validation croisée
                    cv_preds = cross_val_predict(trial_pipeline, X_train, y_train, cv=5, n_jobs=-1)
                    rmse_cv, _, _ = calculate_metrics(y_train, cv_preds)

                    # 2. Entraînement et prédictions Train/Test
                    trial_pipeline.fit(X_train, y_train)
                    rmse_train, _, _ = calculate_metrics(y_train, trial_pipeline.predict(X_train))
                    rmse_test, _, _ = calculate_metrics(y_test, trial_pipeline.predict(X_test))

                    # B. Calcul du Score de Robustesse
                    gap = abs(rmse_train - rmse_test)
                    robust_score = rmse_test + (0.5 * gap)

                    # C. Logging unifié (Réutilisation de ta fonction log_mlflow_data)
                    trial_metrics = {
                        "rmse_cv": rmse_cv,
                        "rmse_train": rmse_train,
                        "rmse_test": rmse_test,
                        "robust_score": robust_score
                    }

                    # On ajoute le model_type aux params pour l'API
                    full_params = {**params, "model_type": args.model_type}

                    log_mlflow_data(params=full_params, metrics=trial_metrics)

                    # D. Retour du score à Optuna
                    return robust_score
            # 4.4 Lancement de l'étude
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.n_trials)

            print(f"\n🏆 Meilleurs paramètres trouvés : {study.best_params}")

            # 4.5 Entraînement du Modèle Champion
            print("🚀 Évaluation et Entraînement final du modèle Champion...")
            champ_models = get_experiment_models(X_train.columns.tolist(),
                                                 n_estimators=study.best_params.get("n_estimators", args.n_estimators),
                                                 max_depth=study.best_params.get("max_depth", args.max_depth),
                                                 learning_rate=study.best_params.get("learning_rate",
                                                                                     args.learning_rate),
                                                 alpha=study.best_params.get("alpha", args.alpha))

            champion_pipeline = champ_models.get(run_name_mapping.get(args.model_type))

            # --- Évaluation ---
            train_evaluate_and_log(
                champion_pipeline, X_train, y_train, X_test, y_test,
                MODEL_OUTPUT_PATH, args.model_type, params=study.best_params, is_optimized=True
            )

    # 4. Branche Classique
    else:
        # CORRECTION : On passe les arguments individuels pour correspondre à la nouvelle fonction
        models = get_experiment_models(X_train.columns.tolist(),
                                       n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       learning_rate=args.learning_rate,
                                       alpha=args.alpha)

        run_name = run_name_mapping.get(args.model_type)
        pipeline = models.get(run_name)
        if pipeline is None:
            raise ValueError(f"Le modèle '{args.model_type}' n'est pas configuré correctement.")

        with mlflow.start_run(run_name=run_name):
            train_evaluate_and_log(
                pipeline, X_train, y_train, X_test, y_test,
                MODEL_OUTPUT_PATH, args.model_type, params=vars(args)
            )


if __name__ == "__main__":
    main()