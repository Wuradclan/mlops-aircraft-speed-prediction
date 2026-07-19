import os
from fastapi import FastAPI
import pandas as pd
import traceback
import mlflow
import mlflow.pyfunc

app = FastAPI()

# Variables globales pour stocker le meilleur modèle et son nom
best_model = None
model_name_info = "Aucun modèle chargé"


def load_best_model_from_mlflow():
    """Cherche et charge le meilleur modèle basé sur un score de robustesse."""
    global best_model, model_name_info

    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        experiment = mlflow.get_experiment_by_name("Prediction_Vitesse_Avion")
        if not experiment:
            return

        # 1. Récupération des runs valides
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="metrics.rmse_test >= 0 AND metrics.rmse_cv >= 0 AND metrics.rmse_train >= 0",
            max_results=50
        )

        if runs.empty:
            print("❌ Aucun modèle complet trouvé.")
            return

        # On exclut les sous-runs (qui n'ont pas de modèle physique)
        if 'tags.mlflow.parentRunId' in runs.columns:
            runs = runs[runs['tags.mlflow.parentRunId'].isnull()]

        # Par sécurité, on exclut aussi tout ce qui s'appelle "trial_"
        if 'tags.mlflow.runName' in runs.columns:
            runs = runs[~runs['tags.mlflow.runName'].str.startswith('trial_', na=False)]

        if runs.empty:
            print("❌ Aucun modèle physique valide trouvé après filtrage des trials.")
            return
        # --- NOUVELLE LOGIQUE INDUSTRIELLE ---

        # 1. Calcul du pourcentage de surapprentissage (Overfit %)
        runs['overfit_pct'] = (runs['metrics.rmse_test'] - runs['metrics.rmse_train']) / runs['metrics.rmse_test']
        runs['overfit_pct'] = runs['overfit_pct'].clip(lower=0)  # Sécurité pour les rares valeurs négatives

        # 2. Application du seuil de tolérance (30% d'écart maximum autorisé)
        TOLERANCE_THRESHOLD = 0.30
        valid_runs = runs[runs['overfit_pct'] <= TOLERANCE_THRESHOLD]

        # 3. Sélection du Champion
        if not valid_runs.empty:
            # Scénario idéal : on prend le modèle avec le meilleur RMSE_Test parmi ceux qui ne trichent pas
            best_run = valid_runs.loc[valid_runs['metrics.rmse_test'].idxmin()]
            print(f"🟢 Modèle validé selon les standards industriels (< {TOLERANCE_THRESHOLD * 100}% overfit).")
        else:
            # Fallback de sécurité : Si TOUS les modèles surapprennent, on prend le "plus stable"
            print("⚠️ Attention : Tous les modèles dépassent le seuil d'overfitting. Mode Fallback activé.")
            best_run = runs.loc[runs['overfit_pct'].idxmin()]

        best_run_id = best_run["run_id"]

        # Logs
        print(f"🏆 Champion détecté : {best_run.get('tags.mlflow.runName', 'Sans nom')}")
        print(f"   RMSE Test: {best_run['metrics.rmse_test']:.2f}")
        print(f"   Surapprentissage: {best_run['overfit_pct'] * 100:.1f}%")

        # 4. Chargement
        model_uri = f"runs:/{best_run_id}/model"
        best_model = mlflow.pyfunc.load_model(model_uri)

        # 5. Mise à jour de l'affichage
        model_type = str(best_run.get("params.model_type", "Modèle"))
        model_name_info = f"{model_type} (RMSE: {best_run['metrics.rmse_test']:.2f} | Overfit: {best_run['overfit_pct'] * 100:.1f}%)"

        print(f"✅ Modèle chargé : {model_name_info}")
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        import traceback
        traceback.print_exc()
# On exécute la recherche au démarrage de l'API
load_best_model_from_mlflow()


@app.post("/predict")
async def predict(data: dict):
    if best_model is None:
        return {"error": "L'API n'a pas pu charger de modèle depuis MLflow."}

    try:
        df = pd.DataFrame([data])

        expected_cols = [
            "Engine Type", "HP or lbs thr ea engine",
            "Stall Knots dirty", "Fuel gal/lbs", "All eng service ceiling",
            "All eng rate of climb", "Gross weight lbs", "Empty weight lbs", "Range N.M."
        ]
        if "Company" in df.columns:
            df = df.drop(columns=["Company"])

        df = df.reindex(columns=expected_cols)

        # PRÉDICTION UNIVERSELLE :
        # Que ce soit H2O ou XGBoost en dessous, pyfunc gère la conversion
        prediction = best_model.predict(df)

        # Selon le modèle, pyfunc renvoie parfois un array, une liste, ou un DataFrame
        # On sécurise l'extraction de la valeur
        if isinstance(prediction, pd.DataFrame):
            pred_value = float(prediction.iloc[0, 0])
        else:
            pred_value = float(prediction[0])

        return {
            "prediction": pred_value,
            "model_name": model_name_info
        }
    except Exception as e:
        return {"error": f"Erreur de prédiction : {str(e)}"}


@app.get("/model-info")
def get_model_info():
    """Renvoie le type du modèle qui a gagné la compétition MLflow."""
    if best_model is None:
        return {"status": "error", "model_name": "Aucun modèle"}
    return {"status": "success", "model_name": model_name_info}


@app.post("/reload-model")
def reload_model():
    """Force l'API à re-chercher le meilleur modèle sur MLflow sans redémarrer le conteneur"""
    try:
        # On appelle la fonction de chargement qui écrase la variable globale 'best_model'
        load_best_model_from_mlflow()

        if best_model is None:
            return {"status": "error", "message": "Échec du chargement. Vérifie les logs MLflow."}

        return {"status": "success", "message": f"Modèle '{model_name_info}' chargé avec succès."}
    except Exception as e:
        return {"status": "error", "message": str(e)}