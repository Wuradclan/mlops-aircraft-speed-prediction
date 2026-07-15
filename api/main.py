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
    """Cherche et charge dynamiquement le meilleur modèle depuis MLflow"""
    global best_model, model_name_info

    try:
        # On force l'adresse du conteneur MLflow sur le réseau Docker interne
        mlflow.set_tracking_uri("http://mlflow:5000")

        # 1. Trouver l'expérience
        experiment = mlflow.get_experiment_by_name("Prediction_Vitesse_Avion")
        if not experiment:
            print("Erreur : L'expérience MLflow est introuvable.")
            return

        # 2. Chercher le meilleur run (le plus petit RMSE)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.rmse ASC"],
            max_results=1
        )

        if runs.empty:
            print("Erreur : Aucun run trouvé dans l'expérience.")
            return

        best_run = runs.iloc[0]
        best_run_id = best_run["run_id"]

        # Récupération du nom du modèle (paramètre qu'on a logué plus tôt)
        model_name_info = best_run.get("params.model_type", "Inconnu (PyFunc)")
        print(f"🏆 Meilleur modèle trouvé : {model_name_info} (Run: {best_run_id})")

        # 3. Charger le modèle via l'interface universelle pyfunc
        model_uri = f"runs:/{best_run_id}/model"
        best_model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Modèle chargé en mémoire avec succès !")

    except Exception as e:
        print(f"❌ Erreur critique lors du chargement MLflow : {e}")
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