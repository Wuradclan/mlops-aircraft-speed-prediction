from fastapi import FastAPI, HTTPException
import pandas as pd
import traceback  # Ajout pour voir l'erreur précise
from src.predict import get_prediction

app = FastAPI()


@app.post("/predict")
async def predict(data: dict):
    try:
        # 1. Création du DataFrame
        df = pd.DataFrame([data])

        # 2. Liste rigoureuse des colonnes attendues (l'ordre compte !)
        expected_cols = [
            "Engine Type", "HP or lbs thr ea engine",
            "Stall Knots dirty", "Fuel gal/lbs", "All eng service ceiling",
            "All eng rate of climb", "Gross weight lbs", "Empty weight lbs", "Range N.M."
        ]
        # Si 'Company' est encore dans df, supprimez-le avant de prédire
        if "Company" in df.columns:
            df = df.drop(columns=["Company"])

        # 3. Réindexation : Force les colonnes dans le bon ordre et supprime les inutiles
        # Si une colonne manque, elle sera créée avec des valeurs NaN (que votre imputer gérera)
        df = df.reindex(columns=expected_cols)

        # 4. Prédiction
        prediction = get_prediction(df)

        return {
            "prediction": float(prediction[0]),
            "model_name": "RandomForestRegressor"
        }
    except Exception as e:
        # Si ça plante, on saura enfin POURQUOI
        return {"error": f"Erreur de prédiction : {str(e)}"}