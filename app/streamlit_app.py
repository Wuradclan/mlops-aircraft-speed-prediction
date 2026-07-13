import streamlit as st
import requests

# Configuration
st.set_page_config(page_title="Prédiction de Vitesse d'Avion", page_icon="✈️")

st.title("✈️ Prédiction de Vitesse d'Avion")
st.markdown("Modèle entraîné sans la colonne 'Company' pour éviter le surapprentissage.")

st.sidebar.header("Paramètres de l'avion")

# --- CHAMPS DE SAISIE (Correspondant exactement au modèle) ---

# Variable catégorielle
engine_type = st.sidebar.selectbox("Engine Type", ["Piston", "Turboprop", "Jet", "Unknown"])

# Variables numériques
hp = st.sidebar.number_input("HP or lbs thr ea engine", value=300.0)
stall_knots = st.sidebar.number_input("Stall Knots dirty", value=50.0)
fuel = st.sidebar.number_input("Fuel gal/lbs", value=100.0)
ceiling = st.sidebar.number_input("All eng service ceiling", value=15000.0)
rate_of_climb = st.sidebar.number_input("All eng rate of climb", value=1000.0)
gross_weight = st.sidebar.number_input("Gross weight lbs", value=3000.0)
empty_weight = st.sidebar.number_input("Empty weight lbs", value=2000.0)
range_nm = st.sidebar.number_input("Range N.M.", value=500.0)

# Construction du dictionnaire
# IMPORTANT: Les clés doivent correspondre EXACTEMENT aux noms de colonnes du dataset
input_data = {
    "Engine Type": engine_type,
    "HP or lbs thr ea engine": hp,
    "Stall Knots dirty": stall_knots,
    "Fuel gal/lbs": fuel,
    "All eng service ceiling": ceiling,
    "All eng rate of climb": rate_of_climb,
    "Gross weight lbs": gross_weight,
    "Empty weight lbs": empty_weight,
    "Range N.M.": range_nm
}

# Bouton de prédiction
if st.sidebar.button("Prédire la vitesse"):
    # URL vers le service "api" du docker-compose
    API_URL = "http://api:8000/predict"

    try:
        response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()

            # DEBUG : Afficher ce que l'API renvoie
            st.write("Réponse brute de l'API :", result)

            if 'prediction' in result:
                st.success(f"La vitesse prédite est : **{result['prediction']:.2f} Knots**")
            else:
                st.error(f"L'API a répondu 200 OK, mais il manque la prédiction. Détails : {result}")
        else:
            st.error(f"Erreur {response.status_code} : {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Impossible de connecter l'API. Vérifiez les logs avec 'docker-compose logs -f mlops_api'")

st.markdown("---")
st.write("Entraîné avec MLflow | Architecture MLOps")