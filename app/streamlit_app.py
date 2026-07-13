import streamlit as st
import requests

# Configuration de la page
st.set_page_config(page_title="Prédiction de Vitesse d'Avion", page_icon="✈️")

# ==========================================
# 1. SECTION ACCUEIL (Toujours visible)
# ==========================================
st.title("✈️ Prédiction de Vitesse d'Avion")

# Mise en valeur de l'équipe avec un bloc d'information
st.info("""
**👨‍💻 Équipe du projet MLOps :** Mohamed Houari | Peter El-Hadad | Jaime Alfonso Robledo Villacob | Morad Ait Abdellah
""")

st.markdown(
    "Ce modèle a été entraîné et optimisé via MLflow (colonne 'Company' exclue pour éviter le surapprentissage).")
st.markdown("---")

# ==========================================
# 2. BARRE LATÉRALE (Paramètres)
# ==========================================
st.sidebar.header("Paramètres de l'avion")

engine_mapping = {
    "Piston": "Piston",
    "Turboprop": "Turbopropulseur",
    "Jet": "Réacteur (Jet)",
    "Unknown": "Inconnu"
}
engine_type = st.sidebar.selectbox(
    "Type de moteur",
    options=list(engine_mapping.keys()),
    format_func=lambda x: engine_mapping[x]
)

hp = st.sidebar.slider("Puissance / Poussée (Ch/lbs)", min_value=50.0, max_value=5000.0, value=300.0, step=10.0)
stall_knots = st.sidebar.slider("Vitesse de décrochage (Nœuds)", min_value=30.0, max_value=150.0, value=50.0, step=1.0)
fuel = st.sidebar.slider("Capacité carburant (Gal/lbs)", min_value=10.0, max_value=5000.0, value=100.0, step=10.0)
ceiling = st.sidebar.slider("Plafond pratique (Pieds)", min_value=5000.0, max_value=50000.0, value=15000.0, step=500.0)
rate_of_climb = st.sidebar.slider("Taux de montée (Pieds/min)", min_value=200.0, max_value=6000.0, value=1000.0,
                                  step=50.0)
gross_weight = st.sidebar.slider("Poids brut (lbs)", min_value=500.0, max_value=50000.0, value=3000.0, step=100.0)
empty_weight = st.sidebar.slider("Poids à vide (lbs)", min_value=300.0, max_value=30000.0, value=2000.0, step=100.0)
range_nm = st.sidebar.slider("Autonomie (NM)", min_value=100.0, max_value=10000.0, value=500.0, step=50.0)

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

# ==========================================
# 3. SECTION RÉSULTAT (Apparaît après le clic)
# ==========================================
if st.sidebar.button("Prédire la vitesse"):
    API_URL = "http://api:8000/predict"

    try:
        # On affiche un petit message de chargement sympa
        with st.spinner("L'API FastAPI calcule la prédiction..."):
            response = requests.post(API_URL, json=input_data)

        if response.status_code == 200:
            result = response.json()

            if 'prediction' in result:
                # Affichage du résultat principal
                st.success(f"🎯 La vitesse maximale estimée est de : **{result['prediction']:.2f} Nœuds**")

                # Signature de l'équipe sous le résultat
                st.caption("✨ *Prédiction générée avec succès par le modèle déployé par l'équipe.*")

                with st.expander("Voir la réponse brute de l'API"):
                    st.json(result)
            else:
                st.error(f"L'API a répondu correctement, mais la prédiction est manquante. Détails : {result}")
        else:
            st.error(f"Erreur de l'API ({response.status_code}) : {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Impossible de se connecter à l'API. Assurez-vous que le conteneur FastAPI est en cours d'exécution.")

# Pied de page global
st.markdown("---")
st.write("🛠️ Pipeline MLOps : Docker | MLflow | FastAPI | Streamlit")