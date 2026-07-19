# *****✈️ Aircraft Speed Prediction: End-to-End MLOps Pipeline*****

    Ce dépôt contient une infrastructure MLOps complète, conçue pour la rigueur et la transparence.
    Contrairement aux approches basiques, ce pipeline intègre une "Model Gate" industrielle qui garantit que seuls les modèles généralisables — et non ceux qui surapprennent — atteignent la production.

## 🛠️ 1. Architecture Technique:

    Le projet est conteneurisé via Docker Compose pour garantir la reproductibilité. Il se compose de 4 microservices :
    - **mlflow** : Serveur de suivi pour les métriques, paramètres et artéfacts.
    - **trainer** : Environnement isolé pour l'entraînement (Scikit-Learn, Optuna, H2O).
    - **api** : API REST (FastAPI) intégrant une logique de filtrage stricte contre l'overfitting.
    - **frontend** : Interface interactive (Streamlit) pour la simulation.

## 🚀 2. Démarrage Rapide:

### Cloner le projet :

```Bash
    git clone https://github.com/Wuradclan/mlops-aircraft-speed-prediction.git
    cd mlops-aircraft-speed-prediction
```
```Bash
#Lancer l'infrastructure :
docker-compose up -d --build
```

## 🧠 3. Model Zoo : Algorithmes Supportés

    Le script src/train.py est polyvalent et supporte les familles de modèles suivantes. Tu peux les entraîner manuellement avec la commande :

    docker-compose exec trainer python src/train.py --model_type [TYPE]

#### 🌲 Modèles à base d'arbres:

    - xgboost : Le standard pour la performance sur données tabulaires.
    - random_forest : Robuste et moins sensible au surapprentissage.
    - extra_trees : Variantes extrêmement aléatoires pour plus de généralisation.

#### 📈 Modèles Linéaires

    - linear : Régression linéaire simple (Baseline).
    - ridge : Régression avec pénalité L2 (évite l'explosion des coefficients).
    - lasso : Régression avec pénalité L1 (sélection de features).

#### 📐 Distance & Réseaux de Neurones:

    - knn : K-Nearest Neighbors (basé sur la proximité spatiale).
    - svr : Support Vector Regression (efficace en haute dimension).
    - mlp : Multi-Layer Perceptron (réseau de neurones simple).

#### 🏗️ Ensembles & AutoML:

    - stacking : Modèle hybride combinant plusieurs prédicteurs.
    - h2o : Le Champion. AutoML qui explore automatiquement les espaces de recherche.

## 🧠 4. Méthodes d'Entraînement (Cycle de vie)

    Le pipeline utilise `src/train.py` pour orchestrer les entraînements selon trois approches distinctes,
    permettant de comparer la robustesse de chaque méthode.

#### A. Exploration Automatique (H2O AutoML)
    L'AutoML est utilisé pour explorer rapidement les espaces de recherche complexes et construire des Stacked Ensembles.
```Bash
docker compose exec trainer python src/train.py --model_type h2o`
````

#### B. Optimisation automatique avec Optuna

    Pour la recherche experte d'hyperparamètres sur les modèles de base (XGBoost, Random Forest, etc.).
    L'optimiseur minimise le RMSE tout en appliquant une pénalité stricte en cas de surapprentissage pour forcer la création de modèles stables.
    Commande pour lancer une optimisation :

```Bash
docker compose exec trainer python -B src/train.py --model_type stacking --tune --n_trials 50

docker compose exec trainer python -B src/train.py --model_type xgboost --tune --n_trials 25
```

#### C. Entraînement Baseline (Manuel):

    Pour entraîner et versionner un modèle spécifique avec des paramètres fixes :ex: XGBoost) :

```Bash
docker-compose exec trainer python src/train.py --model_type xgboost --n_estimators 500 --max_depth 5
```

## ⚖️ 5. Model Gate : Sélection Industrielle du Champion (API)

    L'API n'accepte pas aveuglément le modèle avec le meilleur score sur le papier. Au démarrage ou via `/reload-model`,
    elle interroge MLflow et applique une **Douane (Model Gate)** stricte :

    1. **Calcul du Surapprentissage :** 
   `Overfit = (RMSE_Test - RMSE_Train) / RMSE_Test`
    2. **Filtrage Anti-Triche :** Tout modèle dépassant un seuil de **30% d'overfit** est instantanément éliminé.
    3. **Sélection Dynamique :** Parmi les modèles validés et stables, l'API charge celui possédant le meilleur `RMSE_Test`.

    Cette approche pénalise les modèles qui mémorisent les données (overfitting) au profit de modèles capables de généraliser en conditions réelles.

## 🔌 6. API d'Inférence et Monitoring:

    Swagger Docs : http://localhost:8000/docs
    Prédiction : POST /predict
    Rechargement dynamique : POST /reload-model (bascule automatiquement sur le nouveau champion détecté).

## 📊 7. Suivi des Expérimentations (MLflow)

    Dashboard : http://localhost:5050
    Auto-discovery : Chaque run Optuna crée un "Parent Run" regroupant tous les "Nested Runs" (trials).

## 🖥️ 8. Interface Utilisateur (Streamlit)

    Accès : http://localhost:8501

## 🧹 9. Maintenance

    Arrêt des services :
    ```Bash
    docker-compose down
    ```
    Pour purger complètement l'environnement (incluant les volumes persistants) :
    ```Bash
    docker-compose down -v
    ```
    Note : Les fichiers IDE (.idea) et les modèles binaires (.pkl) sont exclus du versioning via .gitignore.