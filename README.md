# *****✈️ Aircraft Speed Prediction: End-to-End MLOps Pipeline*****

This repository contains a complete MLOps infrastructure designed for rigor and transparency. Unlike basic approaches, this pipeline integrates an industrial "Model Gate" that ensures only generalizable models—and not those that overfit—reach production.

## 🛠️ 1. Technical Architecture

The project is containerized via Docker Compose to guarantee reproducibility and portability. It consists of 4 isolated microservices:
*   **`mlflow`**: Tracking server for metrics, parameters, and artifact registry (`.pkl` models).
*   **`trainer`**: Isolated training and experimentation environment (via `src/train.py`).
*   **`api`**: REST API (FastAPI) for low-latency inference, integrating a strict "Model Gate" logic for dynamic champion model loading.
*   **`frontend`**: Interactive user interface built with Streamlit for simulation and prediction.

---

## 🚀 2. Quick Start

**1. Clone the project:**
```bash
git clone [https://github.com/Wuradclan/mlops-aircraft-speed-prediction.git](https://github.com/Wuradclan/mlops-aircraft-speed-prediction.git)
cd mlops-aircraft-speed-prediction
```

## 2. Run the infrastructure:

```Bash
docker compose up -d --build
```

## 🧠 3. Model Zoo: Supported Algorithms

The src/train.py script is highly modular and supports multiple algorithm families. 
You can trigger a manual training run via the trainer container by specifying the --model_type parameter.

### 🌲 Tree-based Models

* xgboost: The industry standard for performance on tabular data.
* random_forest: Robust and inherently less prone to overfitting.
* extra_trees: Extremely randomized trees to maximize generalization.

### 📈 Linear Models

* linear: Simple linear regression (Baseline model).
* ridge: Regression with L2 penalty (prevents coefficient explosion).
* lasso: Regression with L1 penalty (built-in feature selection).

### 📐 Distance & Neural Networks

* knn: K-Nearest Neighbors (based on spatial proximity).
* svr: Support Vector Regression (effective in high dimensions).
* mlp: Multi-Layer Perceptron (dense neural network).

### 🏗️ Ensembles & AutoML

* stacking: Hybrid model combining predictions from several base estimators.
* h2o: AutoML engine that automatically explores search spaces and builds an optimal Stacked Ensemble.

## 🔄 4. Training Methods (Lifecycle)

The pipeline uses src/train.py to orchestrate training using three distinct approaches, allowing us to compare the robustness of each method.

#### A. Automated Exploration (H2O AutoML)

AutoML is used to quickly explore complex search spaces and build Stacked Ensembles.
Command:
```Bash
docker compose exec trainer python src/train.py --model_type h2o
```

#### B. Advanced & Robust Optimization (Optuna)

For expert hyperparameter tuning on base models (XGBoost, Random Forest, etc.). The optimizer minimizes the RMSE while applying a strict penalty for overfitting to force the creation of stable models.
Command:
```Bash
docker compose exec trainer python -B src/train.py --model_type xgboost --tune --n_trials 50
```

#### C. Baseline Training (Manual)

To quickly train, evaluate, and version a specific model with fixed parameters:
Command:
```Bash
docker compose exec trainer python src/train.py --model_type xgboost --n_estimators 500 --max_depth 5
```

## ⚖️ 5. Model Gate: Industrial Champion Selection (API)

The API does not blindly accept the model with the best paper score. At startup (or via /reload-model), it queries MLflow and applies a strict Model Gate to prevent data leakage and memorization:
Overfit Calculation:
`Overfit = (RMSE_Test - RMSE_Train) / RMSE_Test`
* Anti-Cheat Filtering: Any model exceeding a strict 30% overfit threshold is instantly eliminated, even if its RMSE seems excellent.
* Dynamic Selection: Among the validated and stable models, the API automatically loads the one with the lowest RMSE_Test.
* This approach penalizes models that merely memorize data (overfitting) in favor of models capable of generalizing in real-world conditions.

## 🔌 6. Inference API & Monitoring

Swagger UI (Interactive Docs): http://localhost:8000/docs
Inference: POST /predict
Model Status: GET /model-info (Displays the loaded algorithm and its validation metrics).
Hot-Reload: POST /reload-model (Forces the API to query MLflow and seamlessly switch to a newly detected champion without downtime).

## 📊 7. MLOps Tracking (MLflow)

MLflow Dashboard: http://localhost:5050
Traceability: Each training generates a Run with its parameters, Train/CV/Test metrics, and physical artifact (.pkl or H2O format). Optuna studies generate a clean "Parent Run" to group all trials.

## 🖥️ 8. User Interface (Streamlit)

Frontend Dashboard: http://localhost:8501
Allows sending aircraft features to the API via an intuitive interface to visualize predictions in real-time.

## 🧹 9. Maintenance & Cleanup

To gracefully stop all services:
```Bash
docker compose down
````
To perform a complete purge (removes containers, networks, and wipes MLflow databases/volumes):
```Bash
docker compose down -v
````
Note: Virtual environment folders (venv), IDE files (.idea), and local binary models (.pkl) are excluded from versioning via .gitignore to keep the repository lightweight.

#######################################################################################################
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