***✈️ Aircraft Speed Prediction: End-to-End MLOps Pipeline***
Ce dépôt contient une infrastructure MLOps complète et prête pour la production, dédiée à la prédiction de la vitesse des avions commerciaux.
Suite à une phase de benchmarking extensive (comparant Scikit-Learn, XGBoost, un Stacking Regressor personnalisé et l'AutoML), le modèle Champion sélectionné pour la production est propulsé par H2O AutoML (Stacked Ensemble), offrant des performances de pointe avec un RMSE de ~5.30.

🛠️ 1. Architecture Technique
Le projet est conteneurisé via Docker Compose pour garantir la reproductibilité. Il se compose de 4 microservices :
mlflow : Serveur de suivi pour les métriques, paramètres et artéfacts.
trainer : Environnement isolé pour l'entraînement et l'expérimentation (via src/train_h2o.py).
api : API REST (FastAPI) pour l'inférence en basse latence avec chargement dynamique du meilleur modèle.
frontend : Interface interactive (Streamlit) pour la simulation.

🚀 2. Démarrage Rapide
Cloner le projet :
```Bash
git clone https://github.com/Wuradclan/mlops-aircraft-speed-prediction.git
cd mlops-aircraft-speed-prediction

#Lancer l'infrastructure :

docker-compose up -d --build
```
🧠 3. Model Zoo : Algorithmes Supportés
Le script src/train_h2o.py est polyvalent et supporte les familles de modèles suivantes. Tu peux les entraîner manuellement avec la commande :
docker-compose exec trainer python src/train_h2o.py --model_type [TYPE]

🌲 Modèles à base d'arbres
xgboost : Le standard pour la performance sur données tabulaires.
random_forest : Robuste et moins sensible au surapprentissage.
extra_trees : Variantes extrêmement aléatoires pour plus de généralisation.

📈 Modèles Linéaires
linear : Régression linéaire simple (Baseline).
ridge : Régression avec pénalité L2 (évite l'explosion des coefficients).
lasso : Régression avec pénalité L1 (sélection de features).

📐 Distance & Réseaux de Neurones
knn : K-Nearest Neighbors (basé sur la proximité spatiale).
svr : Support Vector Regression (efficace en haute dimension).
mlp : Multi-Layer Perceptron (réseau de neurones simple).

🏗️ Ensembles & AutoML
stacking : Modèle hybride combinant plusieurs prédicteurs.
h2o : Le Champion. AutoML qui explore automatiquement les espaces de recherche.

🧠 4. Cycle de vie des modèles
A. Le Champion : H2O AutoML
Le pipeline utilise src/train_h2o.py pour orchestrer les entraînements. H2O AutoML est utilisé pour explorer automatiquement les espaces de recherche complexes, effectuer le feature scaling et construire un Stacked Ensemble optimisé.
B. Optimisation automatique avec Optuna
En complément, nous utilisons Optuna pour l'optimisation fine des hyperparamètres sur les modèles de base (XGBoost, Random Forest, etc.).
Commande pour lancer une optimisation :
```Bash
docker compose exec trainer python -B src/train_h2o.py --model_type stacking --tune --n_trials 50
```

C. Entraînement Baseline (Manuel)
Pour entraîner un modèle spécifique (ex: XGBoost) :

```Bash
docker-compose exec trainer python src/train_h2o.py --model_type xgboost --n_estimators 500 --max_depth 5
```

⚖️ 5. Sélection intelligente du "Modèle Champion" (API)
L'API utilise une logique de sélection basée sur la robustesse pour éviter le surapprentissage. Au démarrage ou via /reload-model, elle interroge MLflow et calcule un Score de Robustesse :
Score=RMSE_Test+(0.5×∣RMSE_Train−RMSE_Test∣)
Cette approche pénalise les modèles qui "trichent" (overfitting) au profit de modèles généralisables.

🔌 6. API d'Inférence et Monitoring
Swagger Docs : http://localhost:8000/docs
Prédiction : POST /predict
Rechargement dynamique : POST /reload-model (bascule automatiquement sur le nouveau champion détecté).

📊 7. Suivi des Expérimentations (MLflow)
Dashboard : http://localhost:5050
Auto-discovery : Chaque run Optuna crée un "Parent Run" regroupant tous les "Nested Runs" (trials).

🖥️ 8. Interface Utilisateur (Streamlit)
Accès : http://localhost:8501

🧹 9. Maintenance
Arrêt des services :
```Bash
docker-compose down
```
Pour purger complètement l'environnement (incluant les volumes persistants) :
```Bash
docker-compose down -v
```
Note : Les fichiers IDE (.idea) et les modèles binaires (.pkl) sont exclus du versioning via .gitignore.