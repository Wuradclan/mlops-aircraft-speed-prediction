✈️ MLOps Airplane Speed Predictor
Ce projet implémente une pipeline MLOps complète pour prédire la vitesse maximale d'un avion à partir de ses caractéristiques techniques. L'architecture est entièrement conteneurisée pour garantir la reproductibilité et la portabilité.
🏗️ Architecture Technique
Le système est composé de quatre services orchestrés par Docker Compose :
Trainer (trainer) : Pipeline d'entraînement automatique avec nettoyage de données et versioning.
Tracking (mlflow) : Serveur MLflow 3.14 pour le suivi des expériences et des métriques.
API (api) : Service FastAPI exposant le modèle entraîné pour l'inférence.
Frontend (streamlit) : Interface utilisateur interactive pour réaliser des prédictions en temps réel.
🚀 Démarrage Rapide
Assurez-vous d'avoir Docker et Docker Compose installés sur votre machine.
Clonez le dépôt (ou accédez au dossier du projet) :
Bash
cd chemin/vers/votre/projet
Lancez l'infrastructure complète :
Bash
docker-compose up --build
🌐 Accès aux services
Une fois les conteneurs lancés, accédez aux composants via votre navigateur :
Interface Streamlit (Prédiction) : http://localhost:8501
MLflow UI (Suivi des modèles) : http://localhost:5050
FastAPI Docs (API documentation) : http://localhost:8000/docs
🛠️ Choix Techniques & MLOps
Gestion de la donnée : Exclusion de la colonne Company pour éviter le surapprentissage (haute cardinalité sur un dataset restreint).
Reproductibilité : Utilisation de conteneurs Docker pour figer l'environnement Python et ses dépendances.
Versioning : Suivi automatisé des paramètres (hyperparamètres, learning rate, profondeur max) et des métriques (RMSE, R²) via MLflow.
Sécurité : Configuration des middlewares natifs de MLflow 3.x pour gérer la communication inter-conteneurs (DNS Rebinding protection).
📊 Entraînement
Si vous souhaitez déclencher un nouvel entraînement manuellement, exécutez la commande suivante :
Bash
docker-compose run --rm trainer
Projet réalisé pour le cours de [Nom du cours].