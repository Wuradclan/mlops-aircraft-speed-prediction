# Dockerfile situé dans le dossier 'train/'
FROM python:3.11-slim

# Installation des dépendances système pour XGBoost
RUN apt-get update && apt-get install -y libomp-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# On copie les dépendances
COPY requirements.txt .
RUN pip install -r requirements.txt

# On copie le code source (le dossier src est au niveau supérieur)
COPY src/ ./src/
# On copie aussi les données si besoin
COPY data/ ./data/

# La commande cruciale : on pointe vers src/train.py
CMD ["python", "src/train.py"]