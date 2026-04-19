# Review Guardian App

## 🚀 Lancer l'application

### Préparation

```bash
# Depuis la racine du projet
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Installer les dépendances du projet
pip install -r requirements.txt

# (Optionnel) Appliquer un profil d'environnement
.\switch_env.ps1 -Profile dev -Force
```

### Option 1: Streamlit (Interface Web Interactive)

```bash
# Lancer Streamlit
streamlit run app/streamlit_app.py
```

L'application sera accessible sur: http://localhost:8501

### Option 2: FastAPI (API REST)

```bash
# Lancer l'API
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

L'API sera accessible sur: http://localhost:8000
Documentation Swagger: http://localhost:8000/docs

## 📡 Exemples d'utilisation de l'API

### Analyser un avis

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Super restaurant!", "rating": 5}'
```

### Mode Chat

```bash
curl -X POST "http://localhost:8000/chat?message=Excellent%20service"
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={"text": "Super restaurant, je recommande!", "rating": 5}
)
print(response.json())
```

## 📦 Dépendances

```bash
pip install -r requirements.txt
```
