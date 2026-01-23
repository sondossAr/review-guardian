# Review Guardian App

## 🚀 Lancer l'application

### Option 1: Streamlit (Interface Web Interactive)

```bash
# Activer l'environnement virtuel
.\venv\Scripts\Activate.ps1

# Lancer Streamlit
streamlit run app/streamlit_app.py
```

L'application sera accessible sur: http://localhost:8501

### Option 2: FastAPI (API REST)

```bash
# Activer l'environnement virtuel
.\venv\Scripts\Activate.ps1

# Installer FastAPI et Uvicorn
pip install fastapi uvicorn

# Lancer l'API
uvicorn app.api:app --reload
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
pip install streamlit fastapi uvicorn textblob
```
