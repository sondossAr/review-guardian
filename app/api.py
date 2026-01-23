"""
🛡️ Review Guardian - API FastAPI
API REST pour la détection de faux avis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import re
from textblob import TextBlob

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Review Guardian API",
    description="API pour la détection de faux avis",
    version="1.0.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le modèle
model = None
scaler = None
feature_cols = None


class ReviewInput(BaseModel):
    """Modèle d'entrée pour l'analyse d'avis"""
    text: str = Field(..., min_length=1, description="Texte de l'avis à analyser")
    rating: int = Field(default=5, ge=1, le=5, description="Note (1-5)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Super restaurant ! La nourriture était excellente.",
                "rating": 5
            }
        }


class ReviewOutput(BaseModel):
    """Modèle de sortie pour l'analyse d'avis"""
    is_fake: bool
    confidence: float
    probability_fake: float
    probability_real: float
    verdict: str
    features: dict
    warnings: List[str]


class BatchInput(BaseModel):
    """Modèle d'entrée pour l'analyse par lot"""
    reviews: List[ReviewInput]


class HealthOutput(BaseModel):
    """Sortie du bilan de santé"""
    status: str
    model_loaded: bool
    features_count: int


def extract_features(text: str, rating: int = 5) -> dict:
    """Extraction des caractéristiques du texte d'avis"""
    
    # Statistiques du texte
    word_count = len(text.split())
    char_count = len(text)
    sentence_count = max(1, len(re.split(r'[.!?]+', text)))
    avg_word_length = char_count / max(1, word_count)
    
    # Comptage de ponctuation
    exclamation_count = text.count('!')
    question_count = text.count('?')
    
    # Ratio de majuscules
    uppercase_count = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_count / max(1, len(text))
    
    # Diversité lexicale
    words = text.lower().split()
    unique_words = set(words)
    lexical_diversity = len(unique_words) / max(1, len(words))
    
    # Analyse de sentiment
    try:
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
    except:
        sentiment_polarity = 0
        sentiment_subjectivity = 0.5
    
    # Décalage note/sentiment
    if rating >= 4:
        expected_sentiment = 0.3
    elif rating <= 2:
        expected_sentiment = -0.3
    else:
        expected_sentiment = 0
    
    rating_sentiment_mismatch = abs(sentiment_polarity - expected_sentiment)
    
    # Ratio de chiffres
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / max(1, len(text))
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'uppercase_ratio': uppercase_ratio,
        'lexical_diversity': lexical_diversity,
        'sentiment_polarity': sentiment_polarity,
        'sentiment_subjectivity': sentiment_subjectivity,
        'rating_sentiment_mismatch': rating_sentiment_mismatch,
        'digit_ratio': digit_ratio,
        'rating': rating,
    }


def get_warnings(features: dict) -> List[str]:
    """Génération des messages d'alerte basés sur les caractéristiques"""
    warnings = []
    
    if features.get('exclamation_count', 0) > 3:
        warnings.append("Beaucoup de points d'exclamation")
    if features.get('uppercase_ratio', 0) > 0.3:
        warnings.append("Beaucoup de majuscules")
    if features.get('word_count', 0) < 10:
        warnings.append("Avis très court")
    if features.get('lexical_diversity', 0) < 0.5:
        warnings.append("Faible diversité du vocabulaire")
    if features.get('rating_sentiment_mismatch', 0) > 0.5:
        warnings.append("Décalage entre note et sentiment")
    
    return warnings


@app.on_event("startup")
async def load_model():
    """Chargement du modèle au démarrage"""
    global model, scaler, feature_cols
    
    models_dir = Path(__file__).parent.parent / 'models'
    
    try:
        model = joblib.load(models_dir / 'best_rf_model.joblib')
        scaler = joblib.load(models_dir / 'scaler.joblib')
        feature_cols = joblib.load(models_dir / 'feature_columns.joblib')
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement du modèle : {e}")
        print("Assurez-vous d'avoir exécuté les notebooks d'entraînement.")


@app.get("/", tags=["General"])
async def root():
    """Point d'entrée d'accueil"""
    return {
        "message": "🛡️ Welcome to Review Guardian API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthOutput, tags=["General"])
async def health_check():
    """Vérification de l'état de l'API et du modèle"""
    return HealthOutput(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        features_count=len(feature_cols) if feature_cols else 0
    )


@app.post("/analyze", response_model=ReviewOutput, tags=["Analysis"])
async def analyze_review(review: ReviewInput):
    """
    Analyser un avis pour déterminer son authenticité.
    
    Retourne la prédiction avec les scores de confiance et les alertes détectées.
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé. Exécutez d'abord les notebooks d'entraînement."
        )
    
    # Extraction des caractéristiques
    features = extract_features(review.text, review.rating)
    
    # Création du dataframe
    df = pd.DataFrame([features])
    
    # Ajout des caractéristiques manquantes
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Sélection et préparation des features
    X = df[feature_cols].copy().fillna(0)
    
    # Prédiction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    
    # Génération des alertes
    warnings = get_warnings(features)
    
    # Détermination du verdict
    if prediction == 1:
        verdict = "Avis authentique"
        confidence = probability[1]
    else:
        verdict = "Avis suspect"
        confidence = probability[0]
    
    return ReviewOutput(
        is_fake=bool(prediction == 0),
        confidence=float(confidence),
        probability_fake=float(probability[0]),
        probability_real=float(probability[1]),
        verdict=verdict,
        features=features,
        warnings=warnings
    )


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(batch: BatchInput):
    """
    Analyser plusieurs avis à la fois.
    
    Retourne une liste de prédictions pour chaque avis.
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé. Exécutez d'abord les notebooks d'entraînement."
        )
    
    results = []
    for review in batch.reviews:
        result = await analyze_review(review)
        results.append(result)
    
    # Statistiques résumées
    fake_count = sum(1 for r in results if r.is_fake)
    real_count = len(results) - fake_count
    
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "fake": fake_count,
            "real": real_count,
            "fake_percentage": fake_count / len(results) * 100 if results else 0
        }
    }


@app.post("/chat", tags=["Chat"])
async def chat_analyze(message: str):
    """
    Point d'entrée style chat pour une analyse conversationnelle.
    
    Envoyez un texte d'avis et recevez une réponse en langage naturel.
    """
    if model is None:
        return {
            "response": "⚠️ Je ne suis pas encore prêt. Veuillez charger le modèle d'abord."
        }
    
    if len(message.strip()) < 5:
        return {
            "response": "⚠️ Le texte est trop court. Envoyez-moi un avis plus détaillé à analyser."
        }
    
    # Analyse
    review = ReviewInput(text=message, rating=5)
    result = await analyze_review(review)
    
    # Génération de la réponse naturelle
    if result.is_fake:
        response = f"""🚨 **Avis suspect détecté!**

J'ai analysé cet avis et il présente des caractéristiques suspectes.

**Probabilité de faux:** {result.probability_fake*100:.1f}%

**Signaux d'alerte:**
{chr(10).join('• ' + w for w in result.warnings) if result.warnings else '• Patterns inhabituels détectés'}

Je recommande une vérification manuelle de cet avis."""
    else:
        response = f"""✅ **Avis authentique**

Cet avis semble légitime.

**Confiance:** {result.probability_real*100:.1f}%

**Caractéristiques:**
• Mots: {result.features['word_count']}
• Sentiment: {result.features['sentiment_polarity']:.2f}
• Diversité lexicale: {result.features['lexical_diversity']:.2f}

L'avis présente des patterns typiques d'avis authentiques."""
    
    return {"response": response, "analysis": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
