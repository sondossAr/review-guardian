"""
🛡️ Review Guardian - Chatbot de Détection de Faux Avis
Application Streamlit pour la détection de faux avis
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import re
from transformers import pipeline
import torch
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")


def _to_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_path(raw_path: str, fallback: Path) -> Path:
    if not raw_path:
        return fallback
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


APP_THREADS = max(1, _to_int(os.getenv("APP_THREADS", str(os.cpu_count() or 1)), os.cpu_count() or 1))
MODELS_DIR = _resolve_path(os.getenv("MODEL_DIR", "models"), BASE_DIR / "models")
HF_CACHE_DIR = _resolve_path(os.getenv("HF_CACHE_DIR", "models/hf_cache"), BASE_DIR / "models" / "hf_cache")
STREAMLIT_MODEL_PRIMARY = os.getenv("STREAMLIT_MODEL_PRIMARY", "best_gb_model.joblib")
STREAMLIT_MODEL_SECONDARY = os.getenv("STREAMLIT_MODEL_SECONDARY", "best_model.joblib")
STREAMLIT_MODEL_FALLBACK = os.getenv("STREAMLIT_MODEL_FALLBACK", "best_rf_model.joblib")
SCALER_FILE = os.getenv("SCALER_FILE", "scaler.joblib")
FEATURE_COLUMNS_FILE = os.getenv("FEATURE_COLUMNS_FILE", "feature_columns.joblib")
SENTIMENT_MODEL_NAME = os.getenv("SENTIMENT_MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment")

# Optimisation CPU Intel - utiliser tous les coeurs
os.environ["OMP_NUM_THREADS"] = str(APP_THREADS)
os.environ["MKL_NUM_THREADS"] = str(APP_THREADS)
torch.set_num_threads(APP_THREADS)

# =============================================================================
# DETECTION DES PATTERNS MARKETING
# =============================================================================

SUPERLATIVES = [
    r"\ble meilleur\b", r"\bla meilleure\b", r"\bles meilleurs\b",
    r"\ble plus\b", r"\bla plus\b", r"\bles plus\b",
    r"\bincroyable\b", r"\bexceptionnel\b", r"\bextraordinaire\b",
    r"\bparfait\b", r"\bparfaite\b", r"\bparfaitement\b",
    r"\bfabuleux\b", r"\bmagnifique\b", r"\bmerveilleux\b",
    r"\bépoustouflant\b", r"\brévolutionnaire\b", r"\bunique\b",
    r"\binégalé\b", r"\bimbattable\b", r"\bsans égal\b",
    r"\bjamais vu\b", r"\bde tous les temps\b", r"\ble top\b",
    r"\bnuméro 1\b", r"\bn°1\b", r"\b#1\b",
    r"\bsensationnel\b", r"\bphénoménal\b", r"\bexquis\b",
    r"\bsublime\b", r"\bformidable\b", r"\bà couper le souffle\b",
    r"\bthe best\b", r"\bbest ever\b", r"\bamazing\b",
    r"\bincredible\b", r"\bunbelievable\b", r"\bperfect\b",
    r"\bphenomenal\b", r"\boutstanding\b", r"\bexceptional\b",
    r"\bmust have\b", r"\bgame changer\b", r"\blife changing\b",
]

CALL_TO_ACTION = [
    r"\bachetez\b", r"\bcommandez\b", r"\bréservez\b",
    r"\bessayez\b", r"\bdécouvrez\b", r"\bprofitez\b",
    r"\bn'hésitez pas\b", r"\bfoncez\b", r"\bcourez-y\b",
    r"\bvenez\b", r"\bdépêchez-vous\b", r"\bvite\b",
    r"\bje recommande\b", r"\bje vous recommande\b",
    r"\bà essayer absolument\b", r"\bincontournable\b",
    r"\boffre limitée\b", r"\bpromotion\b", r"\bréduction\b",
    r"\bcode promo\b", r"\b-\d+%\b",
    r"\bbuy now\b", r"\border now\b", r"\bget it now\b",
    r"\bdon't miss\b", r"\bhurry\b", r"\blimited time\b",
    r"\bact now\b", r"\bcall now\b", r"\bclick here\b",
]

ADVERTISING_TONE = [
    r"\bsatisfait ou remboursé\b", r"\bgaranti\b",
    r"\bmeilleur rapport qualité[\s-]prix\b",
    r"\bprix imbattable\b", r"\bau meilleur prix\b",
    r"\blivraison gratuite\b", r"\bfrais de port offerts\b",
    r"\bexclusif\b", r"\bédition limitée\b",
    r"\bnouveauté\b", r"\btout nouveau\b",
    r"\bqualité premium\b", r"\bhaut de gamme\b",
    r"\bprofessionnel\b", r"\bcertifié\b",
    r"\b100%\b", r"\b5 étoiles\b", r"\b5\s*\*\b",
    r"www\.", r"http", r"\.com\b", r"\.fr\b",
    r"@\w+", r"\d{2}[\s.-]\d{2}[\s.-]\d{2}",
]

TOO_PERFECT_INDICATORS = [
    r"\baucun défaut\b", r"\baucun problème\b",
    r"\bsans aucun\b", r"\bpas le moindre\b",
    r"\brien à redire\b", r"\brien à signaler\b",
    r"\btout est parfait\b", r"\btout était parfait\b",
    r"\bimpeccable\b", r"\birréprochable\b",
    r"\btoujours\b", r"\bjamais eu\b",
    r"\bchaque fois\b", r"\bà chaque fois\b",
    r"\bsans exception\b", r"\bsans faute\b",
    r"\bexcellent[!]+", r"\bparfait[!]+", r"\bsuper[!]+",
]

EMOTIONAL_EXAGGERATION = [
    r"\btrop\b", r"\btrès très\b", r"\bvraiment très\b",
    r"\babsolument\b", r"\btotalement\b", r"\bcomplètement\b",
    r"\bénormément\b", r"\bextrêmement\b", r"\binfiniment\b",
    r"\bfollement\b", r"\bpassionnément\b",
    r"\bje suis fan\b", r"\bje suis addict\b", r"\bj'adore\b",
    r"\bfan absolu\b", r"\bcoup de coeur\b", r"\bcoup de foudre\b",
]

# =============================================================================
# LEXIQUE DE SENTIMENT FRANCAIS (TextBlob ne fonctionne pas pour le français !)
# =============================================================================

POSITIVE_WORDS_FR = [
    # Très positif (+2)
    "excellent", "exceptionnel", "extraordinaire", "fantastique", "formidable",
    "génial", "magnifique", "merveilleux", "parfait", "sublime", "superbe",
    "incroyable", "fabuleux", "phénoménal", "remarquable", "sensationnel",
    # Positif (+1)
    "bon", "bonne", "bons", "bonnes", "bien", "super", "top", "chouette",
    "agréable", "sympa", "sympathique", "cool", "joli", "jolie", "beau", "belle",
    "satisfait", "satisfaite", "content", "contente", "heureux", "heureuse",
    "ravi", "ravie", "enchanté", "enchantée", "impressionné", "impressionnée",
    "recommande", "recommander", "adore", "adoré", "aime", "aimé", "apprécie",
    "apprécié", "plaisir", "qualité", "efficace", "rapide", "propre", "frais",
    "délicieux", "savoureux", "succulent", "accueillant", "chaleureux",
    "professionnel", "compétent", "attentif", "souriant", "aimable", "gentil",
    "helpful", "great", "good", "nice", "amazing", "wonderful", "love", "loved",
    "best", "perfect", "excellent", "awesome", "fantastic", "brilliant",
    # Ajouts - qualités produit/service
    "facile", "simple", "pratique", "intuitif", "intuitive", "ergonomique",
    "fiable", "solide", "robuste", "stable", "performant", "performante",
    "utile", "fonctionnel", "fonctionnelle", "commode", "accessible",
    "fluide", "réactif", "réactive", "rapide", "rapidement", "instantané",
    "moderne", "innovant", "innovante", "intelligent", "intelligente",
    "confortable", "léger", "légère", "compact", "compacte", "élégant",
    "précis", "précise", "exact", "exacte", "fidèle", "optimal", "optimale",
    # Expressions positives
    "fonctionne bien", "marche bien", "ça marche", "nickel", "impec", "tip top",
    "au top", "parfaitement", "idéal", "idéale", "adapté", "adaptée",
]

NEGATIVE_WORDS_FR = [
    # Très négatif (-2)
    "horrible", "affreux", "atroce", "catastrophique", "désastreux", "épouvantable",
    "exécrable", "infect", "inmonde", "lamentable", "minable", "nul", "nulle",
    "pathétique", "pitoyable", "scandaleux", "terrible", "honteux",
    # Négatif (-1)
    "mauvais", "mauvaise", "mal", "décevant", "décevante", "déçu", "déçue",
    "déception", "médiocre", "moyen", "moyenne", "bof", "pas terrible",
    "cher", "chère", "trop cher", "arnaque", "voleur", "escroquerie",
    "sale", "dégoûtant", "froid", "tiède", "long", "lent", "lente",
    "attente", "bruyant", "désagréable", "impoli", "impolie", "agressif",
    "incompétent", "indifférent", "négligent", "absent", "fermé",
    "problème", "problèmes", "erreur", "oublié", "manque", "manquant",
    "dommage", "hélas", "malheureusement", "regret", "regrette",
    "éviter", "évitez", "fuyez", "jamais", "plus jamais", "dernière fois",
    "bad", "terrible", "horrible", "awful", "worst", "poor", "disappointed",
    "disappointing", "avoid", "never", "waste", "rude", "slow", "cold",
]

INTENSIFIERS_FR = [
    "très", "vraiment", "extrêmement", "absolument", "totalement", "complètement",
    "énormément", "incroyablement", "particulièrement", "super", "trop", "hyper",
    "ultra", "méga", "archi", "vachement", "carrément", "franchement",
    "really", "very", "extremely", "absolutely", "totally", "completely",
]

NEGATIONS_FR = [
    "ne", "n'", "pas", "plus", "jamais", "rien", "aucun", "aucune",
    "non", "ni", "sans", "not", "no", "never", "nothing", "none",
]


# =============================================================================
# ANALYSE DE SENTIMENT BASEE SUR BERT + TRADUCTION
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    """
    Charge le modèle de sentiment BERT multilingue.
    Le modèle supporte directement le polonais, français, anglais, etc.
    Utilise le cache local pour éviter les téléchargements répétés.
    Optimisé pour CPU Intel.
    """
    try:
        cache_dir = str(HF_CACHE_DIR) if HF_CACHE_DIR.exists() else None
        
        # Modèle de sentiment multilingue (supporte directement le polonais)
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL_NAME,
            device="cpu",
            torch_dtype=torch.float32,
            truncation=True,
            max_length=512,
            model_kwargs={"cache_dir": cache_dir, "low_cpu_mem_usage": True} if cache_dir else {"low_cpu_mem_usage": True}
        )
        
        return {"sentiment": sentiment_pipeline}, None
    except Exception as e:
        return None, str(e)


def analyze_sentiment_bert(text, sentiment_model):
    """
    Analyse du sentiment avec BERT multilingue.
    Supporte directement le polonais, français, anglais, etc.
    Retourne la polarité (-1 à 1) et la subjectivité (0 à 1)
    """
    if sentiment_model is None:
        return analyze_sentiment_lexicon(text)
    
    try:
        text_str = str(text)[:500]  # Limiter la longueur
        
        sentiment_pipeline = sentiment_model.get("sentiment")
        
        if sentiment_pipeline is None:
            return analyze_sentiment_lexicon(text)
        
        # Analyser le sentiment directement (BERT multilingue)
        result = sentiment_pipeline(text_str)[0]
        label = result['label']
        score = result['score']
        
        # Conversion étoiles → polarité (même formule que l'entraînement)
        # Label format: "1 star", "2 stars", ..., "5 stars"
        stars = int(label.split()[0])
        polarity = (stars - 3) / 2  # 1→-1, 3→0, 5→+1
        
        # Subjectivité = confiance du modèle
        subjectivity = score
        
        return polarity, subjectivity
        
    except Exception as e:
        return analyze_sentiment_lexicon(text)


def analyze_sentiment_lexicon(text):
    """
    Repli : Analyse de sentiment par approche lexicale.
    Retourne la polarité (-1 à 1) et la subjectivité (0 à 1)
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    if not words:
        return 0.0, 0.5
    
    positive_score = 0
    negative_score = 0
    sentiment_words = 0
    
    # Vérification du contexte de négation
    has_negation = any(neg in text_lower for neg in NEGATIONS_FR)
    
    # Vérification des intensificateurs
    has_intensifier = any(intens in text_lower for intens in INTENSIFIERS_FR)
    intensifier_multiplier = 1.5 if has_intensifier else 1.0
    
    # Comptage des mots positifs
    for word in POSITIVE_WORDS_FR:
        if word in text_lower:
            count = text_lower.count(word)
            if word in POSITIVE_WORDS_FR[:16]:
                positive_score += count * 2 * intensifier_multiplier
            else:
                positive_score += count * 1 * intensifier_multiplier
            sentiment_words += count
    
    # Comptage des mots négatifs
    for word in NEGATIVE_WORDS_FR:
        if word in text_lower:
            count = text_lower.count(word)
            if word in NEGATIVE_WORDS_FR[:18]:
                negative_score += count * 2 * intensifier_multiplier
            else:
                negative_score += count * 1 * intensifier_multiplier
            sentiment_words += count
    
    # Appliquer la négation
    if has_negation and sentiment_words > 0:
        positive_score, negative_score = negative_score * 0.5, positive_score * 0.5
    
    # Calcul de la polarité
    total_score = positive_score - negative_score
    max_possible = max(1, sentiment_words * 2)
    polarity = max(-1, min(1, total_score / max_possible))
    
    # Calcul de la subjectivité (0 à 1) - basée sur la densité de mots émotionnels/sentiment
    subjectivity = min(1.0, sentiment_words / max(1, len(words)) * 5)
    
    # Les points d'exclamation augmentent la subjectivité
    exclamation_count = text.count('!')
    subjectivity = min(1.0, subjectivity + exclamation_count * 0.1)
    
    return polarity, subjectivity


# Configuration de la page
st.set_page_config(
    page_title="Review Guardian",
    page_icon="🛡️",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fake-review {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
    }
    .real-review {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    .metric-card {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Charge le modèle entraîné et ses artefacts"""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    try:
        candidate_models = [
            (STREAMLIT_MODEL_PRIMARY, "Gradient Boosting"),
            (STREAMLIT_MODEL_SECONDARY, "Best Model"),
            (STREAMLIT_MODEL_FALLBACK, "Random Forest"),
        ]

        selected_model_file = None
        model_type = None
        for model_file, label in candidate_models:
            if (MODELS_DIR / model_file).exists():
                selected_model_file = model_file
                model_type = label
                break

        if selected_model_file is None:
            expected = ", ".join([m for m, _ in candidate_models])
            raise FileNotFoundError(
                f"Aucun modèle trouvé dans '{MODELS_DIR}'. Fichiers attendus: {expected}"
            )

        required_files = [SCALER_FILE, FEATURE_COLUMNS_FILE]
        missing_required = [f for f in required_files if not (MODELS_DIR / f).exists()]
        if missing_required:
            raise FileNotFoundError(
                f"Fichiers manquants dans '{MODELS_DIR}': {', '.join(missing_required)}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(MODELS_DIR / selected_model_file, mmap_mode=None)

        scaler = joblib.load(MODELS_DIR / SCALER_FILE, mmap_mode=None)
        feature_cols = joblib.load(MODELS_DIR / FEATURE_COLUMNS_FILE, mmap_mode=None)
        return model, scaler, feature_cols, None, model_type
    except Exception as e:
        error_msg = str(e)
        return None, None, None, error_msg, None


def get_sentiment_label(polarity):
    """
    Convertit la polarité du sentiment (-1 à 1) en étiquette lisible.
    Retourne un tuple : (label, emoji, couleur)
    """
    if polarity >= 0.3:
        return "Très +", "😊", "green"
    elif polarity >= 0.1:
        return "Positif", "🙂", "lightgreen"
    elif polarity > -0.1:
        return "Neutre", "😐", "gray"
    elif polarity > -0.3:
        return "Négatif", "😕", "orange"
    else:
        return "Très -", "😠", "red"


def count_pattern_matches(text, patterns):
    """Compte le nombre de patterns correspondants dans le texte"""
    text = str(text).lower()
    count = 0
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        count += len(matches)
    return count


def extract_features(text: str, rating: int = 5, sentiment_model=None) -> dict:
    """Extraction des caractéristiques d'un texte d'avis"""
    
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
    # Nettoyer le texte : enlever ponctuation pour mieux compter les répétitions
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = clean_text.split()
    words = [w for w in words if w]  # Enlever les chaînes vides
    unique_words = set(words)
    lexical_diversity = len(unique_words) / max(1, len(words))
    
    # NOUVEAU : Détection répétition excessive (indicateur de spam IA)
    word_repetition_ratio = 0
    try:
        if len(words) > 5:
            from collections import Counter
            
            # Stemming simple français
            def simple_stem(w):
                if len(w) <= 4:
                    return w
                # Enlever terminaisons courantes
                for suffix in [('ment', 4), ('ation', 5), ('ement', 5), ('ant', 3), ('ent', 3), 
                              ('aux', 3), ('eux', 3), ('ais', 3), ('ait', 3), ('es', 2), ('e', 1), ('s', 1)]:
                    if w.endswith(suffix[0]) and len(w) > suffix[1]:
                        return w[:-suffix[1]]
                return w
            
            # Stemmer les mots pour détecter les variations
            stemmed_words = [simple_stem(w) for w in words]
            word_counts = Counter(stemmed_words)
            
            # Stop words (stemmer aussi)
            stop_words_raw = ['le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'mais', 'de', 'du', 
                             'pour', 'par', 'dans', 'sur', 'est', 'sont', 'ne', 'pas', 'vraiment',
                             'the', 'a', 'an', 'and', 'or', 'but', 'to', 'of', 'in', 'on', 'is', 'was', 'were',
                             'tres', 'beaucoup', 'aussi', 'tout', 'toute']
            stop_words = set(simple_stem(sw) for sw in stop_words_raw)
            
            # Compter mots significatifs répétés 2+ fois
            repeated_count = sum(1 for stemmed_word, cnt in word_counts.items() 
                               if cnt >= 2 and stemmed_word not in stop_words and len(stemmed_word) > 3)
            word_repetition_ratio = repeated_count / max(1, len(unique_words))
    except Exception:
        word_repetition_ratio = 0
    
    # Analyse de sentiment - UTILISER LE MODELE BERT (CamemBERT pour le français)
    sentiment_polarity, sentiment_subjectivity = analyze_sentiment_bert(text, sentiment_model)
    
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
    
    # NOUVEAU : Détection des patterns marketing
    superlatives_count = count_pattern_matches(text, SUPERLATIVES)
    cta_count = count_pattern_matches(text, CALL_TO_ACTION)
    advertising_count = count_pattern_matches(text, ADVERTISING_TONE)
    too_perfect_count = count_pattern_matches(text, TOO_PERFECT_INDICATORS)
    emotional_count = count_pattern_matches(text, EMOTIONAL_EXAGGERATION)
    
    # Score de spam (pondéré)
    spam_score = (
        superlatives_count * 1.0 +
        cta_count * 2.0 +
        advertising_count * 2.5 +
        too_perfect_count * 1.5 +
        emotional_count * 0.5
    )
    
    # Score de perfection
    perfection_score = 0
    if rating == 5:
        perfection_score += 1
    negative_words = ['mais', 'cependant', 'toutefois', 'sauf', 'except', 
                      'unfortunately', 'however', 'although', 'dommage', 'bémol']
    has_negative = any(word in text.lower() for word in negative_words)
    if not has_negative:
        perfection_score += 1
    if exclamation_count >= 3:
        perfection_score += 1
    
    features = {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'uppercase_ratio': uppercase_ratio,
        'lexical_diversity': lexical_diversity,
        'word_repetition_ratio': word_repetition_ratio,
        'sentiment_polarity': sentiment_polarity,
        'sentiment_subjectivity': sentiment_subjectivity,
        'rating_sentiment_mismatch': rating_sentiment_mismatch,
        'digit_ratio': digit_ratio,
        'rating': rating,
        # NOUVELLES caractéristiques
        'superlatives_count': superlatives_count,
        'cta_count': cta_count,
        'advertising_count': advertising_count,
        'too_perfect_count': too_perfect_count,
        'emotional_count': emotional_count,
        'spam_score': spam_score,
        'perfection_score': perfection_score,
        'has_superlatives': 1 if superlatives_count > 0 else 0,
        'has_cta': 1 if cta_count > 0 else 0,
        'has_advertising': 1 if advertising_count > 0 else 0,
        'is_too_perfect': 1 if too_perfect_count > 0 else 0,
    }
    
    return features


def predict_review(text: str, rating: int, model, feature_cols, sentiment_model=None) -> tuple:
    """Prédiction si un avis est faux ou vrai avec approche hybride ML + règles"""
    
    # Extraction des caractéristiques
    features = extract_features(text, rating, sentiment_model)
    
    # Création du dataframe avec toutes les caractéristiques requises
    df = pd.DataFrame([features])
    
    # Ajout des caractéristiques manquantes avec valeurs par défaut
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Sélection uniquement des caractéristiques utilisées par le modèle
    X = df[feature_cols].copy()
    
    # Remplissage des valeurs NaN
    X = X.fillna(0)
    
    # Prédiction ML
    ml_prediction = model.predict(X)[0]
    ml_probability = model.predict_proba(X)[0].copy()
    
    # ==========================================================================
    # APPROCHE HYBRIDE : Ajustement de la probabilité selon les signaux marketing
    # ==========================================================================
    
    spam_score = features.get('spam_score', 0)
    superlatives = features.get('superlatives_count', 0)
    cta = features.get('cta_count', 0)
    advertising = features.get('advertising_count', 0)
    too_perfect = features.get('too_perfect_count', 0)
    perfection_score = features.get('perfection_score', 0)
    exclamation = features.get('exclamation_count', 0)
    uppercase_ratio = features.get('uppercase_ratio', 0)
    word_repetition = features.get('word_repetition_ratio', 0)
    
    # Calcul de la pénalité pour les patterns suspects
    penalty = 0.0
    
    # Pénalité du score de spam (indicateur principal)
    if spam_score >= 10:
        penalty += 0.4
    elif spam_score >= 5:
        penalty += 0.25
    elif spam_score >= 3:
        penalty += 0.15
    
    # Les appels à l'action sont très suspects
    if cta >= 2:
        penalty += 0.2
    elif cta >= 1:
        penalty += 0.1
    
    # Ton publicitaire
    if advertising >= 2:
        penalty += 0.25
    elif advertising >= 1:
        penalty += 0.1
    
    # Avis trop parfaits
    if too_perfect >= 2:
        penalty += 0.15
    elif too_perfect >= 1:
        penalty += 0.08
    
    # Répétition excessive (signe de spam IA ou bot)
    if word_repetition >= 0.3:
        penalty += 0.35  # Très suspect
    elif word_repetition >= 0.2:
        penalty += 0.25
    elif word_repetition >= 0.1:
        penalty += 0.15
    
    # Ponctuation/majuscules excessives
    if exclamation >= 5:
        penalty += 0.1
    if uppercase_ratio >= 0.3:
        penalty += 0.1
    
    # Superlatifs multiples
    if superlatives >= 3:
        penalty += 0.1
    
    # Plafonner la pénalité à 0.8
    penalty = min(penalty, 0.8)
    
    # Ajustement des probabilités
    # probability[0] = P(Faux), probability[1] = P(Vrai)
    adjusted_prob = np.array([
        ml_probability[0] + penalty * ml_probability[1],  # Augmenter P(Faux)
        ml_probability[1] * (1 - penalty)  # Diminuer P(Vrai)
    ])
    
    # Normaliser pour que la somme fasse 1
    adjusted_prob = adjusted_prob / adjusted_prob.sum()
    
    # Faire la prédiction finale basée sur la probabilité ajustée
    final_prediction = 0 if adjusted_prob[0] > 0.5 else 1
    
    # Stocker les infos d'ajustement dans les features pour l'affichage
    features['_penalty_applied'] = penalty
    features['_ml_prob_real'] = ml_probability[1]
    features['_adjusted_prob_real'] = adjusted_prob[1]
    
    return final_prediction, adjusted_prob, features


def main():
    # En-tête
    st.markdown('<h1 class="main-header">🛡️ Review Guardian</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Détection de faux avis par Intelligence Artificielle</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Chargement du modèle
    model, scaler, feature_cols, error, model_type = load_model()
    
    if error:
        st.error(f"❌ Erreur de chargement du modèle: {error}")
        st.info("💡 Ce dépôt public n'inclut pas les fichiers .joblib. Générez-les localement (retrain_models.py) ou fournissez-les dans models/ avant de lancer l'app.")
        return
    
    # Chargement du modèle de sentiment (CamemBERT)
    with st.spinner("🧠 Chargement du modèle de sentiment (CamemBERT)..."):
        sentiment_model, sentiment_error = load_sentiment_model()
    
    if sentiment_error and sentiment_error != "multilingual":
        st.warning(f"⚠️ Modèle de sentiment: fallback sur lexique ({sentiment_error})")
    
    # Barre latérale
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        rating = st.slider("Note de l'avis", 1, 5, 5, help="Note donnée par l'utilisateur")
        
        st.divider()
        
        st.header(" Exemples")
        if st.button("📝 Exemple Authentique"):
            st.session_state['example_text'] = """Super restaurant ! Nous y sommes allés hier soir en famille. 
Le service était un peu long mais la nourriture excellente. 
Le tiramisu maison vaut le détour. Je recommande pour une occasion spéciale."""
            
        if st.button("🚨 Exemple Suspect"):
            st.session_state['example_text'] = """INCROYABLE!!! MEILLEUR RESTAURANT DE LA VILLE!!! 
5 ETOILES MERITEES!!! TOUT EST PARFAIT!!! 
JE RECOMMANDE A 100%!!! VENEZ VITE!!!"""
        
        if st.button("📺 Exemple Publicitaire"):
            st.session_state['example_text'] = """Le meilleur produit de tous les temps ! Qualité premium garantie. 
N'hésitez pas, achetez maintenant ! Prix imbattable, livraison gratuite. 
Visitez www.exemple.com pour profiter de l'offre limitée ! Code promo: -50%"""
        
        if st.button("✨ Exemple Trop Parfait"):
            st.session_state['example_text'] = """Absolument parfait ! Rien à redire, tout est impeccable.
J'y vais chaque fois et c'est toujours sans faute. Aucun défaut, aucun problème.
Je suis fan absolu, c'est un coup de coeur total ! Tout était parfait sans exception."""
    
    # Contenu principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Entrez un avis à analyser")
        
        # Saisie de texte
        default_text = st.session_state.get('example_text', '')
        review_text = st.text_area(
            "Texte de l'avis",
            value=default_text,
            height=150,
            placeholder="Collez ou écrivez un avis ici..."
        )
        
        # Bouton d'analyse
        if st.button("🔍 Analyser l'avis", type="primary", use_container_width=True):
            if review_text.strip():
                with st.spinner("Analyse en cours..."):
                    prediction, probability, features = predict_review(
                        review_text, rating, model, feature_cols, sentiment_model
                    )
                    
                    st.session_state['result'] = {
                        'prediction': prediction,
                        'probability': probability,
                        'features': features,
                        'text': review_text
                    }
            else:
                st.warning("⚠️ Veuillez entrer un texte à analyser")
    
    with col2:
        st.header("🎯 Résultat")
        
        if 'result' in st.session_state:
            result = st.session_state['result']
            prediction = result['prediction']
            probability = result['probability']
            features = result['features']
            
            # Affichage du résultat
            if prediction == 1:  # Avis authentique
                prob_real = probability[1] * 100
                st.success(f"✅ **AVIS AUTHENTIQUE**")
                st.metric("Confiance", f"{prob_real:.1f}%")
                st.progress(prob_real / 100)
            else:  # Fake review
                prob_fake = probability[0] * 100
                st.error(f"🚨 **AVIS SUSPECT**")
                st.metric("Probabilité de faux", f"{prob_fake:.1f}%")
                st.progress(prob_fake / 100)
            
            # Afficher les infos d'ajustement si une pénalité a été appliquée
            penalty = features.get('_penalty_applied', 0)
            if penalty > 0:
                st.divider()
                st.caption("🔧 **Ajustement appliqué**")
                ml_prob = features.get('_ml_prob_real', 0) * 100
                adj_prob = features.get('_adjusted_prob_real', 0) * 100
                st.caption(f"ML brut: {ml_prob:.0f}% → Ajusté: {adj_prob:.0f}%")
                st.caption(f"Pénalité: -{penalty*100:.0f}%")
            
            # Jauge de confiance
            st.divider()
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("P(Fake)", f"{probability[0]*100:.1f}%")
            with col_b:
                st.metric("P(Real)", f"{probability[1]*100:.1f}%")
    
    # Détails des caractéristiques (extensible)
    if 'result' in st.session_state:
        st.divider()
        
        with st.expander("📊 Détails de l'analyse"):
            features = st.session_state['result']['features']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mots", features.get('word_count', 0))
                st.metric("Phrases", features.get('sentence_count', 0))
            
            with col2:
                polarity = features.get('sentiment_polarity', 0)
                sent_label, sent_emoji, _ = get_sentiment_label(polarity)
                st.metric("Ton de l'avis", f"{sent_emoji} {sent_label}", help="Positif = l'auteur est content, Négatif = l'auteur est mécontent. Ce n'est PAS la fiabilité de l'avis!")
                st.metric("Subjectivité", f"{features.get('sentiment_subjectivity', 0):.2f}")
            
            with col3:
                st.metric("! Count", features.get('exclamation_count', 0))
                st.metric("? Count", features.get('question_count', 0))
            
            with col4:
                st.metric("Diversité lexicale", f"{features.get('lexical_diversity', 0):.2f}")
                st.metric("Ratio majuscules", f"{features.get('uppercase_ratio', 0):.2%}")
            
            # Signaux d'alerte
            st.subheader("🔍 Signaux détectés")
            
            warnings = []
            if features.get('exclamation_count', 0) > 3:
                warnings.append("⚠️ Beaucoup de points d'exclamation")
            if features.get('uppercase_ratio', 0) > 0.3:
                warnings.append("⚠️ Beaucoup de majuscules")
            if features.get('word_count', 0) < 10:
                warnings.append("⚠️ Avis très court")
            if features.get('lexical_diversity', 0) < 0.5:
                warnings.append("⚠️ Faible diversité du vocabulaire")
            if features.get('rating_sentiment_mismatch', 0) > 0.5:
                warnings.append("⚠️ Décalage note/sentiment")
            
            # NOUVEAU : Alertes des patterns marketing
            if features.get('superlatives_count', 0) > 0:
                warnings.append(f"🎯 Superlatifs marketing détectés ({features.get('superlatives_count')})")
            if features.get('cta_count', 0) > 0:
                warnings.append(f"📢 Appels à l'action détectés ({features.get('cta_count')})")
            if features.get('advertising_count', 0) > 0:
                warnings.append(f"📺 Ton publicitaire détecté ({features.get('advertising_count')})")
            if features.get('too_perfect_count', 0) > 0:
                warnings.append(f"✨ Indicateurs d'avis trop parfait ({features.get('too_perfect_count')})")
            if features.get('emotional_count', 0) >= 2:
                warnings.append(f"💢 Exagérations émotionnelles ({features.get('emotional_count')})")
            if features.get('spam_score', 0) >= 5:
                warnings.append(f"🚨 Score de spam élevé: {features.get('spam_score'):.1f}")
            if features.get('perfection_score', 0) >= 2:
                warnings.append(f"⭐ Avis suspicieusement parfait")
            
            # Alerte critique: répétition excessive (spam IA/bot)
            word_rep = features.get('word_repetition_ratio', 0)
            if word_rep >= 0.2:
                warnings.append(f"🤖 RÉPÉTITION EXCESSIVE détectée - Possible spam IA ({word_rep:.0%})")
            elif word_rep >= 0.1:
                warnings.append(f"🔁 Répétitions suspectes détectées ({word_rep:.0%})")
            
            if warnings:
                for w in warnings:
                    st.write(w)
            else:
                st.write("✅ Aucun signal suspect détecté")
            
            # NOUVEAU : Section d'analyse marketing
            st.subheader("📊 Analyse Marketing")
            mcol1, mcol2, mcol3 = st.columns(3)
            with mcol1:
                st.metric("Superlatifs", features.get('superlatives_count', 0))
                st.metric("Appels action", features.get('cta_count', 0))
            with mcol2:
                st.metric("Ton pub", features.get('advertising_count', 0))
                st.metric("Trop parfait", features.get('too_perfect_count', 0))
            with mcol3:
                st.metric("Score Spam", f"{features.get('spam_score', 0):.1f}")
                st.metric("Score Perfection", features.get('perfection_score', 0))
    
    # Interface de chat
    st.divider()
    st.header("💬 Mode Chat")
    
    # Initialisation de l'historique du chat
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Bonjour ! 👋 Je suis Review Guardian. Envoyez-moi un avis et je vous dirai s'il semble authentique ou suspect."}
        ]
    
    # Affichage des messages du chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Saisie du chat
    if prompt := st.chat_input("Entrez un avis à analyser..."):
        # Ajouter le message de l'utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Génération de la réponse
        with st.chat_message("assistant"):
            if len(prompt.strip()) < 5:
                response = "⚠️ L'avis est trop court pour être analysé. Pouvez-vous m'envoyer un texte plus long ?"
            else:
                prediction, probability, features = predict_review(prompt, rating, model, feature_cols, sentiment_model)
                
                if prediction == 1:
                    prob = probability[1] * 100
                    sent_label, sent_emoji, _ = get_sentiment_label(features['sentiment_polarity'])
                    response = f"""✅ **Cet avis semble AUTHENTIQUE**

**Confiance:** {prob:.1f}%

📊 **Analyse du texte:**
- Mots: {features['word_count']}
- Ton de l'avis: {sent_emoji} {sent_label}
- Diversité lexicale: {features['lexical_diversity']:.2f}

💡 *Note: Un avis authentique peut être positif OU négatif - c'est l'opinion réelle de l'auteur.*"""
                else:
                    prob = probability[0] * 100
                    sent_label, sent_emoji, _ = get_sentiment_label(features['sentiment_polarity'])
                    response = f"""🚨 **Cet avis semble SUSPECT**

**Probabilité de faux:** {prob:.1f}%

📊 **Analyse du texte:**
- Mots: {features['word_count']}
- Ton de l'avis: {sent_emoji} {sent_label}
- Points d'exclamation: {features['exclamation_count']}

⚠️ Signaux suspects détectés. Nous recommandons une vérification manuelle."""
            
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Pied de page
    st.divider()
    st.markdown("""
    <p style="text-align: center; color: #888;">
        🛡️ Review Guardian - Projet 5MEM | Modèle Gradient Boosting
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
