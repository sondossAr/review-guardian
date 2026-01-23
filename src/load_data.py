"""
Module de Chargement des Données
Chargement et inspection du jeu de données GMR-PL
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# CHEMINS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"

# ============================================================================
# FONCTIONS DE CHARGEMENT
# ============================================================================

def load_reviews():
    """Chargement du jeu de données des avis"""
    reviews_path = DATA_RAW_DIR / "reviews.csv"
    print(f"📥 Chargement des avis depuis : {reviews_path}")
    
    if not reviews_path.exists():
        raise FileNotFoundError(f"Fichier d'avis non trouvé à {reviews_path}")
    
    reviews = pd.read_csv(reviews_path)
    print(f"✅ {len(reviews):,} avis chargés")
    return reviews

def load_accounts():
    """Chargement du jeu de données des comptes"""
    accounts_path = DATA_RAW_DIR / "accounts.csv"
    print(f"📥 Chargement des comptes depuis : {accounts_path}")
    
    if not accounts_path.exists():
        raise FileNotFoundError(f"Fichier de comptes non trouvé à {accounts_path}")
    
    accounts = pd.read_csv(accounts_path)
    print(f"✅ {len(accounts):,} comptes chargés")
    return accounts

# ============================================================================
# FONCTIONS D'INFORMATION
# ============================================================================

def print_dataset_info(reviews, accounts):
    """Affiche les informations complètes sur le jeu de données"""
    
    print("\n" + "=" * 80)
    print("APERCU DU JEU DE DONNEES")
    print("=" * 80)
    
    print(f"\n📊 AVIS :")
    print(f"  • Dimensions : {reviews.shape}")
    print(f"  • Colonnes : {list(reviews.columns)}")
    print(f"\n📊 COMPTES :")
    print(f"  • Dimensions : {accounts.shape}")
    print(f"  • Colonnes : {list(accounts.columns)}")
    
    # Affichage d'échantillon
    print(f"\n📝 ECHANTILLON D'AVIS (3 premières lignes) :")
    print(reviews.head(3).to_string())
    
    print(f"\n📝 ECHANTILLON DE COMPTES (3 premières lignes) :")
    print(accounts.head(3).to_string())

def print_data_types(reviews, accounts):
    """Affiche les types de données"""
    print("\n" + "=" * 80)
    print("TYPES DE DONNEES")
    print("=" * 80)
    
    print(f"\n📋 TYPES DES AVIS :")
    print(reviews.dtypes)
    
    print(f"\n📋 TYPES DES COMPTES :")
    print(accounts.dtypes)

def print_missing_values(reviews, accounts):
    """Affiche l'analyse des valeurs manquantes"""
    print("\n" + "=" * 80)
    print("ANALYSE DES VALEURS MANQUANTES")
    print("=" * 80)
    
    print(f"\n❓ VALEURS MANQUANTES AVIS :")
    missing_reviews = reviews.isnull().sum()
    if missing_reviews.sum() == 0:
        print("  ✅ Aucune valeur manquante !")
    else:
        for col, count in missing_reviews[missing_reviews > 0].items():
            pct = (count / len(reviews)) * 100
            print(f"  • {col}: {count:,} ({pct:.2f}%)")
    
    print(f"\n❓ VALEURS MANQUANTES COMPTES :")
    missing_accounts = accounts.isnull().sum()
    if missing_accounts.sum() == 0:
        print("  ✅ Aucune valeur manquante !")
    else:
        for col, count in missing_accounts[missing_accounts > 0].items():
            pct = (count / len(accounts)) * 100
            print(f"  • {col}: {count:,} ({pct:.2f}%)")

def print_label_distribution(reviews, accounts):
    """Affiche la distribution des étiquettes (faux vs vrai)"""
    print("\n" + "=" * 80)
    print("DISTRIBUTION DES ETIQUETTES (Faux vs Vrai)")
    print("=" * 80)
    
    # Avis
    print(f"\n📊 AVIS :")
    reviews_dist = reviews['is_real'].value_counts()
    reviews_pct = reviews['is_real'].value_counts(normalize=True) * 100
    
    print(f"  • Vrai (True) : {reviews_dist[True]:,} ({reviews_pct[True]:.2f}%)")
    print(f"  • Faux (False) : {reviews_dist[False]:,} ({reviews_pct[False]:.2f}%)")
    print(f"  • Ratio de déséquilibre : {reviews_dist[True]/reviews_dist[False]:.2f}:1")
    
    # Comptes
    print(f"\n📊 COMPTES :")
    accounts_dist = accounts['is_real'].value_counts()
    accounts_pct = accounts['is_real'].value_counts(normalize=True) * 100
    
    print(f"  • Vrai (True) : {accounts_dist[True]:,} ({accounts_pct[True]:.2f}%)")
    print(f"  • Faux (False) : {accounts_dist[False]:,} ({accounts_pct[False]:.2f}%)")
    print(f"  • Ratio de déséquilibre : {accounts_dist[True]/accounts_dist[False]:.2f}:1")

def print_statistics(reviews, accounts):
    """Affiche les statistiques détaillées"""
    print("\n" + "=" * 80)
    print("STATISTIQUES")
    print("=" * 80)
    
    print(f"\n📈 STATISTIQUES DES AVIS :")
    print(reviews.describe().to_string())
    
    print(f"\n📈 STATISTIQUES DES COMPTES :")
    print(accounts.describe().to_string())

# ============================================================================
# PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Démarrage du chargement et de l'exploration des données...\n")
    
    try:
        # Chargement des données
        reviews = load_reviews()
        accounts = load_accounts()
        
        # Affichage des informations
        print_dataset_info(reviews, accounts)
        print_data_types(reviews, accounts)
        print_missing_values(reviews, accounts)
        print_label_distribution(reviews, accounts)
        print_statistics(reviews, accounts)
        
        print("\n" + "=" * 80)
        print("✅ Chargement des données terminé !")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Erreur : {e}\n")
        import traceback
        traceback.print_exc()
