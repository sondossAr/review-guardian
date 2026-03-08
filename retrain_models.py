#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de réentraînement rapide des modèles avec l'environnement Python correct
"""
import sys
print(f"Python version: {sys.version}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

print("=" * 80)
print("RÉENTRAÎNEMENT DES MODÈLES - Environnement Python Anaconda")
print("=" * 80)

# Chemins
project_root = Path(__file__).parent
data_path = project_root / 'data' / 'processed' / 'features_extracted.csv'
models_dir = project_root / 'models'
models_dir.mkdir(exist_ok=True)

# 1. Charger les données
print("\n1. Chargement des données...")
df = pd.read_csv(data_path)
print(f"   ✓ {len(df)} reviews chargées")

# 2. Sélectionner toutes les features numériques (sauf le label et les IDs)
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                if col not in ['review_id', 'account_id', 
                               'approximate_localization.lat', 
                               'approximate_localization.lon', 
                               'cluster', 'is_real']]

X = df[feature_cols].fillna(0)
y = df['is_real']
features = feature_cols

print(f"   ✓ {len(features)} features sélectionnées: {features[:5]}...")

# 3. Train/test split
print("\n2. Séparation train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# 4. Normalisation
print("\n3. Normalisation des features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ Scaler entraîné")

# 5. Entraîner les modèles
print("\n4. Entraînement des modèles...")

# Logistic Regression
print("\n   a) Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
print(f"      ✓ Accuracy: {lr_acc:.4f}, F1: {lr_f1:.4f}")

# Random Forest
print("\n   b) Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
print(f"      ✓ Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")

# Gradient Boosting
print("\n   c) Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_pred = gb.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)
gb_prec = precision_score(y_test, gb_pred)
gb_rec = recall_score(y_test, gb_pred)
print(f"      ✓ Accuracy: {gb_acc:.4f}, F1: {gb_f1:.4f}")
print(f"      ✓ Precision: {gb_prec:.4f}, Recall: {gb_rec:.4f}")

# 6. Sauvegarder les modèles
print("\n5. Sauvegarde des modèles...")

# Sauvegarder avec protocole pickle compatible
joblib.dump(gb, models_dir / 'best_gb_model.joblib', compress=3, protocol=4)
joblib.dump(gb, models_dir / 'best_model.joblib', compress=3, protocol=4)
joblib.dump(rf, models_dir / 'best_rf_model.joblib', compress=3, protocol=4)
joblib.dump(scaler, models_dir / 'scaler.joblib', compress=3, protocol=4)
joblib.dump(features, models_dir / 'feature_columns.joblib', compress=3, protocol=4)

print(f"   ✓ best_gb_model.joblib (Gradient Boosting - {gb_acc:.4f})")
print(f"   ✓ best_model.joblib (Gradient Boosting - copie)")
print(f"   ✓ best_rf_model.joblib (Random Forest - {rf_acc:.4f})")
print(f"   ✓ scaler.joblib")
print(f"   ✓ feature_columns.joblib")

print("\n" + "=" * 80)
print("✅ RÉENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
print(f"   Meilleur modèle: Gradient Boosting - {gb_acc:.2%} accuracy")
print("=" * 80)
