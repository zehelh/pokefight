import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import StackingClassifier

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Réutiliser les fonctions améliorées du script précédent
from src.models.improved_features_model import load_and_enhance_dataset, prepare_advanced_data

def train_stacking_model(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle de stacking qui combine plusieurs algorithmes.
    """
    print("Entraînement d'un modèle stacking...")
    
    # Premier niveau de modèles (base estimators)
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
        ('xgb', xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method='hist',
            device='cuda', random_state=42
        )),
        ('lgb', lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42
        ))
    ]
    
    # Modèle final (méta-learner)
    final_estimator = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    
    # Création du modèle de stacking
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    # Entraînement
    print("Entraînement des modèles de base et du méta-modèle...")
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("\nPerformances du modèle stacking:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return model, metrics

def main():
    # Charger et améliorer le dataset (30% des données)
    data = load_and_enhance_dataset(sample_fraction=0.3)
    
    # Préparer les données avec techniques avancées
    X_train, X_test, y_train, y_test, scaler = prepare_advanced_data(data)
    
    # Entraîner le modèle stacking
    model, metrics = train_stacking_model(X_train, y_train, X_test, y_test)
    
    # Sauvegarder le modèle et ses métadonnées
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stacking_counter_model.joblib')
    
    metadata = {
        'metrics': metrics,
        'features': list(X_train.columns),
        'scaler': scaler,
        'sample_fraction': 0.3
    }
    
    joblib.dump(metadata, 'models/stacking_counter_model_metadata.joblib')
    
    print("\n" + "="*80)
    print("MODÈLE STACKING ENTRAÎNÉ AVEC SUCCÈS!")
    print("="*80)

if __name__ == "__main__":
    main() 