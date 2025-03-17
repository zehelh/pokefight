import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_and_enhance_dataset(file_path='data/massive_combat_dataset.csv', sample_fraction=0.3):
    """
    Charge et améliore le dataset avec des features plus informatives.
    """
    print(f"Chargement et amélioration du dataset ({sample_fraction*100}%)...")
    
    # Charger les données (même code que précédemment)
    if sample_fraction < 1.0:
        chunks = []
        total_rows = sum(1 for _ in open(file_path)) - 1
        sample_size = int(total_rows * sample_fraction)
        
        print(f"Échantillonnage de {sample_fraction*100}% du dataset (~{sample_size} lignes)")
        
        chunk_size = min(100000, total_rows // 10)
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            sample = chunk.sample(frac=sample_fraction, random_state=42)
            chunks.append(sample)
        
        data = pd.concat(chunks, ignore_index=True)
    else:
        data = pd.read_csv(file_path, low_memory=False)
    
    print(f"Dataset chargé: {len(data)} échantillons")
    
    # Feature engineering avancé
    print("Application du feature engineering...")
    
    # 1. Créer des ratios plus significatifs
    data['offensive_power_ratio'] = data['counter_attack'] / data['target_defense']
    data['special_power_ratio'] = data['counter_sp_attack'] / data['target_sp_defense']
    data['defensive_power_ratio'] = data['counter_defense'] / data['target_attack']
    data['special_defense_ratio'] = data['counter_sp_defense'] / data['target_sp_attack']
    
    # 2. Créer des features d'avantage combiné
    data['combined_power'] = data[['offensive_power_ratio', 'special_power_ratio']].max(axis=1)
    data['combined_defense'] = data[['defensive_power_ratio', 'special_defense_ratio']].max(axis=1)
    
    # 3. Créer des features d'avantage quadratiques (pour capturer les effets non linéaires)
    data['offensive_power_squared'] = data['offensive_power_ratio'] ** 2
    data['special_power_squared'] = data['special_power_ratio'] ** 2
    data['speed_advantage_squared'] = data['speed_ratio'] ** 2
    
    # 4. Créer des features d'interaction
    data['speed_offensive_interaction'] = data['speed_ratio'] * data['offensive_power_ratio']
    data['speed_special_interaction'] = data['speed_ratio'] * data['special_power_ratio']
    
    # 5. Créer des buckets (discrétisation) pour certaines features importantes
    data['bst_ratio_bucket'] = pd.qcut(data['bst_ratio'], 5, labels=False)
    data['hp_ratio_bucket'] = pd.qcut(data['hp_ratio'], 5, labels=False)
    
    # 6. Différences absolues pour certaines stats
    data['hp_difference'] = data['counter_hp'] - data['target_hp']
    data['attack_difference'] = data['counter_attack'] - data['target_attack']
    data['defense_difference'] = data['counter_defense'] - data['target_defense']
    data['sp_attack_difference'] = data['counter_sp_attack'] - data['target_sp_attack']
    data['sp_defense_difference'] = data['counter_sp_defense'] - data['target_sp_defense']
    data['speed_difference'] = data['counter_speed'] - data['target_speed']
    
    # 7. Indicateurs booléens pour avantages significatifs
    data['has_speed_advantage'] = (data['speed_ratio'] > 1.2).astype(int)
    data['has_offensive_advantage'] = (data['offensive_power_ratio'] > 1.2).astype(int)
    data['has_defensive_advantage'] = (data['defensive_power_ratio'] > 1.2).astype(int)
    
    # 8. Créer une feature composite pour l'avantage offensif vs défensif
    data['offense_vs_defense_bias'] = (data['offensive_power_ratio'] + data['special_power_ratio']) / (data['defensive_power_ratio'] + data['special_defense_ratio'])
    
    # Afficher les nouvelles features
    print(f"Nombre de features après engineering: {len(data.columns)}")
    
    return data

def prepare_advanced_data(data):
    """
    Prépare les données avec des techniques avancées.
    """
    print("Préparation avancée des données...")
    
    # Séparer les caractéristiques et la cible
    X = data.drop(['target_pokemon', 'counter_pokemon', 'outcome', 'win_probability'], axis=1)
    y = data['outcome']
    
    # Gérer les valeurs infinies ou NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)  # Remplacer les NaN par 0
    
    # Convertir les colonnes catégorielles
    cat_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=cat_columns)
    
    # Normalisation des features numériques
    num_columns = X.select_dtypes(include=['float64', 'float32', 'int64']).columns
    scaler = StandardScaler()
    X[num_columns] = scaler.fit_transform(X[num_columns])
    
    # Diviser en ensembles d'entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Dimensions finales: X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def train_ensemble_model(X_train, y_train, X_test, y_test):
    """
    Entraîne un modèle ensemble basé sur XGBoost avec paramètres optimisés.
    """
    print("Entraînement d'un modèle ensemble optimisé...")
    
    # Paramètres optimisés manuellement (basés sur les résultats Optuna précédents)
    params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.2,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1.2,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 42
    }
    
    # Entraîner le modèle
    model = xgb.XGBClassifier(**params)
    
    # Utiliser early stopping
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=True
        )
    except TypeError:
        try:
            # Tentative avec callbacks
            callbacks = [xgb.callback.EarlyStopping(rounds=50)]
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=callbacks,
                verbose=True
            )
        except:
            # Fallback sans early stopping
            model.fit(X_train, y_train)
    
    # Évaluer le modèle
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred)
    }
    
    print("\nPerformances du modèle:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Analyse des features importantes
    print("\nTop 20 features les plus importantes:")
    feature_importance = model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    for i, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics

def main():
    # Charger et améliorer le dataset (30% des données)
    data = load_and_enhance_dataset(sample_fraction=0.3)
    
    # Préparer les données avec techniques avancées
    X_train, X_test, y_train, y_test, scaler = prepare_advanced_data(data)
    
    # Entraîner le modèle ensemble
    model, metrics = train_ensemble_model(X_train, y_train, X_test, y_test)
    
    # Sauvegarder le modèle et ses métadonnées
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/enhanced_counter_model.joblib')
    
    metadata = {
        'metrics': metrics,
        'features': list(X_train.columns),
        'scaler': scaler,
        'sample_fraction': 0.3
    }
    
    joblib.dump(metadata, 'models/enhanced_counter_model_metadata.joblib')
    
    print("\n" + "="*80)
    print("MODÈLE AVANCÉ ENTRAÎNÉ AVEC SUCCÈS!")
    print("="*80)

if __name__ == "__main__":
    main() 