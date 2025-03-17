import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_complete_dataset(file_path='data/massive_combat_dataset.csv'):
    """
    Charge le dataset complet et applique le feature engineering.
    """
    print(f"Chargement du dataset complet depuis {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
    
    # Charger le dataset complet
    data = pd.read_csv(file_path, low_memory=False)
    print(f"Dataset complet chargé: {len(data)} échantillons")
    
    # Afficher quelques statistiques
    print(f"Distribution des résultats de combat:")
    print(data['outcome'].value_counts())
    print(f"Ratio de victoires: {data['outcome'].mean():.2%}")
    
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
    
    # 3. Créer des features d'avantage quadratiques
    data['offensive_power_squared'] = data['offensive_power_ratio'] ** 2
    data['special_power_squared'] = data['special_power_ratio'] ** 2
    data['speed_advantage_squared'] = data['speed_ratio'] ** 2
    
    # 4. Créer des features d'interaction
    data['speed_offensive_interaction'] = data['speed_ratio'] * data['offensive_power_ratio']
    data['speed_special_interaction'] = data['speed_ratio'] * data['special_power_ratio']
    
    # 5. Créer des buckets pour certaines features importantes
    data['bst_ratio_bucket'] = pd.qcut(data['bst_ratio'], 5, labels=False, duplicates='drop')
    data['hp_ratio_bucket'] = pd.qcut(data['hp_ratio'], 5, labels=False, duplicates='drop')
    
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
    
    # 9. Caractéristiques de type (si elles existent)
    if 'counter_has_type_advantage' in data.columns:
        data['type_and_speed'] = data['counter_has_type_advantage'] * data['has_speed_advantage']
        data['type_and_offense'] = data['counter_has_type_advantage'] * data['has_offensive_advantage']
    
    # 10. Nouvelles caractéristiques logarithmiques pour capturer les effets non linéaires
    # Ajouter un petit epsilon pour éviter log(0)
    epsilon = 1e-5
    data['log_hp_ratio'] = np.log(data['hp_ratio'] + epsilon)
    data['log_speed_ratio'] = np.log(data['speed_ratio'] + epsilon)
    data['log_offensive_power'] = np.log(data['offensive_power_ratio'] + epsilon)
    data['log_special_power'] = np.log(data['special_power_ratio'] + epsilon)
    
    print(f"Nombre de features après engineering: {len(data.columns)}")
    
    return data

def prepare_data(data):
    """
    Prépare les données pour l'entraînement.
    """
    print("Préparation des données...")
    
    # Séparer les caractéristiques et la cible
    X = data.drop(['target_pokemon', 'counter_pokemon', 'outcome', 'win_probability'], axis=1)
    y = data['outcome']
    
    # Gérer les valeurs infinies ou NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)  # Remplacer les NaN par 0
    
    # Convertir les colonnes catégorielles
    cat_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=cat_columns)
    
    # Normalisation des features numériques (optionnel pour XGBoost, mais aide parfois)
    num_columns = X.select_dtypes(include=['float64', 'float32', 'int64']).columns
    scaler = StandardScaler()
    X[num_columns] = scaler.fit_transform(X[num_columns])
    
    # Diviser en ensembles d'entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Dimensions finales: X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def train_final_model(X_train, y_train, X_test, y_test):
    """
    Entraîne le modèle final avec les hyperparamètres optimaux.
    """
    print("Entraînement du modèle final sur l'ensemble du dataset...")
    
    # Hyperparamètres optimaux trouvés par Optuna
    best_params = {
        'n_estimators': 1250,
        'max_depth': 16,
        'learning_rate': 0.011941556341855092,
        'subsample': 0.6426320301863463,
        'colsample_bytree': 0.973192516435812,
        'colsample_bylevel': 0.9766850132536867,
        'colsample_bynode': 0.6220287884219879,
        'gamma': 7.975863861919555,
        'min_child_weight': 14,
        'max_delta_step': 19,
        'reg_alpha': 9.978839462375149,
        'reg_lambda': 3.4809742175925833,
        'scale_pos_weight': 1.446308380167218,
        'grow_policy': 'lossguide',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device': 'cuda',
        'tree_method': 'hist',
        'random_state': 464
    }
    
    # Créer et entraîner le modèle
    start_time = time.time()
    
    model = xgb.XGBClassifier(**best_params)
    
    try:
        # Essayer avec callbacks pour early stopping
        callbacks = [xgb.callback.EarlyStopping(rounds=50)]
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=callbacks,
            verbose=True
        )
    except Exception as e:
        print(f"Erreur lors de l'entraînement avec callbacks: {str(e)}")
        try:
            # Essayer avec early_stopping_rounds
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=True
            )
        except Exception as e2:
            print(f"Erreur lors de l'entraînement avec early_stopping_rounds: {str(e2)}")
            # Fallback: entraînement standard
            model.fit(X_train, y_train, verbose=True)
    
    # Calculer le temps d'entraînement
    training_time = time.time() - start_time
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    seconds = training_time % 60
    
    print(f"Entraînement terminé en {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return model, training_time

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle final avec des métriques détaillées.
    """
    print("Évaluation du modèle final...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculer diverses métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }
    
    print("\nMétriques d'évaluation finale:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Rapport détaillé
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nRapport de classification détaillé:")
    print(classification_report(y_test, y_pred))
    
    print("\nMatrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Analyse des features importantes
    importance = model.feature_importances_
    feature_names = X_test.columns
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 features les plus importantes:")
    for i, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return metrics, feature_importance

def save_final_model(model, metrics, feature_importance, scaler, training_time):
    """
    Sauvegarde le modèle final et ses métadonnées.
    """
    print("Sauvegarde du modèle final...")
    
    # Créer le répertoire si nécessaire
    os.makedirs('models', exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = 'models/final_model_complete_dataset.joblib'
    joblib.dump(model, model_path)
    
    # Sauvegarder les métadonnées
    metadata = {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'scaler': scaler,
        'training_time': training_time,
        'hyperparameters': model.get_params(),
        'creation_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'input_features': feature_importance['feature'].tolist(),
        'dataset_size': 'complete'
    }
    
    metadata_path = 'models/final_model_complete_dataset_metadata.joblib'
    joblib.dump(metadata, metadata_path)
    
    print(f"Modèle sauvegardé dans {model_path}")
    print(f"Métadonnées sauvegardées dans {metadata_path}")

def main():
    # Charger le dataset complet
    data = load_complete_dataset()
    
    # Préparer les données
    X_train, X_test, y_train, y_test, scaler = prepare_data(data)
    
    # Entraîner le modèle final avec les hyperparamètres optimaux
    model, training_time = train_final_model(X_train, y_train, X_test, y_test)
    
    # Évaluer le modèle
    metrics, feature_importance = evaluate_model(model, X_test, y_test)
    
    # Sauvegarder le modèle et les métadonnées
    save_final_model(model, metrics, feature_importance, scaler, training_time)
    
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU MODÈLE FINAL SUR DATASET COMPLET TERMINÉ AVEC SUCCÈS!")
    print("="*80)

if __name__ == "__main__":
    main() 