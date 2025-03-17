import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
from tqdm import tqdm

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_and_enhance_dataset(file_path='data/massive_combat_dataset.csv', sample_fraction=0.35):
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

def objective(trial, X_train, y_train):
    """
    Fonction objectif pour Optuna - optimise les hyperparamètres de XGBoost.
    """
    # Définir un espace de recherche très large
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 20),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 5.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device': 'cuda',  # Utiliser CUDA pour le GPU
        'tree_method': 'hist',  # Méthode moderne
        'random_state': trial.suggest_int('random_state', 0, 1000)
    }
    
    # Validation croisée
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**param)
        
        # Entraînement avec early stopping
        try:
            # Essayer différentes méthodes d'early stopping
            try:
                callbacks = [xgb.callback.EarlyStopping(rounds=50)]
                model.fit(
                    X_cv_train, y_cv_train,
                    eval_set=[(X_cv_val, y_cv_val)],
                    callbacks=callbacks,
                    verbose=False
                )
            except:
                try:
                    model.fit(
                        X_cv_train, y_cv_train,
                        eval_set=[(X_cv_val, y_cv_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                except:
                    # Fallback sans early stopping
                    model.fit(X_cv_train, y_cv_train)
            
            # Prédiction et métriques
            y_pred = model.predict(X_cv_val)
            y_proba = model.predict_proba(X_cv_val)[:, 1]
            
            acc = accuracy_score(y_cv_val, y_pred)
            auc = roc_auc_score(y_cv_val, y_proba)
            f1 = f1_score(y_cv_val, y_pred)
            
            # Score composite multi-métrique
            score = 0.2 * acc + 0.6 * auc + 0.2 * f1
            cv_scores.append(score)
        except Exception as e:
            print(f"Erreur dans l'essai avec paramètres {param}: {str(e)}")
            return 0.0  # Retourner un score minimal en cas d'erreur
    
    # Moyenne des scores de validation croisée
    return np.mean(cv_scores) if cv_scores else 0.0

def optimize_hyperparameters(X_train, y_train, n_trials=100):
    """
    Optimise les hyperparamètres avec Optuna.
    """
    print(f"Optimisation des hyperparamètres avec {n_trials} essais...")
    
    start_time = time.time()
    
    # Créer une étude Optuna
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    # Optimiser avec un timeout optionnel (7200 secondes = 2 heures)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=n_trials,
        timeout=10800  # 3 heures
    )
    
    # Récupérer les meilleurs hyperparamètres
    best_params = study.best_params
    best_score = study.best_value
    
    # Calculer le temps d'optimisation
    optimization_time = time.time() - start_time
    hours = optimization_time // 3600
    minutes = (optimization_time % 3600) // 60
    seconds = optimization_time % 60
    
    print(f"Optimisation terminée en {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Meilleur score composite: {best_score:.4f}")
    print("Meilleurs hyperparamètres:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Afficher les 5 meilleurs essais
    print("\nTop 5 des meilleurs essais:")
    best_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]
    for i, trial in enumerate(best_trials, 1):
        print(f"Essai {i}: Score = {trial.value:.4f}")
        print("  Paramètres importants:")
        params = trial.params
        for key in ['n_estimators', 'max_depth', 'learning_rate', 'scale_pos_weight']:
            print(f"    {key}: {params[key]}")
    
    # Sauvegarder l'étude pour analyse future
    joblib.dump(study, 'models/optuna_study_enhanced_features.joblib')
    print("Étude Optuna sauvegardée pour analyse future")
    
    return best_params

def train_optimized_model(X_train, y_train, X_test, y_test, best_params):
    """
    Entraîne un modèle XGBoost avec les hyperparamètres optimisés.
    """
    print("Entraînement du modèle final avec hyperparamètres optimisés...")
    
    # Ajouter les paramètres fixes
    params = best_params.copy()
    params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device': 'cuda',
        'tree_method': 'hist',
        'random_state': 42
    })
    
    # Créer et entraîner le modèle
    start_time = time.time()
    model = xgb.XGBClassifier(**params)
    
    # Entraînement avec early stopping
    try:
        try:
            callbacks = [xgb.callback.EarlyStopping(rounds=50)]
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=callbacks,
                verbose=True
            )
        except:
            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    early_stopping_rounds=50,
                    verbose=True
                )
            except:
                print("Note: L'arrêt anticipé n'est pas disponible. Entraînement sans early stopping.")
                model.fit(X_train, y_train)
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {str(e)}")
        raise
    
    # Temps d'entraînement
    training_time = time.time() - start_time
    
    # Évaluer le modèle
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculer diverses métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'training_time': training_time
    }
    
    print("\nMétriques d'évaluation du modèle optimisé:")
    for metric, value in metrics.items():
        if metric != 'training_time':
            print(f"  {metric}: {value:.4f}")
    
    # Rapport détaillé
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nRapport de classification détaillé:")
    print(classification_report(y_test, y_pred))
    
    print("\nMatrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Analyse des features importantes
    feature_importance = model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'feature': features, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    print("\nTop 20 features les plus importantes:")
    for i, row in importance_df.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, metrics, importance_df

def main():
    # Charger et améliorer le dataset (30% des données)
    data = load_and_enhance_dataset(sample_fraction=0.35)
    
    # Préparer les données
    X_train, X_test, y_train, y_test, scaler = prepare_advanced_data(data)
    
    # Optimiser les hyperparamètres avec Optuna
    best_params = optimize_hyperparameters(X_train, y_train, n_trials=200)
    
    # Entraîner le modèle avec les hyperparamètres optimisés
    model, metrics, feature_importance = train_optimized_model(X_train, y_train, X_test, y_test, best_params)
    
    # Sauvegarder le modèle et les métadonnées
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/optuna_enhanced_counter_model.joblib')
    
    # Sauvegarder les métadonnées
    metadata = {
        'best_params': best_params,
        'metrics': metrics,
        'features': list(X_train.columns),
        'scaler': scaler,
        'feature_importance': feature_importance,
        'sample_fraction': 0.35
    }
    
    joblib.dump(metadata, 'models/optuna_enhanced_counter_model_metadata.joblib')
    
    print("\n" + "="*80)
    print("MODÈLE OPTIMISÉ AVEC OPTUNA ENTRAÎNÉ AVEC SUCCÈS!")
    print("="*80)

if __name__ == "__main__":
    main() 