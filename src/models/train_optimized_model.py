import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
from tqdm import tqdm

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_massive_dataset(file_path='data/massive_combat_dataset.csv', sample_fraction=0.1):
    """
    Charge le dataset massif généré précédemment.
    
    Args:
        file_path: Chemin vers le fichier du dataset
        sample_fraction: Fraction du dataset à échantillonner (0.1 = 10%)
    """
    print(f"Chargement du dataset depuis {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
    
    # Charger le dataset avec échantillonnage
    if sample_fraction < 1.0:
        # Pour les gros fichiers, on peut utiliser chunksize pour lire par morceaux
        # et échantillonner à la volée
        chunks = []
        total_rows = sum(1 for _ in open(file_path)) - 1  # -1 pour l'en-tête
        sample_size = int(total_rows * sample_fraction)
        
        print(f"Échantillonnage de {sample_fraction*100}% du dataset (environ {sample_size} lignes sur {total_rows})")
        
        # Lire en mode chunk
        chunk_size = min(100000, total_rows // 10)
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            sample = chunk.sample(frac=sample_fraction, random_state=42)
            chunks.append(sample)
        
        data = pd.concat(chunks, ignore_index=True)
        print(f"Dataset échantillonné avec succès: {len(data)} échantillons sur {total_rows}")
    else:
        # Charger tout le dataset
        data = pd.read_csv(file_path, low_memory=False)
        print(f"Dataset complet chargé: {len(data)} échantillons")
    
    # Afficher quelques statistiques
    print(f"Distribution des résultats de combat:")
    print(data['outcome'].value_counts())
    print(f"Ratio de victoires: {data['outcome'].mean():.2%}")
    
    return data

def prepare_data(data):
    """
    Prépare les données pour l'entraînement.
    """
    print("Préparation des données...")
    
    # Séparer les caractéristiques et la cible
    X = data.drop(['target_pokemon', 'counter_pokemon', 'outcome', 'win_probability'], axis=1)
    y = data['outcome']
    
    # Encodage one-hot pour les variables catégorielles
    cat_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=cat_columns)
    
    # Diviser en ensembles d'entraînement, validation et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Dimensions des données: X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def objective(trial, X_train, y_train):
    """
    Fonction objectif pour Optuna.
    """
    # Définir l'espace de recherche des hyperparamètres élargi
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 3.0),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device': 'cuda',  # Utiliser CUDA au lieu de tree_method='gpu_hist'
        'tree_method': 'hist',  # Méthode moderne recommandée avec device='cuda'
        'random_state': 42
    }
    
    # Valider les hyperparamètres avec validation croisée
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBClassifier(**param)
        
        # Créer un dictionnaire des paramètres de fit
        fit_params = {
            'eval_set': [(X_cv_val, y_cv_val)],
            'verbose': False
        }
        
        # Tenter d'ajouter early_stopping_rounds ou callbacks selon ce qui est disponible
        try:
            # Essayer la nouvelle API avec callbacks
            fit_params['callbacks'] = [xgb.callback.EarlyStopping(rounds=50)]
            model.fit(X_cv_train, y_cv_train, **fit_params)
        except Exception as e:
            # Revenir à l'ancienne API avec early_stopping_rounds ou sans arrêt anticipé
            try:
                fit_params.pop('callbacks', None)
                fit_params['early_stopping_rounds'] = 50
                model.fit(X_cv_train, y_cv_train, **fit_params)
            except Exception as e2:
                # Si les deux échouent, utiliser l'API de base sans arrêt anticipé
                model.fit(X_cv_train, y_cv_train, eval_set=[(X_cv_val, y_cv_val)], verbose=False)
        
        # Calculer plusieurs métriques d'évaluation et faire une moyenne pondérée
        y_pred = model.predict(X_cv_val)
        y_proba = model.predict_proba(X_cv_val)[:, 1]
        
        acc = accuracy_score(y_cv_val, y_pred)
        auc = roc_auc_score(y_cv_val, y_proba)
        f1 = f1_score(y_cv_val, y_pred)
        
        # Score composite - on valorise plus l'AUC qui est plus stable
        composite_score = 0.3 * acc + 0.5 * auc + 0.2 * f1
        cv_scores.append(composite_score)
    
    # Retourner la moyenne des scores
    return np.mean(cv_scores)

def optimize_hyperparameters(X_train, y_train, n_trials=100):
    """
    Optimise les hyperparamètres avec Optuna.
    """
    print(f"Optimisation des hyperparamètres avec {n_trials} essais...")
    
    start_time = time.time()
    
    # Créer une étude Optuna
    study = optuna.create_study(direction='maximize')
    
    # Pruner pour arrêter les essais non prometteurs
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    # Optimisation avec timeout facultatif (7200 secondes = 2 heures)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train), 
        n_trials=n_trials,
        timeout=7200  # Limite à 2 heures
    )
    
    # Récupérer les meilleurs hyperparamètres
    best_params = study.best_params
    best_score = study.best_value
    
    optimization_time = time.time() - start_time
    hours = optimization_time // 3600
    minutes = (optimization_time % 3600) // 60
    seconds = optimization_time % 60
    
    print(f"Optimisation terminée en {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Meilleur score: {best_score:.4f}")
    print("Meilleurs hyperparamètres:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Afficher les 5 meilleurs essais pour analyse
    print("\nTop 5 des meilleurs essais:")
    best_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('-inf'), reverse=True)[:5]
    for i, trial in enumerate(best_trials, 1):
        print(f"Essai {i}: Score = {trial.value:.4f}")
        for param_name, param_value in trial.params.items():
            print(f"  {param_name}: {param_value}")
    
    return best_params

def train_model(X_train, y_train, X_test, y_test, best_params):
    """
    Entraîne un modèle avec les meilleurs hyperparamètres.
    """
    print("Entraînement du modèle final...")
    
    # Ajouter les paramètres fixes
    params = best_params.copy()
    params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'device': 'cuda',
        'tree_method': 'hist',
        'random_state': 42
    })
    
    # Entraîner le modèle
    start_time = time.time()
    
    model = xgb.XGBClassifier(**params)
    
    # Créer un dictionnaire des paramètres de fit
    fit_params = {
        'eval_set': [(X_test, y_test)],
        'verbose': True
    }
    
    # Tenter d'ajouter early_stopping_rounds ou callbacks selon ce qui est disponible
    try:
        # Essayer la nouvelle API avec callbacks
        fit_params['callbacks'] = [xgb.callback.EarlyStopping(rounds=50)]
        model.fit(X_train, y_train, **fit_params)
    except Exception as e:
        # Revenir à l'ancienne API avec early_stopping_rounds
        try:
            fit_params.pop('callbacks', None)
            fit_params['early_stopping_rounds'] = 50
            model.fit(X_train, y_train, **fit_params)
        except Exception as e2:
            # Si les deux échouent, utiliser l'API de base sans arrêt anticipé
            print("Note: L'arrêt anticipé n'est pas disponible dans votre version de XGBoost.")
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    
    training_time = time.time() - start_time
    
    print(f"Entraînement terminé en {training_time:.2f} secondes")
    
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
    
    print("\nMétriques d'évaluation:")
    for metric, value in metrics.items():
        if metric != 'training_time':
            print(f"  {metric}: {value:.4f}")
    
    # Afficher un rapport plus détaillé
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nRapport de classification détaillé:")
    print(classification_report(y_test, y_pred))
    
    print("\nMatrice de confusion:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return model, metrics

def save_model(model, path='models/optimized_counter_model.joblib'):
    """
    Sauvegarde le modèle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")

def main():
    """
    Fonction principale.
    """
    # Charger 10% du dataset pour les tests
    data = load_massive_dataset(sample_fraction=0.1)
    
    # Préparer les données
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Optimiser les hyperparamètres avec plus d'essais
    best_params = optimize_hyperparameters(X_train, y_train, n_trials=100)
    
    # Entraîner le modèle
    model, metrics = train_model(X_train, y_train, X_test, y_test, best_params)
    
    # Sauvegarder le modèle
    save_model(model)
    
    # Sauvegarder les hyperparamètres et les métriques pour référence future
    results = {
        'best_params': best_params,
        'metrics': metrics,
        'features': list(X_train.columns),
        'sample_fraction': 0.1  # Ajouter cette information aux méta-données
    }
    
    joblib.dump(results, 'models/optimized_counter_model_info.joblib')
    print("Informations du modèle sauvegardées dans models/optimized_counter_model_info.joblib")
    
    print("\n" + "="*80)
    print("ENTRAÎNEMENT DU MODÈLE OPTIMISÉ TERMINÉ AVEC SUCCÈS!")
    print("="*80)

if __name__ == "__main__":
    main() 