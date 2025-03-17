import os
import sys
import pandas as pd
import time
import xgboost as xgb

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model_distiller import PokemonModelDistiller

def train_from_large_dataset(use_gpu=True):
    """
    Entraîne un modèle distillé à partir du grand dataset généré.
    
    Args:
        use_gpu: Si True, utilise le GPU pour l'entraînement
    """
    print("Entraînement du modèle distillé à partir du grand dataset...")
    if use_gpu:
        print("Configuration pour utilisation GPU (RTX 3070)...")
    
    # Vérifier si le fichier de données existe
    data_path = "data/generated_combat_data.csv"
    if not os.path.exists(data_path):
        print(f"Erreur: Le fichier {data_path} n'existe pas.")
        print("Veuillez d'abord générer le dataset avec generate_large_dataset.py")
        return
    
    # Créer une instance du distillateur de modèle
    distiller = PokemonModelDistiller()
    
    # Charger le système existant (nécessaire pour les comparaisons)
    if not distiller.load_existing_system():
        print("Impossible de charger le système existant. Arrêt du processus.")
        return
    
    # Charger les données générées
    print(f"Chargement des données depuis {data_path}...")
    distiller.generated_data = pd.read_csv(data_path)
    
    # Afficher des statistiques sur les données
    print(f"Nombre de combats chargés: {len(distiller.generated_data)}")
    
    # Modifier la méthode d'entraînement pour utiliser le GPU si demandé
    if use_gpu:
        original_train_method = distiller.train_distilled_model
        
        def gpu_train_method():
            """Version modifiée utilisant le GPU"""
            if distiller.generated_data is None or len(distiller.generated_data) == 0:
                print("Pas de données générées. Veuillez d'abord appeler generate_combat_data().")
                return
            
            print("Entraînement du modèle distillé avec accélération GPU...")
            
            # Préparer les données
            X = distiller.generated_data.drop(['target_pokemon', 'counter_pokemon', 'win_probability'], axis=1)
            y = distiller.generated_data['win_probability']
            
            # Encodage one-hot pour les variables catégorielles
            X = pd.get_dummies(X, columns=['target_type1', 'target_type2', 'counter_type1', 'counter_type2'])
            
            # Diviser en ensembles d'entraînement et de test
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entraîner un modèle sur GPU
            start_time = time.time()
            
            # Hyperparamètres pour le modèle distillé avec GPU
            params = {
                'n_estimators': 200,        # Plus d'arbres car le GPU est plus rapide
                'max_depth': 8,             # Augmenté pour mieux capturer les relations
                'learning_rate': 0.05,      # Diminué pour plus de précision
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist',  # Utiliser le GPU
                'gpu_id': 0,                # ID du premier GPU
                'predictor': 'gpu_predictor', # Prédicteur GPU
                'random_state': 42
            }
            
            distiller.distilled_model = xgb.XGBRegressor(**params)
            distiller.distilled_model.fit(X_train, y_train)
            
            training_time = time.time() - start_time
            
            # Évaluer le modèle
            from sklearn.metrics import mean_squared_error
            import numpy as np
            y_pred = distiller.distilled_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            print(f"Modèle distillé entraîné en {training_time:.2f} secondes avec GPU")
            print(f"RMSE: {rmse:.4f}")
            
            # Sauvegarder le modèle
            import joblib
            joblib.dump(distiller.distilled_model, 'models/distilled_counter_finder.joblib')
            print("Modèle distillé sauvegardé dans models/distilled_counter_finder.joblib")
            
            return distiller.distilled_model
        
        # Remplacer temporairement la méthode
        distiller.train_distilled_model = gpu_train_method
    
    # Entraîner le modèle
    start_time = time.time()
    model = distiller.train_distilled_model()
    training_time = time.time() - start_time
    
    print(f"Entraînement terminé en {training_time:.2f} secondes.")
    print("Modèle distillé sauvegardé dans models/distilled_counter_finder.joblib")
    
    # Tester le modèle sur quelques Pokémon populaires
    test_pokemon = ['Tyranitar', 'Charizard', 'Dragonite', 'Gengar', 'Mewtwo']
    
    print("\n=== Test du modèle distillé sur des Pokémon populaires ===")
    for pokemon in test_pokemon:
        distiller.test_specific_pokemon(pokemon, top_k=3)
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entraîne le modèle distillé sur un grand dataset.')
    parser.add_argument('--cpu', action='store_true', help='Force l\'utilisation du CPU au lieu du GPU')
    args = parser.parse_args()
    
    train_from_large_dataset(use_gpu=not args.cpu) 