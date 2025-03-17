import os
import sys

# Ajouter le répertoire parent au chemin Python pour permettre les importations relatives
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.xgboost_counter_finder import XGBoostCounterFinder

def train_model():
    """
    Entraîne le modèle XGBoostCounterFinder et le sauvegarde.
    """
    print("Début de l'entraînement du modèle...")
    
    # Définir les chemins des fichiers
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Chemins pour les données
    pokemon_path = os.path.join(base_dir, 'data', 'pokemon.csv')
    matchups_path = os.path.join(base_dir, 'data', 'pokemon_matchups.csv')
    usage_path = os.path.join(base_dir, 'data', 'pokemon_usage_stats.csv')
    
    # Chemin pour sauvegarder le modèle
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_counter_finder.joblib')
    
    # Vérifier si les fichiers de données existent
    if not os.path.exists(pokemon_path):
        print(f"Erreur: Fichier de données Pokémon non trouvé à {pokemon_path}")
        return False
    
    if not os.path.exists(matchups_path):
        print(f"Erreur: Fichier de matchups non trouvé à {matchups_path}")
        return False
    
    if not os.path.exists(usage_path):
        print(f"Erreur: Fichier de statistiques d'utilisation non trouvé à {usage_path}")
        return False
    
    # Créer et entraîner le modèle
    try:
        model = XGBoostCounterFinder()
        model.load_data(pokemon_path, matchups_path, usage_path)
        model.train(save_path=model_path)
        print(f"Modèle entraîné et sauvegardé avec succès à {model_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de l'entraînement du modèle: {e}")
        return False

if __name__ == "__main__":
    train_model() 