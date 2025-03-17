import os
import sys

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.xgboost_counter_finder_prod import XGBoostCounterFinderProd as XGBoostCounterFinder


def main():
    """
    Fonction principale pour entraîner et tester le modèle XGBoostCounterFinder.
    """
    # Chemins vers les fichiers de données
    pokemon_path = 'data/pokemon.csv'
    matchups_path = 'data/pokemon_matchups.csv'
    usage_path = 'data/pokemon_usage_stats.csv'
    
    # Créer et entraîner le modèle
    counter_finder = XGBoostCounterFinder()
    counter_finder.load_data(pokemon_path, matchups_path, usage_path)
    
    # Entraîner le modèle avec Optuna et sauvegarder
    # Réduire le nombre d'essais pour accélérer l'optimisation
    metrics = counter_finder.train(save_path='models/xgboost_counter_finder.joblib')
    
    # Afficher un résumé des métriques
    print("\n" + "="*80)
    print("RÉSUMÉ DES MÉTRIQUES D'ÉVALUATION FINALES:")
    print("="*80)
    print(f"Précision: {metrics['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Précision: {metrics['precision']:.4f}")
    print(f"Temps d'entraînement: {metrics['training_time']:.2f} secondes")
    print("="*80)
    
    # Tester le modèle avec plus de Pokémon populaires
    test_pokemon = [
        'Tyranitar', 'Charizard', 'Landorus', 'Garchomp', 'Zapdos',
        'Gengar', 'Ferrothorn', 'Heatran', 'Excadrill', 'Toxapex'
    ]
    
    for pokemon in test_pokemon:
        print(f"\n{'='*80}")
        print(f"MEILLEURS CONTRE-POKÉMON POUR {pokemon.upper()}")
        print(f"{'='*80}")
        
        counters = counter_finder.find_counters(pokemon, top_k=5)
        
        for i, counter in enumerate(counters, 1):
            print(f"\n{i}. {counter['pokemon']} - {counter['win_probability']:.1%} de chance de gagner")
            print(f"   Source: {counter['source']}")
            print(f"   Raisons:")
            for reason in counter['reasons']:
                print(f"   • {reason}")
    
    # Afficher un message de conclusion
    print("\n" + "="*80)
    print("MODÈLE XGBOOST ENTRAÎNÉ ET TESTÉ AVEC SUCCÈS!")
    print("="*80)
    print(f"Le modèle a été sauvegardé dans 'models/xgboost_counter_finder.joblib'")
    print(f"Utilisez XGBoostCounterFinder.load_model() pour charger le modèle entraîné")
    print("="*80)


if __name__ == "__main__":
    main() 