import os
import sys
import time

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model_distiller import PokemonModelDistiller

def generate_large_dataset(target_count=200000):
    """
    Génère un grand dataset de combats pour l'entraînement du modèle distillé.
    
    Args:
        target_count: Nombre approximatif de combats à générer
    """
    print(f"Génération d'un dataset d'environ {target_count} combats...")
    
    # Créer une instance du distillateur de modèle
    distiller = PokemonModelDistiller()
    
    # Charger le système existant
    if not distiller.load_existing_system():
        print("Impossible de charger le système existant. Arrêt du processus.")
        return
    
    # Obtenir le nombre de Pokémon dans la base de données
    total_pokemon = len(distiller.pokemon_df)
    
    # Augmenter le top_k pour générer plus de combats par Pokémon
    # Modifie la méthode generate_combat_data pour augmenter le nombre de counters par Pokémon
    original_method = distiller.generate_combat_data
    
    def enhanced_generate_combat_data(sample_size=None, test_mode=True):
        if test_mode:
            return original_method(sample_size, test_mode)
        
        print("Utilisation du mode optimisé pour générer plus de combats...")
        
        # Liste des Pokémon cibles
        target_pokemon = distiller.pokemon_df['name'].tolist()
        
        # Échantillonner si nécessaire
        if sample_size is not None and sample_size < len(target_pokemon):
            import numpy as np
            np.random.seed(42)  # Pour la reproductibilité
            target_pokemon = np.random.choice(target_pokemon, size=sample_size, replace=False)
        
        # Liste pour stocker les données générées
        import pandas as pd
        from tqdm import tqdm
        
        # Créer un DataFrame vide pour stocker les résultats
        all_data = pd.DataFrame()
        
        # Générer les données en plusieurs passes
        # Cette approche permet d'écrire les résultats progressivement et d'économiser la mémoire
        batch_size = 50  # Nombre de Pokémon à traiter par lot
        
        for i in range(0, len(target_pokemon), batch_size):
            batch = target_pokemon[i:i+batch_size]
            print(f"Traitement du lot {i//batch_size + 1}/{len(target_pokemon)//batch_size + 1} ({len(batch)} Pokémon)")
            
            combat_data = []
            for pokemon in tqdm(batch):
                try:
                    # Augmenter le top_k à 50 (au lieu de 20) pour générer plus de counters
                    counters = distiller.existing_model.find_counters(pokemon, top_k=50)
                    
                    for counter in counters:
                        row = distiller._create_feature_row(pokemon, counter['pokemon'])
                        row['win_probability'] = counter['win_probability']
                        row['counter_has_type_advantage'] = any('super efficace' in reason for reason in counter['reasons'])
                        row['counter_has_stat_advantage'] = any(('def faible' in reason.lower() and 'atk haute' in reason.lower()) for reason in counter['reasons'])
                        combat_data.append(row)
                except Exception as e:
                    print(f"Erreur lors du traitement de {pokemon}: {str(e)}")
            
            # Convertir en DataFrame et l'ajouter aux résultats
            batch_df = pd.DataFrame(combat_data)
            all_data = pd.concat([all_data, batch_df], ignore_index=True)
            
            # Sauvegarder les résultats intermédiaires (pour éviter de tout perdre en cas d'erreur)
            all_data.to_csv('data/generated_combat_data_partial.csv', index=False)
            print(f"Données partielles sauvegardées. Total actuel: {len(all_data)} combats")
        
        # Sauvegarder les données complètes
        all_data.to_csv('data/generated_combat_data.csv', index=False)
        print(f"Données complètes sauvegardées. Total: {len(all_data)} combats")
        
        distiller.generated_data = all_data
        return all_data
    
    # Remplacer temporairement la méthode
    distiller.generate_combat_data = enhanced_generate_combat_data
    
    # Calculer un échantillon plus grand (utiliser pratiquement tous les Pokémon)
    # Le facteur 4 représente l'augmentation de top_k (de 20 à 50)
    sample_size = min(total_pokemon, max(200, target_count // 50))
    
    print(f"Nombre total de Pokémon disponibles: {total_pokemon}")
    print(f"Nombre de Pokémon à échantillonner: {sample_size}")
    print(f"Estimation du nombre de combats qui seront générés: ~{sample_size * 50}")
    
    # Mesurer le temps d'exécution
    start_time = time.time()
    
    # Générer les données
    data = distiller.generate_combat_data(sample_size=sample_size, test_mode=False)
    
    # Vérifier le nombre de combats générés
    combats_generes = len(data)
    
    # Afficher les statistiques
    execution_time = time.time() - start_time
    print(f"\nGénération terminée en {execution_time:.2f} secondes.")
    print(f"Nombre de combats effectivement générés: {combats_generes}")
    print(f"Dataset sauvegardé dans: data/generated_combat_data.csv")
    
    return data

if __name__ == "__main__":
    generate_large_dataset(200000)  # Augmenter la cible à 200 000 