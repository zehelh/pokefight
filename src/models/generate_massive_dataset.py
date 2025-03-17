import os
import sys
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from tqdm import tqdm

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.model_distiller import PokemonModelDistiller
from src.models.xgboost_counter_finder_prod import XGBoostCounterFinderProd

class GPUAcceleratedCounterFinder(XGBoostCounterFinderProd):
    """
    Version accélérée par GPU du modèle XGBoostCounterFinder.
    """
    
    def __init__(self, model_path=None):
        # Initialiser sans charger de modèle
        self.pokemon_df = None
        self.matchups_df = None
        self.usage_df = None
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
        # Si model_path est un objet Pipeline, l'utiliser directement
        if model_path is not None and not isinstance(model_path, (str, bytes, os.PathLike)):
            self.model = model_path
            if hasattr(self.model, 'named_steps'):
                self.preprocessor = self.model.named_steps.get('preprocessor')
                
                # Récupérer les noms des caractéristiques
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = self.model.feature_names_in_
                else:
                    # Fallback si les noms ne sont pas stockés dans le modèle
                    self.feature_names = [
                        'p1_type1', 'p1_type2', 'p2_type1', 'p2_type2',
                        'p1_hp', 'p1_attack', 'p1_defense', 'p1_sp_attack', 'p1_sp_defense', 'p1_speed',
                        'p2_hp', 'p2_attack', 'p2_defense', 'p2_sp_attack', 'p2_sp_defense', 'p2_speed',
                        'p1_bst', 'p2_bst', 'ratio_speed', 'ratio_attack_vs_defense', 
                        'ratio_sp_attack_vs_sp_defense', 'ratio_bst'
                    ]
        # Sinon, utiliser le constructeur parent pour charger le modèle depuis un fichier
        elif model_path and os.path.exists(model_path):
            super().__init__(model_path)
            
        # Convertir le modèle pour utilisation GPU si possible
        if hasattr(self, 'model') and hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
            try:
                # Extraire le modèle XGBoost
                xgb_model = self.model.named_steps['classifier']
                
                # Configurer pour utiliser le GPU
                xgb_model.set_params(tree_method='gpu_hist', predictor='gpu_predictor')
                
                print("Modèle XGBoost configuré pour utiliser le GPU")
            except Exception as e:
                print(f"Impossible de configurer le GPU pour le modèle XGBoost: {e}")
    
    def find_counters_batch(self, pokemon_list, top_k=800):
        """
        Trouve les meilleurs contre-Pokémon pour une liste de Pokémon en mode batch.
        Cette méthode est optimisée pour le traitement par lots.
        """
        results = {}
        
        # Préparer les données pour la prédiction par lots
        all_candidates = []
        all_matchups = []
        
        print(f"Préparation des candidats pour {len(pokemon_list)} Pokémon...")
        
        # Construire toutes les paires possibles pour le traitement par lots
        for pokemon_name in pokemon_list:
            # Vérifier si le Pokémon existe
            target_pokemon = self.pokemon_df[self.pokemon_df['name'] == pokemon_name]
            if target_pokemon.empty:
                print(f"Pokémon {pokemon_name} non trouvé")
                results[pokemon_name] = []
                continue
            
            target_pokemon = target_pokemon.iloc[0]
            candidates = []
            
            # Parcourir tous les Pokémon comme contre-candidats
            for _, counter_pokemon in self.pokemon_df.iterrows():
                counter_name = counter_pokemon['name']
                
                # Ne pas comparer avec lui-même
                if counter_name == pokemon_name:
                    continue
                
                # Vérifier si le Pokémon a plus de 300 stats totales
                total_stats = (
                    counter_pokemon['hp'] + counter_pokemon['attack'] + counter_pokemon['defense'] + 
                    counter_pokemon['sp_attack'] + counter_pokemon['sp_defense'] + counter_pokemon['speed']
                )
                
                if total_stats < 300:
                    continue
                
                # Ajouter à la liste des candidats
                candidates.append((counter_name, counter_pokemon, target_pokemon))
                
                # Créer un DataFrame pour la prédiction
                matchup = pd.DataFrame({
                    'Pokemon1': [counter_name],
                    'Pokemon2': [pokemon_name],
                    'Win_Rate': [0.5]
                })
                
                # Ajouter aux matchups à prédire
                all_candidates.append((pokemon_name, counter_name, counter_pokemon, target_pokemon))
                all_matchups.append(matchup)
        
        if not all_matchups:
            return results
        
        # Combiner tous les matchups en un seul DataFrame
        combined_matchups = pd.concat(all_matchups, ignore_index=True)
        
        # Ajouter les caractéristiques
        print("Ajout des caractéristiques et préparation du modèle...")
        combined_matchups = self._add_pokemon_features(combined_matchups, 'Pokemon1', 'p1')
        combined_matchups = self._add_pokemon_features(combined_matchups, 'Pokemon2', 'p2')
        combined_matchups = self._calculate_derived_features(combined_matchups)
        
        # Sélectionner les caractéristiques
        X = combined_matchups[self.feature_names]
        
        # Prédire toutes les probabilités de victoire en une seule fois (utilise le GPU)
        print("Prédiction des probabilités de victoire par lots (GPU)...")
        win_probs = self.model.predict_proba(X)[:, 1]
        
        # Organiser les résultats
        for i, (pokemon_name, counter_name, counter_pokemon, target_pokemon) in enumerate(all_candidates):
            if pokemon_name not in results:
                results[pokemon_name] = []
            
            # Vérifier les avantages de type
            try:
                type_advantages, has_super_effective = self._check_type_advantages(counter_pokemon, target_pokemon)
                
                # Récupérer la probabilité prédite
                win_prob = win_probs[i]
                
                # Ajuster la probabilité si nécessaire
                if has_super_effective:
                    adjusted_win_prob = min(win_prob * 1.2, 0.95)
                else:
                    adjusted_win_prob = win_prob
                
                # Générer les raisons
                reasons = []
                
                # Type
                type_text = counter_pokemon['type1']
                if pd.notna(counter_pokemon['type2']):
                    type_text += f"/{counter_pokemon['type2']}"
                reasons.append(f"Types: {type_text}")
                
                # Avantages statistiques
                for stat, label in [
                    ('hp', 'PV'), ('attack', 'Attaque'), ('defense', 'Défense'),
                    ('sp_attack', 'Attaque Spéciale'), ('sp_defense', 'Défense Spéciale'),
                    ('speed', 'Vitesse')
                ]:
                    if counter_pokemon[stat] > target_pokemon[stat] + 20:
                        reasons.append(f"Meilleur en {label} (+{counter_pokemon[stat] - target_pokemon[stat]})")
                
                # Ajouter les avantages de type
                reasons.extend(type_advantages)
                
                # Créer l'objet counter
                counter_data = {
                    'pokemon': counter_name,
                    'win_probability': float(adjusted_win_prob),
                    'raw_win_probability': float(win_prob),
                    'source': 'Modèle XGBoost',
                    'reasons': reasons,
                    'has_super_effective': has_super_effective
                }
                
                # Ajouter à la liste des résultats
                results[pokemon_name].append(counter_data)
                
            except Exception as e:
                print(f"Erreur lors de la prédiction pour {counter_name} contre {pokemon_name}: {str(e)}")
        
        # Trier et limiter les résultats
        for pokemon_name in results:
            if results[pokemon_name]:
                # Trier par probabilité de victoire
                results[pokemon_name].sort(key=lambda x: x['win_probability'], reverse=True)
                # Limiter au nombre demandé
                results[pokemon_name] = results[pokemon_name][:top_k]
        
        return results

def generate_massive_dataset(target_count=500000, batch_size=10, counters_per_pokemon=100, simulations_per_pair=10):
    """
    Génère un dataset massif de combats avec résultats binaires (victoire/défaite) pour l'entraînement.
    
    Args:
        target_count: Nombre approximatif de combats à générer
        batch_size: Nombre de Pokémon à traiter par lot
        counters_per_pokemon: Nombre de counters à générer par Pokémon
        simulations_per_pair: Nombre de combats simulés pour chaque paire de Pokémon
    """
    print(f"Génération d'un dataset d'environ {target_count} combats avec simulation de résultats...")
    print(f"Configuration: {batch_size} Pokémon par lot, {counters_per_pokemon} counters par Pokémon, {simulations_per_pair} simulations par paire")
    
    # Créer une instance du distillateur de modèle
    distiller = PokemonModelDistiller()
    
    # Charger le système existant
    if not distiller.load_existing_system():
        print("Impossible de charger le système existant. Arrêt du processus.")
        return
    
    # Remplacer le modèle par la version accélérée GPU
    model_obj = distiller.existing_model.model  # C'est un objet Pipeline, pas un chemin
    gpu_finder = GPUAcceleratedCounterFinder(model_obj)  # Passer l'objet modèle directement
    gpu_finder.pokemon_df = distiller.pokemon_df
    gpu_finder.matchups_df = distiller.matchups_df
    gpu_finder.usage_df = distiller.usage_df
    distiller.existing_model = gpu_finder
    
    # Obtenir le nombre de Pokémon dans la base de données
    total_pokemon = len(distiller.pokemon_df)
    
    # Calculer combien de Pokémon nous devons traiter pour atteindre la cible
    estimated_battles_per_pokemon = counters_per_pokemon * simulations_per_pair
    sample_size = min(total_pokemon, max(200, (target_count + estimated_battles_per_pokemon - 1) // estimated_battles_per_pokemon))
    
    print(f"Nombre total de Pokémon disponibles: {total_pokemon}")
    print(f"Nombre de Pokémon à échantillonner: {sample_size}")
    print(f"Estimation du nombre de combats qui seront générés: ~{sample_size * estimated_battles_per_pokemon}")
    
    # Liste des Pokémon cibles (tous)
    target_pokemon = distiller.pokemon_df['name'].tolist()
    
    # Pas besoin d'échantillonner si on veut tout utiliser
    if sample_size < len(target_pokemon):
        np.random.seed(42)  # Pour la reproductibilité
        target_pokemon = np.random.choice(target_pokemon, size=sample_size, replace=False)
    
    # Créer un DataFrame vide pour stocker les résultats
    all_data = pd.DataFrame()
    current_total = 0
    
    # Mesurer le temps d'exécution
    start_time = time.time()
    total_batches = (len(target_pokemon) + batch_size - 1) // batch_size
    
    # Créer le répertoire pour les fichiers intermédiaires s'il n'existe pas
    os.makedirs('data/temp', exist_ok=True)
    
    # Traiter les Pokémon par lots
    for i in range(0, len(target_pokemon), batch_size):
        batch = target_pokemon[i:i+batch_size]
        batch_num = i//batch_size + 1
        
        print(f"\n=== Traitement du lot {batch_num}/{total_batches} ({len(batch)} Pokémon) ===")
        
        # Estimer le temps restant
        if batch_num > 1:
            elapsed_time = time.time() - start_time
            time_per_batch = elapsed_time / (batch_num - 1)
            remaining_batches = total_batches - batch_num + 1
            remaining_time = time_per_batch * remaining_batches
            hours = remaining_time // 3600
            minutes = (remaining_time % 3600) // 60
            print(f"Temps écoulé: {elapsed_time/60:.1f} minutes | Temps restant estimé: {int(hours)}h {int(minutes)}m")
        
        # Utiliser la méthode batch pour trouver les counters pour tout le lot
        batch_results = distiller.existing_model.find_counters_batch(batch, top_k=counters_per_pokemon)
        
        # Convertir les résultats en caractéristiques pour l'entraînement avec simulation de combats
        combat_data = []
        
        for pokemon in tqdm(batch):
            if pokemon not in batch_results or not batch_results[pokemon]:
                continue
                
            counters = batch_results[pokemon]
            for counter in counters:
                try:
                    # Créer les caractéristiques de base pour cette paire
                    base_features = distiller._create_feature_row(pokemon, counter['pokemon'])
                    win_prob = counter['win_probability']
                    
                    # Ajouter des caractéristiques supplémentaires
                    base_features['counter_has_type_advantage'] = any('super efficace' in reason for reason in counter['reasons'])
                    base_features['counter_has_stat_advantage'] = any(('def faible' in reason.lower() and 'atk haute' in reason.lower()) for reason in counter['reasons'])
                    
                    # Simuler plusieurs combats avec cette paire
                    for _ in range(simulations_per_pair):
                        # Copier les caractéristiques de base
                        features = base_features.copy()
                        
                        # Simuler le résultat du combat (1 = victoire, 0 = défaite)
                        # Un nombre aléatoire entre 0 et 1 est comparé à la probabilité de victoire
                        battle_outcome = 1 if np.random.random() < win_prob else 0
                        
                        # Ajouter le résultat et la probabilité originale
                        features['outcome'] = battle_outcome
                        features['win_probability'] = win_prob
                        
                        # Ajouter de légères variations aléatoires pour diversifier le dataset
                        # (±5% sur les statistiques numériques)
                        for stat in ['hp_ratio', 'attack_ratio', 'defense_ratio', 'sp_attack_ratio', 'sp_defense_ratio', 'speed_ratio', 'bst_ratio']:
                            if stat in features:
                                variation = 1.0 + (np.random.random() * 0.1 - 0.05)  # ±5%
                                features[stat] *= variation
                        
                        combat_data.append(features)
                except Exception as e:
                    print(f"Erreur lors de la création des caractéristiques pour {pokemon} vs {counter['pokemon']}: {str(e)}")
        
        # Convertir en DataFrame et l'ajouter aux résultats
        if combat_data:
            batch_df = pd.DataFrame(combat_data)
            current_total += len(batch_df)
            
            # Sauvegarder ce lot dans un fichier temporaire
            batch_file = f'data/temp/batch_{batch_num:03d}.csv'
            batch_df.to_csv(batch_file, index=False)
            
            print(f"Lot {batch_num} sauvegardé (+ {len(batch_df)} combats)")
            print(f"Total actuel: {current_total} combats")
            
            # Vérifier si on a atteint la cible
            if current_total >= target_count:
                print(f"Objectif atteint ({current_total} >= {target_count}). Arrêt du traitement.")
                break

        # Libérer la mémoire
        del batch_results
        del combat_data
        if 'batch_df' in locals():
            del batch_df
        import gc
        gc.collect()
    
    # Combiner tous les fichiers de lot dans un seul fichier final
    print("\nCombination de tous les fichiers intermédiaires...")
    all_batches = [f for f in os.listdir('data/temp') if f.startswith('batch_') and f.endswith('.csv')]
    all_batches.sort()  # Trier pour avoir un ordre cohérent
    
    # Créer le fichier final
    with open('data/massive_combat_dataset.csv', 'w') as outfile:
        # Écrire l'en-tête à partir du premier fichier
        if all_batches:
            with open(os.path.join('data/temp', all_batches[0]), 'r') as infile:
                header = infile.readline()
                outfile.write(header)
            
            # Écrire les données de chaque fichier
            for batch_file in tqdm(all_batches, desc="Fusion des fichiers"):
                with open(os.path.join('data/temp', batch_file), 'r') as infile:
                    # Ignorer l'en-tête
                    next(infile)
                    # Copier le reste
                    for line in infile:
                        outfile.write(line)
    
    # Vérifier la taille du fichier final
    final_lines = 0
    with open('data/massive_combat_dataset.csv', 'r') as f:
        for _ in f:
            final_lines += 1
    final_lines -= 1  # Soustraire l'en-tête
    
    # Afficher les statistiques
    execution_time = time.time() - start_time
    hours = execution_time // 3600
    minutes = (execution_time % 3600) // 60
    seconds = execution_time % 60
    
    print(f"\nGénération terminée en {int(hours)}h {int(minutes)}m {int(seconds)}s.")
    print(f"Nombre de combats générés: {final_lines}")
    print(f"Dataset sauvegardé dans: data/massive_combat_dataset.csv")
    
    return final_lines

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Génère un dataset massif avec simulation de résultats de combat.')
    parser.add_argument('--target', type=int, default=500000, help='Nombre cible de combats à générer')
    parser.add_argument('--batch', type=int, default=10, help='Taille des lots de Pokémon')
    parser.add_argument('--counters', type=int, default=100, help='Nombre de counters par Pokémon')
    parser.add_argument('--simulations', type=int, default=10, help='Nombre de simulations par paire')
    args = parser.parse_args()
    
    generate_massive_dataset(
        target_count=args.target,
        batch_size=args.batch,
        counters_per_pokemon=args.counters,
        simulations_per_pair=args.simulations
    ) 