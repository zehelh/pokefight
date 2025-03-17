import os
import sys
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
from tqdm import tqdm  # Pour afficher une barre de progression

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importer notre modèle existant
from src.models.xgboost_counter_finder_prod import XGBoostCounterFinderProd

class PokemonModelDistiller:
    """
    Classe pour générer un modèle distillé à partir du système existant.
    """
    
    def __init__(self):
        self.existing_model = None
        self.distilled_model = None
        self.pokemon_df = None
        self.matchups_df = None
        self.usage_df = None
        self.generated_data = None
    
    def load_existing_system(self, model_path='models/xgboost_counter_finder.joblib'):
        """
        Charge le système existant (modèle et données).
        """
        print("Chargement du système existant...")
        
        # Vérifier si le modèle existe
        if not os.path.exists(model_path):
            print(f"Modèle non trouvé à {model_path}")
            # Chercher dans d'autres emplacements potentiels
            possible_paths = [
                'models/xgboost_counter_finder.joblib',
                'src/models/xgboost_counter_finder.joblib',
                '../models/xgboost_counter_finder.joblib',
                '../../models/xgboost_counter_finder.joblib'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Modèle trouvé à {path}")
                    break
            else:
                print("Modèle non trouvé. Veuillez l'entraîner d'abord.")
                return False
        
        # Chemins pour les données
        data_dir = os.path.join(os.path.dirname(model_path), "..", "data")
        pokemon_path = os.path.join(data_dir, 'pokemon.csv')
        matchups_path = os.path.join(data_dir, 'pokemon_matchups.csv')
        usage_path = os.path.join(data_dir, 'pokemon_usage_stats.csv')
        
        # Vérifier les chemins alternatifs si nécessaire
        if not os.path.exists(pokemon_path):
            alt_data_dir = 'data'
            pokemon_path = os.path.join(alt_data_dir, 'pokemon.csv')
            matchups_path = os.path.join(alt_data_dir, 'pokemon_matchups.csv')
            usage_path = os.path.join(alt_data_dir, 'pokemon_usage_stats.csv')
        
        # Charger le modèle existant
        self.existing_model = XGBoostCounterFinderProd(model_path)
        self.existing_model.load_data(pokemon_path, matchups_path, usage_path)
        
        # Stocker les dataframes pour une utilisation ultérieure
        self.pokemon_df = self.existing_model.pokemon_df
        self.matchups_df = self.existing_model.matchups_df
        self.usage_df = self.existing_model.usage_df
        
        print(f"Système existant chargé avec succès. {len(self.pokemon_df)} Pokémon disponibles.")
        return True
    
    def generate_combat_data(self, sample_size=None, test_mode=True):
        """
        Génère des données de combat en utilisant le système existant.
        
        Args:
            sample_size: Nombre de Pokémon à échantillonner (None pour tous)
            test_mode: Si True, génère seulement pour Tyranitar et Tyrantrum
        """
        if self.existing_model is None:
            print("Le système existant n'est pas chargé. Veuillez d'abord appeler load_existing_system().")
            return
        
        print("Génération des données de combat...")
        
        # Liste des Pokémon cibles
        if test_mode:
            target_pokemon = ['Tyranitar', 'Tyrantrum']
        else:
            target_pokemon = self.pokemon_df['name'].tolist()
            
            # Échantillonner si nécessaire
            if sample_size is not None and sample_size < len(target_pokemon):
                np.random.seed(42)  # Pour la reproductibilité
                target_pokemon = np.random.choice(target_pokemon, size=sample_size, replace=False)
        
        # Liste pour stocker les données générées
        combat_data = []
        
        # Générer les données
        for pokemon in tqdm(target_pokemon):
            try:
                # Trouver les contre-Pokémon pour ce Pokémon
                counters = self.existing_model.find_counters(pokemon, top_k=20)
                
                # Ajouter les données pour chaque contre-Pokémon
                for counter in counters:
                    row = self._create_feature_row(pokemon, counter['pokemon'])
                    row['win_probability'] = counter['win_probability']
                    
                    # Ajouter d'autres caractéristiques
                    row['counter_has_type_advantage'] = any('super efficace' in reason for reason in counter['reasons'])
                    row['counter_has_stat_advantage'] = any(('def faible' in reason.lower() and 'atk haute' in reason.lower()) for reason in counter['reasons'])
                    
                    combat_data.append(row)
            except Exception as e:
                print(f"Erreur lors du traitement de {pokemon}: {str(e)}")
        
        # Convertir en DataFrame
        self.generated_data = pd.DataFrame(combat_data)
        
        print(f"Données générées avec succès. {len(self.generated_data)} combats générés.")
        
        # Sauvegarder les données générées
        if not test_mode:
            self.generated_data.to_csv('data/generated_combat_data.csv', index=False)
            print("Données sauvegardées dans data/generated_combat_data.csv")
        
        return self.generated_data
    
    def _create_feature_row(self, target_pokemon, counter_pokemon):
        """
        Crée une ligne de caractéristiques pour une paire (target, counter).
        """
        # Obtenir les données des Pokémon
        target_data = self.pokemon_df[self.pokemon_df['name'] == target_pokemon].iloc[0]
        counter_data = self.pokemon_df[self.pokemon_df['name'] == counter_pokemon].iloc[0]
        
        # Créer un dictionnaire pour stocker les caractéristiques
        row = {
            'target_pokemon': target_pokemon,
            'counter_pokemon': counter_pokemon,
            
            # Types
            'target_type1': target_data['type1'],
            'target_type2': target_data['type2'] if pd.notna(target_data['type2']) else '',
            'counter_type1': counter_data['type1'],
            'counter_type2': counter_data['type2'] if pd.notna(counter_data['type2']) else '',
            
            # Statistiques
            'target_hp': target_data['hp'],
            'target_attack': target_data['attack'],
            'target_defense': target_data['defense'],
            'target_sp_attack': target_data['sp_attack'],
            'target_sp_defense': target_data['sp_defense'],
            'target_speed': target_data['speed'],
            
            'counter_hp': counter_data['hp'],
            'counter_attack': counter_data['attack'],
            'counter_defense': counter_data['defense'],
            'counter_sp_attack': counter_data['sp_attack'],
            'counter_sp_defense': counter_data['sp_defense'],
            'counter_speed': counter_data['speed'],
            
            # Ratios
            'hp_ratio': counter_data['hp'] / target_data['hp'],
            'attack_ratio': counter_data['attack'] / target_data['attack'],
            'defense_ratio': counter_data['defense'] / target_data['defense'],
            'sp_attack_ratio': counter_data['sp_attack'] / target_data['sp_attack'],
            'sp_defense_ratio': counter_data['sp_defense'] / target_data['sp_defense'],
            'speed_ratio': counter_data['speed'] / target_data['speed'],
            
            # BST (Base Stat Total)
            'target_bst': target_data['hp'] + target_data['attack'] + target_data['defense'] + target_data['sp_attack'] + target_data['sp_defense'] + target_data['speed'],
            'counter_bst': counter_data['hp'] + counter_data['attack'] + counter_data['defense'] + counter_data['sp_attack'] + counter_data['sp_defense'] + counter_data['speed'],
            'bst_ratio': (counter_data['hp'] + counter_data['attack'] + counter_data['defense'] + counter_data['sp_attack'] + counter_data['sp_defense'] + counter_data['speed']) / (target_data['hp'] + target_data['attack'] + target_data['defense'] + target_data['sp_attack'] + target_data['sp_defense'] + target_data['speed']),
            
            # Type effectiveness
            'counter_against_target_type1': self._get_type_effectiveness(counter_data, target_data['type1']),
            'counter_against_target_type2': self._get_type_effectiveness(counter_data, target_data['type2']) if pd.notna(target_data['type2']) else 1.0,
            'target_against_counter_type1': self._get_type_effectiveness(target_data, counter_data['type1']),
            'target_against_counter_type2': self._get_type_effectiveness(target_data, counter_data['type2']) if pd.notna(counter_data['type2']) else 1.0,
        }
        
        return row
    
    def _get_type_effectiveness(self, attacker_data, defender_type):
        """
        Calcule l'efficacité de type de l'attaquant contre le type du défenseur.
        """
        if pd.isna(defender_type) or defender_type == '':
            return 1.0
        
        # Simplification: utiliser les colonnes 'against_X' du DataFrame
        effectiveness = 1.0
        
        # Type 1 de l'attaquant
        type1_col = f'against_{defender_type.lower()}'
        if type1_col in attacker_data:
            effectiveness *= attacker_data[type1_col]
        
        # Type 2 de l'attaquant (s'il existe)
        if pd.notna(attacker_data['type2']) and attacker_data['type2'] != '':
            type2_col = f'against_{defender_type.lower()}'
            if type2_col in attacker_data:
                effectiveness *= attacker_data[type2_col]
        
        return effectiveness
    
    def train_distilled_model(self):
        """
        Entraîne un modèle distillé sur les données générées.
        """
        if self.generated_data is None or len(self.generated_data) == 0:
            print("Pas de données générées. Veuillez d'abord appeler generate_combat_data().")
            return
        
        print("Entraînement du modèle distillé...")
        
        # Préparer les données
        X = self.generated_data.drop(['target_pokemon', 'counter_pokemon', 'win_probability'], axis=1)
        y = self.generated_data['win_probability']
        
        # Encodage one-hot pour les variables catégorielles
        X = pd.get_dummies(X, columns=['target_type1', 'target_type2', 'counter_type1', 'counter_type2'])
        
        # Diviser en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraîner un modèle plus simple et plus rapide
        start_time = time.time()
        
        # Hyperparamètres plus simples pour le modèle distillé
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        self.distilled_model = xgb.XGBRegressor(**params)
        self.distilled_model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Évaluer le modèle
        y_pred = self.distilled_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print(f"Modèle distillé entraîné en {training_time:.2f} secondes")
        print(f"RMSE: {rmse:.4f}")
        
        # Sauvegarder le modèle
        joblib.dump(self.distilled_model, 'models/distilled_counter_finder.joblib')
        print("Modèle distillé sauvegardé dans models/distilled_counter_finder.joblib")
        
        return self.distilled_model
    
    def test_specific_pokemon(self, pokemon_name, top_k=5):
        """
        Teste le modèle distillé sur un Pokémon spécifique et compare avec le système existant.
        """
        if self.distilled_model is None:
            print("Le modèle distillé n'est pas entraîné. Veuillez d'abord appeler train_distilled_model().")
            return
        
        print(f"\nTest pour {pokemon_name}:")
        
        # Obtenir les counters avec le système existant
        print("\nCounters avec le système existant:")
        existing_counters = self.existing_model.find_counters(pokemon_name, top_k=top_k)
        
        for i, counter in enumerate(existing_counters):
            print(f"{i+1}. {counter['pokemon']} - {counter['win_probability']:.2%}")
        
        # Obtenir les counters avec le modèle distillé
        print("\nCounters avec le modèle distillé:")
        distilled_counters = self._find_counters_with_distilled_model(pokemon_name, top_k=top_k)
        
        for i, counter in enumerate(distilled_counters):
            raw_prob = counter.get('raw_win_probability', 0)
            adjusted_prob = counter['win_probability']
            print(f"{i+1}. {counter['pokemon']} - {adjusted_prob:.2%} (brut: {raw_prob:.2%})")
        
        # Calculer le chevauchement des deux listes
        existing_names = [c['pokemon'] for c in existing_counters]
        distilled_names = [c['pokemon'] for c in distilled_counters]
        
        overlap = set(existing_names).intersection(set(distilled_names))
        overlap_percentage = len(overlap) / top_k * 100
        
        print(f"\nChevauchement des listes: {len(overlap)}/{top_k} ({overlap_percentage:.1f}%)")
        print(f"Pokémon communs: {', '.join(overlap)}")
        
        return existing_counters, distilled_counters
    
    def _find_counters_with_distilled_model(self, pokemon_name, top_k=5):
        """
        Trouve les meilleurs contre-Pokémon en utilisant le modèle distillé.
        """
        # Vérifier si le Pokémon existe
        if pokemon_name not in self.pokemon_df['name'].values:
            print(f"Pokémon {pokemon_name} non trouvé")
            return []
        
        # Générer des caractéristiques pour tous les contre-Pokémon potentiels
        counter_candidates = []
        
        for counter_name in self.pokemon_df['name']:
            if counter_name == pokemon_name:
                continue
            
            # Vérifier si le Pokémon a plus de 300 stats totales
            counter_data = self.pokemon_df[self.pokemon_df['name'] == counter_name].iloc[0]
            total_stats = (
                counter_data['hp'] + counter_data['attack'] + counter_data['defense'] + 
                counter_data['sp_attack'] + counter_data['sp_defense'] + counter_data['speed']
            )
            
            if total_stats < 300:
                continue
            
            # Créer les caractéristiques
            features = self._create_feature_row(pokemon_name, counter_name)
            counter_candidates.append((counter_name, features))
        
        # Convertir en DataFrame pour la prédiction
        features_df = pd.DataFrame([c[1] for c in counter_candidates])
        
        # Préparer les caractéristiques
        X = features_df.drop(['target_pokemon', 'counter_pokemon'], axis=1)
        X = pd.get_dummies(X, prefix_sep='_', columns=['target_type1', 'target_type2', 'counter_type1', 'counter_type2'])
        
        # Aligner les colonnes avec celles utilisées pour l'entraînement
        missing_cols = set(self.distilled_model.feature_names_in_) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[self.distilled_model.feature_names_in_]
        
        # Prédire les probabilités de victoire
        win_probabilities = self.distilled_model.predict(X)
        
        # Créer les objets counter
        counters = []
        for i, (counter_name, _) in enumerate(counter_candidates):
            # Obtenir les données des Pokémon
            target_data = self.pokemon_df[self.pokemon_df['name'] == pokemon_name].iloc[0]
            counter_data = self.pokemon_df[self.pokemon_df['name'] == counter_name].iloc[0]
            
            # Générer les raisons
            reasons = self._generate_reasons(pokemon_name, counter_name, win_probabilities[i])
            
            # Ajuster la probabilité de victoire
            raw_win_prob = win_probabilities[i]
            adjusted_win_prob = self._adjust_win_probability(
                raw_win_prob, 
                counter_data, 
                target_data,
                reasons
            )
            
            counter_data = {
                'pokemon': counter_name,
                'win_probability': adjusted_win_prob,  # Utiliser la probabilité ajustée
                'raw_win_probability': raw_win_prob,   # Garder la probabilité brute
                'source': 'Modèle Distillé',
                'reasons': reasons
            }
            counters.append(counter_data)
        
        # Trier par probabilité de victoire ajustée
        counters.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return counters[:top_k]
    
    def _adjust_win_probability(self, original_prob, counter_pokemon, target_pokemon, reasons):
        """
        Ajuste la probabilité de victoire en fonction de divers facteurs.
        """
        # Transformation de base pour augmenter les valeurs basses
        adjusted_prob = np.sqrt(original_prob)
        
        # Bonus pour les avantages de type
        type_bonus = 0.0
        for reason in reasons:
            if 'super efficace' in reason:
                if 'x4' in reason or 'x4.0' in reason:
                    type_bonus += 0.15  # Bonus plus important pour les efficacités x4
                elif 'x2' in reason or 'x2.0' in reason:
                    type_bonus += 0.08  # Bonus standard pour les efficacités x2
            if 'Avantage de type double' in reason:
                type_bonus += 0.05
        
        # Bonus pour les avantages statistiques
        stat_bonus = 0.0
        
        # Vérifier si le counter a des attaques fortes contre une défense faible
        if counter_pokemon['attack'] > target_pokemon['defense'] + 30:
            stat_bonus += 0.05
        
        if counter_pokemon['sp_attack'] > target_pokemon['sp_defense'] + 30:
            stat_bonus += 0.05
        
        # Bonus de vitesse
        if counter_pokemon['speed'] > target_pokemon['speed'] + 20:
            stat_bonus += 0.03
        
        # Bonus BST (Base Stat Total)
        counter_bst = (
            counter_pokemon['hp'] + counter_pokemon['attack'] + counter_pokemon['defense'] + 
            counter_pokemon['sp_attack'] + counter_pokemon['sp_defense'] + counter_pokemon['speed']
        )
        target_bst = (
            target_pokemon['hp'] + target_pokemon['attack'] + target_pokemon['defense'] + 
            target_pokemon['sp_attack'] + target_pokemon['sp_defense'] + target_pokemon['speed']
        )
        
        if counter_bst > target_bst + 100:
            stat_bonus += 0.05
        
        # Appliquer les bonus
        adjusted_prob += type_bonus + stat_bonus
        
        # Limiter la probabilité entre 0.4 et 0.95
        adjusted_prob = max(0.4, min(0.95, adjusted_prob))
        
        return adjusted_prob
    
    def _generate_reasons(self, target_pokemon, counter_pokemon, win_probability):
        """
        Génère des raisons pour l'efficacité du contre-Pokémon.
        """
        reasons = []
        
        # Obtenir les données des Pokémon
        target_data = self.pokemon_df[self.pokemon_df['name'] == target_pokemon].iloc[0]
        counter_data = self.pokemon_df[self.pokemon_df['name'] == counter_pokemon].iloc[0]
        
        # Type
        type_text = counter_data['type1']
        if pd.notna(counter_data['type2']):
            type_text += f"/{counter_data['type2']}"
        reasons.append(f"Types: {type_text}")
        
        # Avantages statistiques
        for stat, label in [
            ('hp', 'PV'), ('attack', 'Attaque'), ('defense', 'Défense'),
            ('sp_attack', 'Attaque Spéciale'), ('sp_defense', 'Défense Spéciale'),
            ('speed', 'Vitesse')
        ]:
            if counter_data[stat] > target_data[stat] + 20:
                reasons.append(f"Meilleur en {label} (+{counter_data[stat] - target_data[stat]})")
        
        # Avantages de type (simplifiés)
        effectiveness = self._check_type_advantages(counter_data, target_data)
        if effectiveness > 1:
            reasons.append(f"Avantage de type (efficacité x{effectiveness:.1f})")
        
        # Si la probabilité est très élevée
        if win_probability > 0.8:
            reasons.append("Counter très efficace selon l'historique des combats")
        
        return reasons
    
    def _check_type_advantages(self, attacker, defender):
        """
        Vérifie les avantages de type entre deux Pokémon (version simplifiée).
        """
        # Type de l'attaquant
        attack_type1 = attacker['type1'].lower()
        attack_type2 = attacker['type2'].lower() if pd.notna(attacker['type2']) else None
        
        # Type du défenseur
        defend_type1 = defender['type1'].lower()
        defend_type2 = defender['type2'].lower() if pd.notna(defender['type2']) else None
        
        # Calculer l'efficacité
        effectiveness = 1.0
        
        # Type 1 de l'attaquant contre les types du défenseur
        type1_vs_defend1 = self._get_single_type_effectiveness(attack_type1, defend_type1)
        effectiveness *= type1_vs_defend1
        
        if defend_type2:
            type1_vs_defend2 = self._get_single_type_effectiveness(attack_type1, defend_type2)
            effectiveness *= type1_vs_defend2
        
        # Type 2 de l'attaquant (s'il existe) contre les types du défenseur
        if attack_type2:
            type2_vs_defend1 = self._get_single_type_effectiveness(attack_type2, defend_type1)
            effectiveness *= type2_vs_defend1
            
            if defend_type2:
                type2_vs_defend2 = self._get_single_type_effectiveness(attack_type2, defend_type2)
                effectiveness *= type2_vs_defend2
        
        return effectiveness
    
    def _get_single_type_effectiveness(self, attack_type, defend_type):
        """
        Obtient l'efficacité d'un seul type d'attaque contre un seul type de défense.
        """
        # Colonnes d'efficacité dans le DataFrame Pokémon
        col_name = f'against_{defend_type}'
        
        # Chercher dans le DataFrame des types
        type_row = self.pokemon_df[self.pokemon_df['type1'] == attack_type].iloc[0] if any(self.pokemon_df['type1'] == attack_type) else None
        
        if type_row is not None and col_name in type_row:
            return type_row[col_name]
        
        # Valeur par défaut
        return 1.0

def main():
    # Créer une instance de la classe
    distiller = PokemonModelDistiller()
    
    # Charger le système existant
    if not distiller.load_existing_system():
        return
    
    # Générer des données pour Tyranitar et Tyrantrum
    print("\n=== Génération des données de test ===")
    distiller.generate_combat_data(test_mode=True)
    
    # Entraîner le modèle distillé
    print("\n=== Entraînement du modèle distillé ===")
    distiller.train_distilled_model()
    
    # Tester sur Tyranitar et Tyrantrum
    print("\n=== Test du modèle distillé ===")
    distiller.test_specific_pokemon('Tyranitar', top_k=5)
    distiller.test_specific_pokemon('Tyrantrum', top_k=5)
    
    # Mesurer la vitesse d'exécution
    print("\n=== Comparaison des vitesses d'exécution ===")
    
    # Vitesse du système existant
    start_time = time.time()
    distiller.existing_model.find_counters('Tyranitar', top_k=5)
    existing_time = time.time() - start_time
    
    # Vitesse du modèle distillé
    start_time = time.time()
    distiller._find_counters_with_distilled_model('Tyranitar', top_k=5)
    distilled_time = time.time() - start_time
    
    print(f"Temps d'exécution du système existant: {existing_time:.4f} secondes")
    print(f"Temps d'exécution du modèle distillé: {distilled_time:.4f} secondes")
    print(f"Accélération: {existing_time / distilled_time:.2f}x")
    
    print("\n=== Terminer ===")
    print("Pour générer des données pour tous les Pokémon et entraîner un modèle complet:")
    print("1. Appelez distiller.generate_combat_data(test_mode=False)")
    print("2. Attendez que la génération soit terminée (cela peut prendre du temps)")
    print("3. Appelez distiller.train_distilled_model()")

if __name__ == "__main__":
    main() 