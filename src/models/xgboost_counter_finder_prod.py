import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os
import logging
import traceback
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score

class XGBoostCounterFinderProd:
    """
    Version de production du modèle XGBoost pour trouver les meilleurs contre-Pokémon.
    """
    
    def __init__(self, model_path=None):
        self.pokemon_df = None
        self.matchups_df = None
        self.usage_df = None
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_data(self, pokemon_path, matchups_path, usage_path):
        """Charge les données nécessaires pour la prédiction."""
        print(f"Chargement des données depuis:")
        print(f"- Pokemon: {pokemon_path}")
        print(f"- Matchups: {matchups_path}")
        print(f"- Usage: {usage_path}")
        
        self.pokemon_df = pd.read_csv(pokemon_path)
        self.matchups_df = pd.read_csv(matchups_path)
        self.usage_df = pd.read_csv(usage_path)
        
        print("✓ Données chargées avec succès")
        return True
    
    def train(self, save_path=None):
        """
        Entraîne le modèle avec les hyperparamètres optimaux pré-définis.
        """
        start_time = time.time()
        
        # Préparer les données
        X, y = self.prepare_features()
        
        # Diviser en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Définir les colonnes catégorielles et numériques
        categorical_cols = [col for col in X.columns if col.endswith('_type1') or col.endswith('_type2')]
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Créer le préprocesseur
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        # Hyperparamètres optimaux
        best_params = {
            'n_estimators': 295,
            'max_depth': 8,
            'learning_rate': 0.01,
            'subsample': 0.6,
            'colsample_bytree': 0.98,
            'gamma': 0.95,
            'min_child_weight': 10,
            'reg_alpha': 0.95,
            'reg_lambda': 0.08,
            'scale_pos_weight': 0.54,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        # Créer et entraîner le modèle
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', xgb.XGBClassifier(**best_params))
        ])
        
        print("Entraînement du modèle...")
        self.model.fit(X_train, y_train)
        
        # Évaluer le modèle
        print("Évaluation du modèle...")
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculer les métriques
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'training_time': time.time() - start_time
        }
        
        if save_path:
            self.save_model(save_path)
            print(f"Modèle sauvegardé dans {save_path}")
        
        return metrics
    
    def prepare_features(self):
        """Prépare les caractéristiques pour l'entraînement."""
        features_df = self.matchups_df.copy()
        
        # Standardiser le nom de la colonne Win Rate
        if 'Win Rate' in features_df.columns:
            features_df['Win_Rate'] = features_df['Win Rate'].str.rstrip('%').astype(float) / 100
        elif 'win_rate' in features_df.columns:
            features_df['Win_Rate'] = features_df['win_rate']
        else:
            features_df['Win_Rate'] = 0.5
        
        # Ajouter les caractéristiques des Pokémon
        features_df = self._add_pokemon_features(features_df, 'Pokemon1', 'p1')
        features_df = self._add_pokemon_features(features_df, 'Pokemon2', 'p2')
        
        # Calculer les caractéristiques dérivées
        features_df = self._calculate_derived_features(features_df)
        
        # Sélectionner les colonnes pour l'entraînement
        feature_cols = [col for col in features_df.columns if col.startswith('p1_') or col.startswith('p2_') or col.startswith('ratio_')]
        self.feature_names = feature_cols
        
        X = features_df[self.feature_names]
        y = (features_df['Win_Rate'] > 0.5).astype(int)
        
        return X, y
    
    def _add_pokemon_features(self, df, pokemon_col, prefix):
        """Ajoute les caractéristiques des Pokémon au DataFrame."""
        result_df = df.copy()
        
        for i, row in result_df.iterrows():
            pokemon_name = row[pokemon_col]
            pokemon_row = self.pokemon_df[self.pokemon_df['name'] == pokemon_name]
            
            if not pokemon_row.empty:
                pokemon = pokemon_row.iloc[0]
                
                # Ajouter les types
                result_df.at[i, f'{prefix}_type1'] = pokemon['type1']
                result_df.at[i, f'{prefix}_type2'] = pokemon['type2'] if pd.notna(pokemon['type2']) else ''
                
                # Ajouter les statistiques
                result_df.at[i, f'{prefix}_hp'] = pokemon['hp']
                result_df.at[i, f'{prefix}_attack'] = pokemon['attack']
                result_df.at[i, f'{prefix}_defense'] = pokemon['defense']
                result_df.at[i, f'{prefix}_sp_attack'] = pokemon['sp_attack']
                result_df.at[i, f'{prefix}_sp_defense'] = pokemon['sp_defense']
                result_df.at[i, f'{prefix}_speed'] = pokemon['speed']
                
                # Calculer BST (Base Stat Total)
                result_df.at[i, f'{prefix}_bst'] = (
                    pokemon['hp'] + pokemon['attack'] + pokemon['defense'] + 
                    pokemon['sp_attack'] + pokemon['sp_defense'] + pokemon['speed']
                )
                
                # Ajouter les efficacités de type
                for col in pokemon.index:
                    if col.startswith('against_'):
                        result_df.at[i, f'{prefix}_{col}'] = pokemon[col]
            else:
                # Valeurs par défaut si le Pokémon n'est pas trouvé
                result_df.at[i, f'{prefix}_type1'] = 'Normal'
                result_df.at[i, f'{prefix}_type2'] = ''
                result_df.at[i, f'{prefix}_hp'] = 50
                result_df.at[i, f'{prefix}_attack'] = 50
                result_df.at[i, f'{prefix}_defense'] = 50
                result_df.at[i, f'{prefix}_sp_attack'] = 50
                result_df.at[i, f'{prefix}_sp_defense'] = 50
                result_df.at[i, f'{prefix}_speed'] = 50
                result_df.at[i, f'{prefix}_bst'] = 300
        
        return result_df
    
    def _calculate_derived_features(self, df):
        """Calcule les caractéristiques dérivées."""
        result_df = df.copy()
        
        # Calculer les ratios
        result_df['ratio_speed'] = result_df['p2_speed'] / (result_df['p1_speed'] + 1)
        result_df['ratio_attack_vs_defense'] = result_df['p2_attack'] / (result_df['p1_defense'] + 1)
        result_df['ratio_sp_attack_vs_sp_defense'] = result_df['p2_sp_attack'] / (result_df['p1_sp_defense'] + 1)
        result_df['ratio_bst'] = result_df['p2_bst'] / (result_df['p1_bst'] + 1)
        
        return result_df
    
    def find_counters(self, pokemon_name, top_k=5):
        """Trouve les meilleurs contre-Pokémon avec un filtrage amélioré par type."""
        print(f"Recherche de contre-Pokémon pour {pokemon_name}...")
        
        # Vérifier si le Pokémon existe
        target_pokemon = self.pokemon_df[self.pokemon_df['name'] == pokemon_name]
        if target_pokemon.empty:
            print(f"Pokémon {pokemon_name} non trouvé")
            return []
        
        target_pokemon = target_pokemon.iloc[0]
        
        # Liste pour stocker les contre-Pokémon potentiels
        all_counters = []
        super_effective_counters = []
        other_counters = []
        
        # Parcourir tous les Pokémon
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
            
            try:
                # Vérifier les avantages de type
                type_advantages, has_super_effective = self._check_type_advantages(counter_pokemon, target_pokemon)
                
                # Créer un DataFrame pour la prédiction
                matchup = pd.DataFrame({
                    'Pokemon1': [counter_name],
                    'Pokemon2': [pokemon_name],
                    'Win_Rate': [0.5]
                })
                
                # Ajouter les caractéristiques
                matchup = self._add_pokemon_features(matchup, 'Pokemon1', 'p1')
                matchup = self._add_pokemon_features(matchup, 'Pokemon2', 'p2')
                
                # Calculer les caractéristiques dérivées
                matchup = self._calculate_derived_features(matchup)
                
                # Sélectionner les caractéristiques
                X = matchup[self.feature_names]
                
                # Prédire la probabilité de victoire
                win_prob = self.model.predict_proba(X)[0, 1]
                
                # Bonus pour les Pokémon avec avantage de type super efficace
                if has_super_effective:
                    # Augmenter significativement la probabilité pour les Pokémon avec avantage de type super efficace
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
                
                # Ajouter à la liste appropriée
                if has_super_effective:
                    super_effective_counters.append(counter_data)
                else:
                    other_counters.append(counter_data)
                
            except Exception as e:
                print(f"Erreur lors de la prédiction pour {counter_name}: {str(e)}")
                continue
        
        # Trier les deux listes par probabilité de victoire
        super_effective_counters.sort(key=lambda x: x['win_probability'], reverse=True)
        other_counters.sort(key=lambda x: x['win_probability'], reverse=True)
        
        # Prioriser les contre-Pokémon avec des attaques super efficaces
        # Prendre au moins 70% des contre-Pokémon avec des attaques super efficaces
        min_super_effective = min(len(super_effective_counters), int(top_k * 0.7))
        remaining_slots = top_k - min_super_effective
        
        # Combiner les listes
        final_counters = super_effective_counters[:min_super_effective]
        if remaining_slots > 0 and other_counters:
            final_counters.extend(other_counters[:remaining_slots])
        
        # Si nous n'avons pas assez de contre-Pokémon avec des attaques super efficaces,
        # compléter avec d'autres contre-Pokémon
        if len(final_counters) < top_k and len(other_counters) > remaining_slots:
            additional_needed = top_k - len(final_counters)
            final_counters.extend(other_counters[remaining_slots:remaining_slots + additional_needed])
        
        # Trier à nouveau par probabilité de victoire
        final_counters.sort(key=lambda x: x['win_probability'], reverse=True)
        
        # Nettoyer les résultats avant de les retourner
        for counter in final_counters:
            counter.pop('raw_win_probability', None)
            counter.pop('has_super_effective', None)
        
        print(f"Nombre de contre-Pokémon trouvés: {len(final_counters)}")
        print(f"Dont {min_super_effective} avec attaques super efficaces")
        
        return final_counters
    
    def _check_type_advantages(self, attacker, defender):
        """Vérifie les avantages de type entre deux Pokémon de façon plus précise et stricte."""
        advantages = []
        
        # Table complète des efficacités de type
        type_chart = {
            "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
            "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
            "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
            "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
            "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
            "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
            "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
            "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
            "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
            "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
            "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
            "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
            "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
            "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
            "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
            "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
            "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
            "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5}
        }
        
        # Vérifier les avantages de type offensifs (attacker → defender)
        attacker_types = [attacker['type1'].lower()]
        if pd.notna(attacker['type2']):
            attacker_types.append(attacker['type2'].lower())
        
        defender_types = [defender['type1'].lower()]
        if pd.notna(defender['type2']):
            defender_types.append(defender['type2'].lower())
        
        # Calculer l'efficacité offensive totale pour chaque type
        offensive_effectiveness = {}
        has_super_effective_attack = False
        
        for attack_type in attacker_types:
            if attack_type in type_chart:
                # Calculer l'efficacité combinée contre tous les types du défenseur
                combined_effectiveness = 1.0
                for defend_type in defender_types:
                    if defend_type in type_chart[attack_type]:
                        combined_effectiveness *= type_chart[attack_type][defend_type]
                
                # Si le type est super efficace contre la combinaison de types du défenseur
                if combined_effectiveness > 1:
                    has_super_effective_attack = True
                    offensive_effectiveness[attack_type] = combined_effectiveness
                    advantages.append(f"Les attaques {attack_type.capitalize()} sont super efficaces (x{combined_effectiveness:.1f})")
        
        # Calculer l'efficacité défensive (résistances du counter)
        defensive_advantages = []
        for attack_type in defender_types:
            if attack_type in type_chart:
                for defend_type in attacker_types:
                    if defend_type in type_chart[attack_type]:
                        if type_chart[attack_type][defend_type] < 1:
                            defensive_advantages.append(f"Résistance aux attaques {attack_type.capitalize()}")
        
        # Ajouter les avantages défensifs (sans doublons)
        for advantage in list(dict.fromkeys(defensive_advantages)):
            advantages.append(advantage)
        
        # Ajouter un avantage global si le counter a un avantage net
        if has_super_effective_attack and len(defensive_advantages) > 0:
            advantages.append("Avantage de type double (offensif et défensif)")
        elif has_super_effective_attack:
            advantages.append("Avantage de type offensif")
        
        return advantages, has_super_effective_attack
    
    def _calculate_type_advantage(self, attacker, defender):
        """Calcule l'avantage de type entre deux Pokémon."""
        # Simplification pour l'API
        return 1.0
    
    def save_model(self, path):
        """Sauvegarde le modèle."""
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
    
    def load_model(self, path):
        """Charge un modèle pré-entraîné."""
        self.model = joblib.load(path)
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