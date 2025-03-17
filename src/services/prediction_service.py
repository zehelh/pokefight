import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class PredictionService:
    """Service de prédiction pour les contre-Pokémon utilisant le modèle optimisé."""
    
    def __init__(self, model_path='models/final_model_complete_dataset.joblib', 
                 metadata_path='models/final_model_complete_dataset_metadata.joblib'):
        """
        Initialise le service de prédiction.
        
        Args:
            model_path: Chemin vers le modèle entraîné
            metadata_path: Chemin vers les métadonnées du modèle
        """
        print(f"Chargement du modèle depuis {model_path}...")
        self.model = joblib.load(model_path)
        
        print(f"Chargement des métadonnées depuis {metadata_path}...")
        self.metadata = joblib.load(metadata_path)
        
        self.expected_features = self.metadata.get('features', [])
        self.scaler = self.metadata.get('scaler', None)
        
        self.feature_names = self.metadata.get('input_features')
        if self.feature_names is None or len(self.feature_names) == 0:
            print("ATTENTION: input_features non trouvé dans les métadonnées, tentative de récupération depuis le modèle...")
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
                print(f"Récupération de {len(self.feature_names)} noms de features depuis le modèle")
            else:
                print("ERREUR: Impossible de récupérer les noms de features. Le service pourrait ne pas fonctionner correctement.")
                self.feature_names = []
        
        self.feature_importance = self.metadata.get('feature_importance')
        
        self.hyperparams = self.metadata.get('hyperparameters')
        
        print(f"Modèle chargé avec succès - Accuracy: {self.metadata['metrics']['accuracy']:.4f}, AUC: {self.metadata['metrics']['auc_roc']:.4f}")
        print(f"Nombre de features attendues: {len(self.expected_features)}")
        
        if self.feature_names and len(self.feature_names) > 0:
            print(f"Exemples de features: {self.feature_names[:10]}")
    
    def prepare_features(self, target_pokemon: Dict[str, Any], counter_pokemon: Dict[str, Any]) -> pd.DataFrame:
        """
        Prépare les caractéristiques pour la prédiction.
        
        Args:
            target_pokemon: Dictionnaire contenant les données du Pokémon cible
            counter_pokemon: Dictionnaire contenant les données du Pokémon counter
            
        Returns:
            DataFrame contenant les caractéristiques formatées pour le modèle
        """
        # Créer un dictionnaire pour stocker les caractéristiques
        features = {}
        
        # Ajouter les statistiques de base pour les deux Pokémon
        for stat in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']:
            features[f'target_{stat}'] = float(target_pokemon.get(stat, 0))
            features[f'counter_{stat}'] = float(counter_pokemon.get(stat, 0))
        
        # Calculer le BST (Base Stat Total) pour les deux Pokémon
        target_bst = sum([float(target_pokemon.get(stat, 0)) for stat in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']])
        counter_bst = sum([float(counter_pokemon.get(stat, 0)) for stat in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']])
        
        features['target_bst'] = target_bst
        features['counter_bst'] = counter_bst
        
        # Ajouter les types
        features['target_type1'] = target_pokemon.get('type1', 'normal')
        features['target_type2'] = target_pokemon.get('type2', 'none')
        features['counter_type1'] = counter_pokemon.get('type1', 'normal')
        features['counter_type2'] = counter_pokemon.get('type2', 'none')
        
        # Calculer les ratios (caractéristiques dérivées)
        features['hp_ratio'] = features['counter_hp'] / features['target_hp'] if features['target_hp'] > 0 else 1.0
        features['attack_ratio'] = features['counter_attack'] / features['target_attack'] if features['target_attack'] > 0 else 1.0
        features['defense_ratio'] = features['counter_defense'] / features['target_defense'] if features['target_defense'] > 0 else 1.0
        features['sp_attack_ratio'] = features['counter_sp_attack'] / features['target_sp_attack'] if features['target_sp_attack'] > 0 else 1.0
        features['sp_defense_ratio'] = features['counter_sp_defense'] / features['target_sp_defense'] if features['target_sp_defense'] > 0 else 1.0
        features['speed_ratio'] = features['counter_speed'] / features['target_speed'] if features['target_speed'] > 0 else 1.0
        features['bst_ratio'] = counter_bst / target_bst if target_bst > 0 else 1.0
        
        # Avantage de vitesse
        features['has_speed_advantage'] = 1 if features['counter_speed'] > features['target_speed'] else 0
        features['speed_advantage'] = features['counter_speed'] - features['target_speed']
        features['speed_advantage_squared'] = features['speed_advantage'] ** 2
        features['log_speed_ratio'] = np.log(features['speed_ratio']) if features['speed_ratio'] > 0 else 0
        
        # Interactions entre les statistiques
        features['speed_offensive_interaction'] = features['counter_speed'] * (features['counter_attack'] + features['counter_sp_attack']) / 2
        
        # Caractéristique combinée type et vitesse
        features['type_and_speed'] = 1 if (features['has_speed_advantage'] == 1) else 0
        
        # Catégoriser le ratio BST
        if features['bst_ratio'] < 0.8:
            features['bst_ratio_bucket'] = 'low'
        elif features['bst_ratio'] < 1.0:
            features['bst_ratio_bucket'] = 'medium_low'
        elif features['bst_ratio'] < 1.2:
            features['bst_ratio_bucket'] = 'medium_high'
        else:
            features['bst_ratio_bucket'] = 'high'
        
        # Gérer les valeurs infinies ou NaN
        for key, value in features.items():
            if isinstance(value, (int, float)):
                if np.isinf(value) or np.isnan(value):
                    features[key] = 0.0
        
        # Créer un DataFrame à partir des caractéristiques
        features_df = pd.DataFrame([features])
        
        # Appliquer le one-hot encoding pour les variables catégorielles
        for col in ['target_type1', 'target_type2', 'counter_type1', 'counter_type2', 'bst_ratio_bucket']:
            if col in features_df.columns:
                # Créer des colonnes one-hot pour chaque valeur unique
                for val in self.metadata.get('categorical_values', {}).get(col, []):
                    col_name = f"{col}_{val}"
                    if col_name in self.expected_features:
                        features_df[col_name] = (features_df[col] == val).astype(int)
        
        # S'assurer que toutes les caractéristiques attendues sont présentes
        for feature in self.expected_features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Ne conserver que les caractéristiques attendues par le modèle
        features_df = features_df[self.expected_features]
        
        # Normaliser les caractéristiques numériques si un scaler est disponible
        if self.scaler:
            numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns
            features_df[numeric_features] = self.scaler.transform(features_df[numeric_features])
        
        return features_df
    
    def predict_win_probability(self, target_pokemon: Dict[str, Any], counter_pokemon: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Prédit la probabilité de victoire du counter_pokemon contre le target_pokemon.
        
        Args:
            target_pokemon: Données du Pokémon cible
            counter_pokemon: Données du Pokémon candidat comme contre
            
        Returns:
            Tuple (win_probability, additional_info)
        """
        try:
            # Préparer les caractéristiques
            features = self.prepare_features(target_pokemon, counter_pokemon)
            
            # Faire la prédiction
            win_probability = self.model.predict_proba(features)[0, 1]
            
            # Déterminer les avantages du counter
            advantages = []
            
            # Avantage de vitesse
            if counter_pokemon['speed'] > target_pokemon['speed']:
                speed_diff = counter_pokemon['speed'] - target_pokemon['speed']
                if speed_diff > 20:
                    advantages.append(f"Avantage de vitesse significatif (+{speed_diff})")
                else:
                    advantages.append(f"Légèrement plus rapide (+{speed_diff})")
            
            # Avantage de statistiques
            counter_bst = sum([counter_pokemon.get(stat, 0) for stat in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']])
            target_bst = sum([target_pokemon.get(stat, 0) for stat in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']])
            
            if counter_bst > target_bst + 50:
                advantages.append(f"Statistiques globales supérieures (+{counter_bst - target_bst})")
            
            # Avantages spécifiques
            for stat in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense']:
                if counter_pokemon[stat] > target_pokemon[stat] * 1.3:  # 30% supérieur
                    stat_name = {
                        'hp': 'PV', 
                        'attack': 'Attaque', 
                        'defense': 'Défense',
                        'sp_attack': 'Attaque Spéciale', 
                        'sp_defense': 'Défense Spéciale'
                    }.get(stat, stat)
                    advantages.append(f"{stat_name} supérieur(e) ({counter_pokemon[stat]} vs {target_pokemon[stat]})")
            
            # Si aucun avantage n'est trouvé mais la probabilité est bonne
            if not advantages and win_probability > 0.6:
                advantages.append("Bon équilibre général contre ce Pokémon")
            
            # Informations supplémentaires
            additional_info = {
                'advantages': advantages,
                'features_used': list(features.columns)
            }
            
            return win_probability, additional_info
            
        except Exception as e:
            print(f"Erreur lors de la prédiction: {e}")
            # Retourner une valeur par défaut en cas d'erreur
            return 0.5, {'advantages': ["Erreur lors de la prédiction"], 'error': str(e)}
    
    def search_counters(self, target_pokemon: Dict[str, Any], potential_counters: List[Dict[str, Any]], 
                        min_probability: float = 0.5, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recherche les meilleurs counters pour un Pokémon cible.
        
        Args:
            target_pokemon: Données du Pokémon cible
            potential_counters: Liste des Pokémon candidats comme counters
            min_probability: Probabilité minimale pour considérer un counter comme valide
            limit: Nombre maximum de counters à retourner
            
        Returns:
            Liste des meilleurs counters avec leurs probabilités et raisons
        """
        results = []
        
        for counter in potential_counters:
            win_probability, additional_info = self.predict_win_probability(target_pokemon, counter)
            
            if win_probability >= min_probability:
                counter_with_prob = counter.copy()
                counter_with_prob['win_probability'] = win_probability
                counter_with_prob['advantages'] = additional_info['advantages']
                
                results.append(counter_with_prob)
        
        # Trier les résultats par probabilité de victoire décroissante
        results.sort(key=lambda x: x['win_probability'], reverse=True)
        
        # Limiter le nombre de résultats
        return results[:limit]

# Initialisation du service (à faire une seule fois au démarrage de l'application)
prediction_service = PredictionService() 