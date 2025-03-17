import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

class CounterFinder:
    """
    Modèle pour trouver les meilleurs contre-Pokémon basé sur les données de matchups.
    """
    
    def __init__(self, model_path=None):
        """
        Initialise le modèle CounterFinder.
        
        Args:
            model_path: Chemin vers un modèle pré-entraîné (optionnel)
        """
        self.pokemon_df = None
        self.matchups_df = None
        self.usage_df = None
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_data(self, pokemon_path, matchups_path, usage_path):
        """
        Charge les données nécessaires pour l'entraînement et la prédiction.
        
        Args:
            pokemon_path: Chemin vers le fichier pokemon.csv
            matchups_path: Chemin vers le fichier pokemon_matchups.csv
            usage_path: Chemin vers le fichier pokemon_usage_stats.csv
        """
        print("Chargement des données...")
        self.pokemon_df = pd.read_csv(pokemon_path)
        self.matchups_df = pd.read_csv(matchups_path)
        self.usage_df = pd.read_csv(usage_path)
        
        # Afficher un aperçu des données
        print("\nAperçu de pokemon_df:")
        print(self.pokemon_df.head())
        print("\nColonnes de pokemon_df:", self.pokemon_df.columns.tolist())
        
        print("\nAperçu de matchups_df:")
        print(self.matchups_df.head())
        print("\nColonnes de matchups_df:", self.matchups_df.columns.tolist())
        
        print("\nAperçu de usage_df:")
        print(self.usage_df.head())
        print("\nColonnes de usage_df:", self.usage_df.columns.tolist())
        
        # Nettoyage des données
        if 'Win Rate' in self.matchups_df.columns:
            self.matchups_df['Win_Rate'] = self.matchups_df['Win Rate'].str.rstrip('%').astype(float) / 100
        
        # Convertir les pourcentages en nombres décimaux dans usage_df
        if 'Usage_Percent' in self.usage_df.columns:
            self.usage_df['Usage_Percent'] = self.usage_df['Usage_Percent'].str.rstrip('%').astype(float) / 100
        
        if 'Win_Percent' in self.usage_df.columns:
            self.usage_df['Win_Percent'] = self.usage_df['Win_Percent'].str.rstrip('%').astype(float) / 100
        
        print(f"\nDonnées chargées: {len(self.pokemon_df)} Pokémon, {len(self.matchups_df)} matchups, {len(self.usage_df)} statistiques d'utilisation")
    
    def prepare_features(self):
        """
        Prépare les caractéristiques pour l'entraînement du modèle.
        
        Returns:
            X: Caractéristiques
            y: Étiquettes (1 si Pokemon1 gagne, 0 sinon)
        """
        print("Préparation des caractéristiques...")
        
        # Créer un DataFrame pour les caractéristiques
        features_df = self.matchups_df.copy()
        
        # Ajouter les informations de base des Pokémon
        features_df = self._add_pokemon_features(features_df, 'Pokemon1', 'p1')
        features_df = self._add_pokemon_features(features_df, 'Pokemon2', 'p2')
        
        # Ajouter les statistiques d'utilisation
        features_df = self._add_usage_stats(features_df, 'Pokemon1', 'p1')
        features_df = self._add_usage_stats(features_df, 'Pokemon2', 'p2')
        
        # Calculer des caractéristiques supplémentaires
        features_df = self._calculate_derived_features(features_df)
        
        # Sélectionner les colonnes pertinentes pour X
        feature_cols = [col for col in features_df.columns if col.startswith('p1_') or col.startswith('p2_') or col.startswith('ratio_')]
        self.feature_names = feature_cols
        
        # Créer X et y
        X = features_df[feature_cols]
        
        # Étiquette: 1 si Pokemon1 gagne (Win_Rate > 50%), 0 sinon
        y = (features_df['Win_Rate'] > 0.5).astype(int)
        
        print(f"Caractéristiques préparées: {X.shape[1]} caractéristiques, {X.shape[0]} exemples")
        return X, y
    
    def _add_pokemon_features(self, df, pokemon_col, prefix):
        """
        Ajoute les caractéristiques de base des Pokémon au DataFrame.
        
        Args:
            df: DataFrame contenant les matchups
            pokemon_col: Nom de la colonne contenant le nom du Pokémon
            prefix: Préfixe pour les nouvelles colonnes
        
        Returns:
            DataFrame enrichi
        """
        result_df = df.copy()
        
        # Adapter les noms de colonnes en fonction de ce qui est disponible
        name_col = 'name' if 'name' in self.pokemon_df.columns else 'Name' if 'Name' in self.pokemon_df.columns else None
        type1_col = 'type1' if 'type1' in self.pokemon_df.columns else 'Type1' if 'Type1' in self.pokemon_df.columns else None
        type2_col = 'type2' if 'type2' in self.pokemon_df.columns else 'Type2' if 'Type2' in self.pokemon_df.columns else None
        hp_col = 'hp' if 'hp' in self.pokemon_df.columns else 'HP' if 'HP' in self.pokemon_df.columns else None
        attack_col = 'attack' if 'attack' in self.pokemon_df.columns else 'Attack' if 'Attack' in self.pokemon_df.columns else None
        defense_col = 'defense' if 'defense' in self.pokemon_df.columns else 'Defense' if 'Defense' in self.pokemon_df.columns else None
        sp_atk_col = 'sp_attack' if 'sp_attack' in self.pokemon_df.columns else 'Sp. Atk' if 'Sp. Atk' in self.pokemon_df.columns else None
        sp_def_col = 'sp_defense' if 'sp_defense' in self.pokemon_df.columns else 'Sp. Def' if 'Sp. Def' in self.pokemon_df.columns else None
        speed_col = 'speed' if 'speed' in self.pokemon_df.columns else 'Speed' if 'Speed' in self.pokemon_df.columns else None
        
        if not name_col:
            print("ERREUR: Impossible de trouver la colonne du nom du Pokémon dans pokemon_df")
            return result_df
        
        # Créer un dictionnaire pour un accès rapide aux données des Pokémon
        pokemon_dict = self.pokemon_df.set_index(name_col).to_dict('index')
        
        # Initialiser les colonnes avec des valeurs par défaut
        stat_columns = [
            f'{prefix}_type1', f'{prefix}_type2', 
            f'{prefix}_hp', f'{prefix}_attack', f'{prefix}_defense',
            f'{prefix}_sp_atk', f'{prefix}_sp_def', f'{prefix}_speed', f'{prefix}_bst'
        ]
        
        for col in stat_columns:
            result_df[col] = 0  # Valeur par défaut pour les stats numériques
            if col.endswith('type1') or col.endswith('type2'):
                result_df[col] = 'Unknown'  # Valeur par défaut pour les types
        
        # Ajouter les caractéristiques pour chaque Pokémon
        for pokemon_name in df[pokemon_col].unique():
            if pokemon_name in pokemon_dict:
                pokemon_data = pokemon_dict[pokemon_name]
                
                # Filtrer les lignes pour ce Pokémon
                mask = df[pokemon_col] == pokemon_name
                
                # Ajouter les caractéristiques si elles existent
                if type1_col and type1_col in pokemon_data:
                    result_df.loc[mask, f'{prefix}_type1'] = pokemon_data[type1_col]
                if type2_col and type2_col in pokemon_data:
                    result_df.loc[mask, f'{prefix}_type2'] = pokemon_data[type2_col]
                if hp_col and hp_col in pokemon_data:
                    result_df.loc[mask, f'{prefix}_hp'] = pokemon_data[hp_col]
                if attack_col and attack_col in pokemon_data:
                    result_df.loc[mask, f'{prefix}_attack'] = pokemon_data[attack_col]
                if defense_col and defense_col in pokemon_data:
                    result_df.loc[mask, f'{prefix}_defense'] = pokemon_data[defense_col]
                if sp_atk_col and sp_atk_col in pokemon_data:
                    result_df.loc[mask, f'{prefix}_sp_atk'] = pokemon_data[sp_atk_col]
                if sp_def_col and sp_def_col in pokemon_data:
                    result_df.loc[mask, f'{prefix}_sp_def'] = pokemon_data[sp_def_col]
                if speed_col and speed_col in pokemon_data:
                    result_df.loc[mask, f'{prefix}_speed'] = pokemon_data[speed_col]
                
                # Calculer le BST (Base Stat Total) si toutes les stats sont disponibles
                if all(col and col in pokemon_data for col in [hp_col, attack_col, defense_col, sp_atk_col, sp_def_col, speed_col]):
                    result_df.loc[mask, f'{prefix}_bst'] = (
                        pokemon_data[hp_col] + pokemon_data[attack_col] + pokemon_data[defense_col] +
                        pokemon_data[sp_atk_col] + pokemon_data[sp_def_col] + pokemon_data[speed_col]
                    )
            else:
                print(f"Attention: {pokemon_name} non trouvé dans pokemon_df")
        
        return result_df
    
    def _add_usage_stats(self, df, pokemon_col, prefix):
        """
        Ajoute les statistiques d'utilisation au DataFrame.
        
        Args:
            df: DataFrame contenant les matchups
            pokemon_col: Nom de la colonne contenant le nom du Pokémon
            prefix: Préfixe pour les nouvelles colonnes
        
        Returns:
            DataFrame enrichi
        """
        result_df = df.copy()
        
        # Créer un dictionnaire pour un accès rapide aux statistiques d'utilisation
        usage_dict = {}
        for _, row in self.usage_df.iterrows():
            pokemon = row['Pokemon']
            if pokemon not in usage_dict:
                usage_dict[pokemon] = {}
            
            season = row['Season']
            usage_dict[pokemon][season] = {
                'usage_percent': row['Usage_Percent'] if 'Usage_Percent' in row else 0,
                'win_percent': row['Win_Percent'] if 'Win_Percent' in row else 0.5
            }
        
        # Initialiser les colonnes avec des valeurs par défaut
        result_df[f'{prefix}_usage_percent'] = 0.0
        result_df[f'{prefix}_win_percent'] = 0.5
        
        # Ajouter les statistiques pour chaque Pokémon
        for pokemon_name in df[pokemon_col].unique():
            if pokemon_name in usage_dict:
                # Prendre la moyenne des statistiques sur toutes les saisons
                avg_usage = np.mean([stats['usage_percent'] for season, stats in usage_dict[pokemon_name].items()])
                avg_win = np.mean([stats['win_percent'] for season, stats in usage_dict[pokemon_name].items()])
                
                # Filtrer les lignes pour ce Pokémon
                mask = df[pokemon_col] == pokemon_name
                
                # Ajouter les statistiques
                result_df.loc[mask, f'{prefix}_usage_percent'] = avg_usage
                result_df.loc[mask, f'{prefix}_win_percent'] = avg_win
        
        return result_df
    
    def _calculate_derived_features(self, df):
        """
        Calcule des caractéristiques dérivées à partir des caractéristiques de base.
        
        Args:
            df: DataFrame contenant les caractéristiques de base
        
        Returns:
            DataFrame enrichi avec des caractéristiques dérivées
        """
        result_df = df.copy()
        
        # Vérifier quelles colonnes sont disponibles
        available_cols = result_df.columns.tolist()
        
        # Calculer les ratios de statistiques si les colonnes existent
        stat_pairs = [
            ('hp', 'hp'), 
            ('attack', 'defense'), 
            ('sp_atk', 'sp_def'), 
            ('speed', 'speed'),
            ('bst', 'bst')
        ]
        
        for stat1, stat2 in stat_pairs:
            p1_col = f'p1_{stat1}'
            p2_col = f'p2_{stat2}'
            
            if p1_col in available_cols and p2_col in available_cols:
                # Éviter la division par zéro
                result_df[f'ratio_{stat1}'] = result_df[p1_col] / result_df[p2_col].replace(0, 1)
            else:
                print(f"Colonnes {p1_col} ou {p2_col} non disponibles pour calculer le ratio")
        
        # Calculer d'autres caractéristiques dérivées si possible
        if 'p1_attack' in available_cols and 'p2_defense' in available_cols:
            result_df['ratio_physical_offense'] = result_df['p1_attack'] / result_df['p2_defense'].replace(0, 1)
        
        if 'p1_sp_atk' in available_cols and 'p2_sp_def' in available_cols:
            result_df['ratio_special_offense'] = result_df['p1_sp_atk'] / result_df['p2_sp_def'].replace(0, 1)
        
        if 'p1_defense' in available_cols and 'p2_attack' in available_cols:
            result_df['ratio_physical_defense'] = result_df['p1_defense'] / result_df['p2_attack'].replace(0, 1)
        
        if 'p1_sp_def' in available_cols and 'p2_sp_atk' in available_cols:
            result_df['ratio_special_defense'] = result_df['p1_sp_def'] / result_df['p2_sp_atk'].replace(0, 1)
        
        return result_df
    
    def train(self, test_size=0.2, random_state=42, save_path=None):
        """
        Entraîne le modèle de prédiction de contre-Pokémon.
        
        Args:
            test_size: Proportion des données à utiliser pour le test
            random_state: Graine aléatoire pour la reproductibilité
            save_path: Chemin pour sauvegarder le modèle (optionnel)
        
        Returns:
            Métriques d'évaluation du modèle
        """
        print("Préparation des données d'entraînement...")
        X, y = self.prepare_features()
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Entraînement sur {X_train.shape[0]} exemples, test sur {X_test.shape[0]} exemples")
        
        # Identifier les colonnes catégorielles et numériques
        categorical_cols = [col for col in X.columns if col.endswith('_type1') or col.endswith('_type2')]
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Créer un préprocesseur pour transformer les caractéristiques
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        # Créer un pipeline avec prétraitement et modèle
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=random_state))
        ])
        
        # Entraîner le modèle
        print("Entraînement du modèle...")
        pipeline.fit(X_train, y_train)
        
        # Évaluer le modèle
        print("Évaluation du modèle...")
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except:
            auc = 0.5  # Valeur par défaut si le calcul échoue
        
        precision = precision_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Précision: {accuracy:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"Précision: {precision:.4f}")
        print("Matrice de confusion:")
        print(conf_matrix)
        
        # Sauvegarder le modèle
        self.model = pipeline
        self.preprocessor = preprocessor
        
        if save_path:
            print(f"Sauvegarde du modèle à {save_path}...")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(pipeline, save_path)
        
        # Analyser l'importance des caractéristiques
        self._analyze_feature_importance()
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'confusion_matrix': conf_matrix
        }
    
    def _analyze_feature_importance(self):
        """
        Analyse et affiche l'importance des caractéristiques du modèle.
        """
        if self.model is None or not hasattr(self.model, 'named_steps'):
            print("Modèle non entraîné, impossible d'analyser l'importance des caractéristiques")
            return
        
        # Extraire le modèle RandomForest du pipeline
        rf_model = self.model.named_steps['classifier']
        
        # Obtenir les noms des caractéristiques après transformation
        preprocessor = self.model.named_steps['preprocessor']
        
        # Obtenir l'importance des caractéristiques
        feature_importances = rf_model.feature_importances_
        
        # Créer un DataFrame pour l'importance des caractéristiques
        importance_df = pd.DataFrame({
            'Feature': [f"Feature_{i}" for i in range(len(feature_importances))],
            'Importance': feature_importances
        })
        
        # Trier par importance décroissante
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Afficher les 20 caractéristiques les plus importantes
        print("\nCaractéristiques les plus importantes:")
        print(importance_df.head(20))
        
        # Visualiser l'importance des caractéristiques
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Importance des caractéristiques')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Graphique d'importance des caractéristiques sauvegardé dans 'feature_importance.png'")
    
    def find_counters(self, pokemon_name, top_k=5):
        """
        Trouve les meilleurs contre-Pokémon pour un Pokémon donné en utilisant une approche hybride optimisée.
        
        Args:
            pokemon_name: Nom du Pokémon pour lequel trouver des contre-Pokémon
            top_k: Nombre de contre-Pokémon à retourner
        
        Returns:
            Liste des meilleurs contre-Pokémon avec leur probabilité de victoire et justification
        """
        if self.model is None:
            print("Modèle non entraîné, impossible de trouver des contre-Pokémon")
            return []
        
        print(f"Recherche de contre-Pokémon pour {pokemon_name}...")
        
        # Déterminer le nom de la colonne contenant les noms des Pokémon
        name_col = 'name' if 'name' in self.pokemon_df.columns else 'Name' if 'Name' in self.pokemon_df.columns else None
        
        if not name_col:
            print("Impossible de trouver la colonne du nom du Pokémon")
            return []
        
        # Vérifier si le Pokémon existe exactement ou comme préfixe (pour gérer les formes)
        exact_match = pokemon_name in self.pokemon_df[name_col].values
        
        if not exact_match:
            print(f"Pokémon '{pokemon_name}' non trouvé exactement dans la base de données")
            try:
                prefix_matches = self.pokemon_df[self.pokemon_df[name_col].str.startswith(pokemon_name + "-")]
                if not prefix_matches.empty:
                    pokemon_name = prefix_matches.iloc[0][name_col]
                    print(f"Utilisation de la forme alternative: {pokemon_name}")
                else:
                    print(f"Aucune forme alternative trouvée pour {pokemon_name}")
                    return []
            except Exception as e:
                print(f"Erreur lors de la recherche de formes alternatives: {e}")
                return []
        
        # Récupérer les données du Pokémon cible
        print(f"Récupération des données pour {pokemon_name}...")
        target_pokemon = self.pokemon_df[self.pokemon_df[name_col] == pokemon_name].iloc[0]
        target_type1 = target_pokemon['type1']
        target_type2 = target_pokemon['type2'] if pd.notna(target_pokemon['type2']) else None
        
        print(f"Types du Pokémon cible: {target_type1}" + (f"/{target_type2}" if target_type2 else ""))
        
        # 1. Vérifier les matchups existants dans les données
        print("Recherche dans les matchups existants...")
        matchup_counters = self._get_existing_matchups(pokemon_name)
        
        # 2. Trouver les Pokémon avec le meilleur avantage de type
        print("Recherche des Pokémon avec avantage de type...")
        type_advantage_counters = self._get_best_type_counters(target_pokemon, 15)
        
        # 3. Combiner les résultats
        all_counters = matchup_counters + type_advantage_counters
        
        # Éliminer les doublons en gardant la meilleure probabilité
        unique_counters = {}
        for counter in all_counters:
            pokemon, prob, source, reasons = counter
            if pokemon not in unique_counters or prob > unique_counters[pokemon][1]:
                unique_counters[pokemon] = counter
        
        # Trier par probabilité de victoire
        potential_counters = sorted(unique_counters.values(), key=lambda x: x[1], reverse=True)[:15]
        
        # 4. Valider les contre-Pokémon avec le modèle ML
        print("Validation des contre-Pokémon avec le modèle ML...")
        validated_counters = self._validate_with_model(pokemon_name, [c[0] for c in potential_counters])
        
        # 5. Fusionner les résultats
        final_counters = []
        for counter_name, model_prob in validated_counters:
            # Trouver les informations originales
            for c in potential_counters:
                if c[0] == counter_name:
                    original_prob = c[1]
                    original_source = c[2]
                    reasons = c[3]
                    
                    # Utiliser la probabilité du modèle si disponible, sinon l'originale
                    prob = model_prob if model_prob is not None else original_prob
                    
                    # Déterminer la source des données
                    if model_prob is not None:
                        if original_source == "matchups":
                            source = "Modèle ML + Matchups historiques"
                        else:
                            source = "Modèle ML + Analyse de types"
                    else:
                        if original_source == "matchups":
                            source = "Matchups historiques"
                        else:
                            source = "Analyse de types"
                    
                    # Vérifier si le Pokémon est populaire dans les statistiques d'utilisation
                    usage_data = self.usage_df[self.usage_df['Pokemon'] == counter_name]
                    if not usage_data.empty:
                        rank = usage_data['Rank'].mean()
                        if rank <= 20:
                            source += " + Très populaire (Top 20)"
                        elif rank <= 50:
                            source += " + Populaire (Top 50)"
                    
                    final_counters.append({
                        'pokemon': counter_name,
                        'win_probability': prob,
                        'source': source,
                        'reasons': reasons
                    })
                    break
        
        # Trier par probabilité de victoire
        final_counters.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return final_counters[:top_k]
    
    def _validate_with_model(self, pokemon_name, counter_candidates, batch_size=10):
        """
        Valide les contre-Pokémon avec le modèle ML.
        
        Args:
            pokemon_name: Nom du Pokémon cible
            counter_candidates: Liste des noms de Pokémon candidats
            batch_size: Taille du lot pour le traitement par lots
        
        Returns:
            Liste des contre-Pokémon validés avec leur probabilité de victoire
        """
        if self.model is None:
            return [(c, None) for c in counter_candidates]
        
        validated_counters = []
        
        # Traiter par lots pour plus d'efficacité
        for i in range(0, len(counter_candidates), batch_size):
            batch = counter_candidates[i:i+batch_size]
            print(f"Validation du lot {i//batch_size + 1}/{(len(counter_candidates) + batch_size - 1)//batch_size}...")
            
            batch_data = []
            valid_candidates = []
            
            for candidate in batch:
                try:
                    # Créer une ligne de données pour ce matchup
                    row = pd.DataFrame({
                        'Pokemon1': [candidate],
                        'Pokemon2': [pokemon_name],
                        'Wins': [0],
                        'Losses': [0],
                        'Total': [0],
                        'Win_Rate': [0.5]
                    })
                    
                    # Ajouter les caractéristiques
                    row = self._add_pokemon_features(row, 'Pokemon1', 'p1')
                    row = self._add_pokemon_features(row, 'Pokemon2', 'p2')
                    row = self._add_usage_stats(row, 'Pokemon1', 'p1')
                    row = self._add_usage_stats(row, 'Pokemon2', 'p2')
                    row = self._calculate_derived_features(row)
                    
                    # Vérifier les colonnes disponibles
                    available_cols = set(row.columns)
                    required_cols = set(self.feature_names)
                    missing_cols = required_cols - available_cols
                    
                    if missing_cols:
                        validated_counters.append((candidate, None))
                        continue
                    
                    # Sélectionner les colonnes pertinentes
                    batch_data.append(row[self.feature_names])
                    valid_candidates.append(candidate)
                    
                except Exception as e:
                    validated_counters.append((candidate, None))
                    continue
            
            if not batch_data:
                continue
            
            # Concaténer toutes les données de prédiction
            X_pred = pd.concat(batch_data, ignore_index=True)
            
            # Prédire les probabilités de victoire
            probas = self.model.predict_proba(X_pred)[:, 1]
            
            # Ajouter les résultats validés
            for candidate, prob in zip(valid_candidates, probas):
                validated_counters.append((candidate, prob))
        
        return validated_counters
    
    def _get_existing_matchups(self, pokemon_name):
        """
        Récupère les matchups existants dans les données pour un Pokémon donné.
        
        Args:
            pokemon_name: Nom du Pokémon
        
        Returns:
            Liste des contre-Pokémon avec leur probabilité de victoire
        """
        # Chercher les matchups où le Pokémon est Pokemon2
        matchups = self.matchups_df[self.matchups_df['Pokemon2'] == pokemon_name]
        
        counters = []
        for _, row in matchups.iterrows():
            counter_name = row['Pokemon1']
            win_rate = row['Win_Rate'] if 'Win_Rate' in row else float(row['Win Rate'].strip('%')) / 100
            
            # Récupérer les données du Pokémon counter
            counter_data = self.pokemon_df[self.pokemon_df['name'] == counter_name]
            if counter_data.empty:
                continue
            
            counter_data = counter_data.iloc[0]
            
            # Récupérer les données du Pokémon cible
            target_data = self.pokemon_df[self.pokemon_df['name'] == pokemon_name]
            if target_data.empty:
                continue
            
            target_data = target_data.iloc[0]
            
            # Calculer l'avantage de type
            type_advantage = self._calculate_type_advantage(counter_data, target_data)
            
            # Calculer l'avantage de statistiques
            stat_advantage = self._calculate_stat_advantage(counter_data, target_data)
            
            # Créer la liste des raisons
            reasons = []
            
            if type_advantage > 1:
                reasons.append(f"Avantage de type (x{type_advantage:.1f})")
            
            for stat, value in stat_advantage.items():
                if value > 20:  # Seulement les avantages significatifs
                    reasons.append(f"Meilleur en {stat} (+{value})")
            
            if not reasons:
                reasons.append("Bon matchup historique")
            
            counters.append((counter_name, win_rate, "matchups", reasons))
        
        # Trier par taux de victoire
        counters.sort(key=lambda x: x[1], reverse=True)
        
        return counters[:10]  # Retourner les 10 meilleurs
    
    def _get_best_type_counters(self, target_pokemon, limit=10):
        """
        Trouve les Pokémon avec le meilleur avantage de type contre le Pokémon cible.
        
        Args:
            target_pokemon: Données du Pokémon cible
            limit: Nombre maximum de contre-Pokémon à retourner
        
        Returns:
            Liste des contre-Pokémon avec leur probabilité de victoire estimée
        """
        target_type1 = target_pokemon['type1']
        target_type2 = target_pokemon['type2'] if pd.notna(target_pokemon['type2']) else None
        
        # Identifier les types qui sont super efficaces contre le Pokémon cible
        super_effective_types = []
        for col in self.pokemon_df.columns:
            if col.startswith('against_'):
                type_name = col.replace('against_', '')
                if target_pokemon[col] > 1.0:
                    super_effective_types.append(type_name)
        
        print(f"Types super efficaces contre {target_pokemon['name']}: {super_effective_types}")
        
        # Filtrer les Pokémon qui ont ces types
        candidates = self.pokemon_df[
            (self.pokemon_df['type1'].isin(super_effective_types)) | 
            ((self.pokemon_df['type2'].notna()) & (self.pokemon_df['type2'].isin(super_effective_types)))
        ]
        
        # Calculer l'avantage de type pour chaque candidat
        type_counters = []
        
        for _, row in candidates.iterrows():
            pokemon_name = row['name']
            
            if pokemon_name == target_pokemon['name']:
                continue
            
            # Calculer l'avantage de type
            type_advantage = self._calculate_type_advantage(row, target_pokemon)
            
            # Calculer l'avantage de statistiques
            stat_advantage = self._calculate_stat_advantage(row, target_pokemon)
            
            # Si avantage de type significatif
            if type_advantage > 1.0:
                reasons = []
                
                reasons.append(f"Avantage de type (x{type_advantage:.1f})")
                
                for stat, value in stat_advantage.items():
                    if value > 20:  # Seulement les avantages significatifs
                        reasons.append(f"Meilleur en {stat} (+{value})")
                
                # Estimer une probabilité de victoire basée sur l'avantage de type et de stats
                estimated_win_prob = min(0.95, (type_advantage * 0.4) + (sum(v for v in stat_advantage.values() if v > 0) / 600))
                
                type_counters.append((pokemon_name, estimated_win_prob, "type", reasons))
        
        # Trier par avantage de type
        type_counters.sort(key=lambda x: x[1], reverse=True)
        
        return type_counters[:limit]
    
    def _calculate_type_advantage(self, attacker, defender):
        """
        Calcule l'avantage de type d'un Pokémon attaquant contre un défenseur.
        """
        # Récupérer les types
        atk_type1 = attacker['type1']
        atk_type2 = attacker['type2'] if pd.notna(attacker['type2']) else None
        
        # Récupérer les multiplicateurs de dégâts du défenseur
        type_multipliers = {}
        for col in self.pokemon_df.columns:
            if col.startswith('against_'):
                type_name = col.replace('against_', '')
                type_multipliers[type_name] = defender[col]
        
        # Calculer l'avantage de type
        advantage = 1.0
        
        if atk_type1 in type_multipliers:
            advantage *= type_multipliers[atk_type1]
        
        if atk_type2 and atk_type2 in type_multipliers:
            advantage *= type_multipliers[atk_type2]
        
        return advantage
    
    def _calculate_stat_advantage(self, pokemon1, pokemon2):
        """
        Calcule l'avantage de statistiques d'un Pokémon par rapport à un autre.
        """
        stat_advantage = {}
        
        # Comparer les statistiques importantes
        for stat in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']:
            if stat in pokemon1 and stat in pokemon2:
                stat_advantage[stat] = int(pokemon1[stat]) - int(pokemon2[stat])
        
        return stat_advantage
    
    def load_model(self, model_path):
        """
        Charge un modèle pré-entraîné.
        
        Args:
            model_path: Chemin vers le fichier du modèle
        """
        print(f"Chargement du modèle depuis {model_path}...")
        self.model = joblib.load(model_path)
        
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            self.preprocessor = self.model.named_steps['preprocessor']
        
        print("Modèle chargé avec succès")
    
    def save_model(self, model_path):
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            model_path: Chemin où sauvegarder le modèle
        """
        if self.model is None:
            print("Aucun modèle à sauvegarder")
            return
        
        print(f"Sauvegarde du modèle à {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
        print("Modèle sauvegardé avec succès")


def main():
    """
    Fonction principale pour entraîner et tester le modèle CounterFinder.
    """
    # Chemins vers les fichiers de données
    pokemon_path = 'data/pokemon.csv'
    matchups_path = 'data/pokemon_matchups.csv'
    usage_path = 'data/pokemon_usage_stats.csv'
    
    # Créer et entraîner le modèle
    counter_finder = CounterFinder()
    counter_finder.load_data(pokemon_path, matchups_path, usage_path)
    
    # Entraîner le modèle et sauvegarder
    metrics = counter_finder.train(save_path='models/counter_finder.joblib')
    
    # Tester le modèle avec quelques Pokémon populaires
    test_pokemon = ['Tyranitar', 'Charizard', 'Landorus', 'Garchomp', 'Zapdos']
    
    for pokemon in test_pokemon:
        print(f"\n{'='*50}")
        print(f"MEILLEURS CONTRE-POKÉMON POUR {pokemon.upper()}")
        print(f"{'='*50}")
        
        counters = counter_finder.find_counters(pokemon, top_k=5)
        
        for i, counter in enumerate(counters, 1):
            print(f"\n{i}. {counter['pokemon']} - {counter['win_probability']:.1%} de chance de gagner")
            print(f"   Source: {counter['source']}")
            print(f"   Raisons:")
            for reason in counter['reasons']:
                print(f"   • {reason}")


if __name__ == "__main__":
    main() 