import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

class RandomForestCounterFinder:
    """
    Classe pour trouver les meilleurs counters pour un Pokémon ou une équipe
    en utilisant un modèle RandomForest.
    """
    
    def __init__(self):
        # Charger les données
        self.usage_stats = pd.read_csv('data/pokemon_usage_stats.csv')
        self.pokemon_data = pd.read_csv('data/pokemon.csv')
        
        # Prétraitement des données
        self._preprocess_data()
        
        # Entraîner le modèle
        self._train_model()
    
    def _preprocess_data(self):
        """Prétraite les données pour l'analyse."""
        # Nettoyer les noms de Pokémon pour correspondre entre les deux datasets
        self.usage_stats['Pokemon'] = self.usage_stats['Pokemon'].str.strip()
        self.pokemon_data['name'] = self.pokemon_data['name'].str.strip()
        
        # Convertir les colonnes numériques
        self.usage_stats['Usage_Percent'] = pd.to_numeric(self.usage_stats['Usage_Percent'], errors='coerce')
        self.usage_stats['Win_Percent'] = pd.to_numeric(self.usage_stats['Win_Percent'], errors='coerce')
        
        # Calculer les statistiques moyennes par Pokémon
        self.avg_stats = self.usage_stats.groupby('Pokemon').agg({
            'Usage_Percent': 'mean',
            'Win_Percent': 'mean',
            'Season': 'count'  # Nombre de saisons où le Pokémon apparaît
        }).rename(columns={'Season': 'Frequency'})
        
        # Créer un dictionnaire de correspondance pour les noms de Pokémon
        self.pokemon_name_map = {}
        for name in self.pokemon_data['name']:
            self.pokemon_name_map[name.lower()] = name
        
        # Préparer les données d'entraînement
        self._prepare_training_data()
    
    def _prepare_training_data(self):
        """Prépare les données pour l'entraînement du modèle."""
        # Créer des paires de Pokémon pour l'entraînement
        training_data = []
        
        # Utiliser les données d'utilisation pour créer des exemples positifs et négatifs
        seasons = self.usage_stats['Season'].unique()
        
        for season in seasons:
            season_data = self.usage_stats[self.usage_stats['Season'] == season]
            
            # Trier par taux d'utilisation
            top_pokemon = season_data.sort_values('Usage_Percent', ascending=False)
            
            # Pour chaque Pokémon populaire, trouver des counters potentiels
            for i, row in top_pokemon.head(20).iterrows():
                pokemon = row['Pokemon']
                
                # Les Pokémon avec un taux de victoire élevé contre les Pokémon populaires
                # sont probablement des counters
                potential_counters = season_data[
                    (season_data['Win_Percent'] > 55) & 
                    (season_data['Pokemon'] != pokemon)
                ]
                
                # Ajouter des exemples positifs (counters)
                for _, counter_row in potential_counters.head(5).iterrows():
                    counter = counter_row['Pokemon']
                    if pokemon in self.pokemon_data['name'].values and counter in self.pokemon_data['name'].values:
                        training_data.append(self._create_pair_features(pokemon, counter, 1))
                
                # Ajouter des exemples négatifs (non-counters)
                non_counters = season_data[
                    (season_data['Win_Percent'] < 45) & 
                    (season_data['Pokemon'] != pokemon)
                ]
                
                for _, non_counter_row in non_counters.head(5).iterrows():
                    non_counter = non_counter_row['Pokemon']
                    if pokemon in self.pokemon_data['name'].values and non_counter in self.pokemon_data['name'].values:
                        training_data.append(self._create_pair_features(pokemon, non_counter, 0))
        
        # Convertir en DataFrame
        self.training_df = pd.DataFrame(training_data)
        
        # Gérer les valeurs manquantes
        self.training_df = self.training_df.fillna(0)
    
    def _create_pair_features(self, pokemon1, pokemon2, is_counter):
        """Crée des caractéristiques pour une paire de Pokémon."""
        # Obtenir les données de base pour les deux Pokémon
        p1_data = self.pokemon_data[self.pokemon_data['name'] == pokemon1].iloc[0]
        p2_data = self.pokemon_data[self.pokemon_data['name'] == pokemon2].iloc[0]
        
        # Caractéristiques basées sur les statistiques
        features = {
            'pokemon1': pokemon1,
            'pokemon2': pokemon2,
            'p1_type1': p1_data['type1'],
            'p1_type2': p1_data['type2'] if not pd.isna(p1_data['type2']) else '',
            'p2_type1': p2_data['type1'],
            'p2_type2': p2_data['type2'] if not pd.isna(p2_data['type2']) else '',
            'p1_hp': p1_data['hp'],
            'p1_attack': p1_data['attack'],
            'p1_defense': p1_data['defense'],
            'p1_sp_attack': p1_data['sp_attack'],
            'p1_sp_defense': p1_data['sp_defense'],
            'p1_speed': p1_data['speed'],
            'p2_hp': p2_data['hp'],
            'p2_attack': p2_data['attack'],
            'p2_defense': p2_data['defense'],
            'p2_sp_attack': p2_data['sp_attack'],
            'p2_sp_defense': p2_data['sp_defense'],
            'p2_speed': p2_data['speed'],
            'speed_diff': p2_data['speed'] - p1_data['speed'],
            'attack_vs_defense': p2_data['attack'] - p1_data['defense'],
            'sp_attack_vs_sp_defense': p2_data['sp_attack'] - p1_data['sp_defense'],
            'is_counter': is_counter
        }
        
        # Ajouter les efficacités de type
        for col in self.pokemon_data.columns:
            if col.startswith('against_'):
                type_name = col.replace('against_', '')
                features[f'p1_{col}'] = p1_data[col]
                features[f'p2_{col}'] = p2_data[col]
                features[f'type_advantage_{type_name}'] = p2_data[col] - p1_data[col]
        
        return features
    
    def _train_model(self):
        """Entraîne le modèle RandomForest."""
        if not hasattr(self, 'training_df') or len(self.training_df) == 0:
            print("Pas assez de données d'entraînement disponibles.")
            return
        
        # Séparer les caractéristiques et la cible
        X = self.training_df.drop(['is_counter', 'pokemon1', 'pokemon2'], axis=1)
        y = self.training_df['is_counter']
        
        # Identifier les colonnes catégorielles et numériques
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        numerical_cols = [col for col in X.columns if X[col].dtype != 'object']
        
        # Créer un préprocesseur
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        # Créer le pipeline avec le préprocesseur et le modèle
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraîner le modèle
        self.model.fit(X_train, y_train)
        
        # Évaluer le modèle
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        print(f"Modèle entraîné avec une précision de {accuracy:.2f}, précision de {precision:.2f} et rappel de {recall:.2f}")
    
    def find_counters(self, pokemon_name, top_n=5):
        """
        Trouve les meilleurs counters pour un Pokémon donné.
        
        Args:
            pokemon_name: Nom du Pokémon
            top_n: Nombre de counters à retourner
            
        Returns:
            DataFrame avec les meilleurs counters
        """
        if not hasattr(self, 'model'):
            print("Le modèle n'a pas été entraîné.")
            return None
        
        # Vérifier si le Pokémon est dans nos données
        if pokemon_name not in self.pokemon_data['name'].values:
            print(f"Pokémon {pokemon_name} non trouvé dans les données.")
            return None
        
        # Obtenir tous les Pokémon possibles comme counters
        candidates = [p for p in self.avg_stats.index if p in self.pokemon_data['name'].values and p != pokemon_name]
        
        # Calculer les scores pour chaque candidat
        counter_scores = {}
        
        for candidate in candidates:
            # Créer les caractéristiques pour cette paire
            pair_features = self._create_pair_features(pokemon_name, candidate, 0)
            
            # Supprimer les colonnes non utilisées pour la prédiction
            features = {k: v for k, v in pair_features.items() 
                       if k not in ['is_counter', 'pokemon1', 'pokemon2']}
            
            # Convertir en DataFrame
            features_df = pd.DataFrame([features])
            
            # Prédire la probabilité que ce soit un counter
            counter_prob = self.model.predict_proba(features_df)[0][1]
            
            counter_scores[candidate] = counter_prob
        
        # Trier et retourner les top_n
        counter_df = pd.DataFrame({
            'Counter': counter_scores.keys(),
            'Score': counter_scores.values()
        }).sort_values('Score', ascending=False).head(top_n)
        
        # Ajouter des informations supplémentaires
        counter_df['Win_Rate'] = [self.avg_stats.loc[p, 'Win_Percent'] if p in self.avg_stats.index else 0 
                                 for p in counter_df['Counter']]
        counter_df['Usage_Rate'] = [self.avg_stats.loc[p, 'Usage_Percent'] if p in self.avg_stats.index else 0 
                                   for p in counter_df['Counter']]
        
        return counter_df
    
    def find_team_counters(self, team_list, top_n=5):
        """
        Trouve les meilleurs counters pour une équipe de Pokémon.
        
        Args:
            team_list: Liste des noms de Pokémon dans l'équipe
            top_n: Nombre de counters à retourner
            
        Returns:
            DataFrame avec les meilleurs counters
        """
        # Vérifier que tous les Pokémon sont dans nos données
        valid_pokemon = [p for p in team_list if p in self.pokemon_data['name'].values]
        
        if len(valid_pokemon) == 0:
            print("Aucun Pokémon de l'équipe n'a été trouvé dans les données.")
            return None
        
        if len(valid_pokemon) < len(team_list):
            print(f"Attention: {len(team_list) - len(valid_pokemon)} Pokémon non trouvés dans les données.")
        
        # Calculer les scores de counter pour chaque Pokémon de l'équipe
        all_counters = {}
        
        for pokemon in valid_pokemon:
            counters = self.find_counters(pokemon, top_n=top_n*2)
            
            if counters is not None:
                for _, row in counters.iterrows():
                    counter = row['Counter']
                    score = row['Score']
                    
                    if counter in all_counters:
                        all_counters[counter] += score
                    else:
                        all_counters[counter] = score
        
        # Normaliser les scores
        for counter in all_counters:
            all_counters[counter] /= len(valid_pokemon)
        
        # Trier et retourner les top_n
        team_counter_df = pd.DataFrame({
            'Counter': all_counters.keys(),
            'Score': all_counters.values()
        }).sort_values('Score', ascending=False).head(top_n)
        
        # Ajouter des informations supplémentaires
        team_counter_df['Win_Rate'] = [self.avg_stats.loc[p, 'Win_Percent'] if p in self.avg_stats.index else 0 
                                      for p in team_counter_df['Counter']]
        team_counter_df['Usage_Rate'] = [self.avg_stats.loc[p, 'Usage_Percent'] if p in self.avg_stats.index else 0 
                                        for p in team_counter_df['Counter']]
        
        return team_counter_df

# Exemple d'utilisation
if __name__ == "__main__":
    counter_finder = RandomForestCounterFinder()
    
    # Trouver les counters pour un Pokémon
    pokemon = "Tyranitar"
    counters = counter_finder.find_counters(pokemon, top_n=5)
    
    print(f"Meilleurs counters pour {pokemon}:")
    print(counters)
    
    # Trouver les counters pour une équipe
    team = ["Tyranitar", "Heatran", "Keldeo", "Clefable", "Rotom-Wash", "Latios"]
    team_counters = counter_finder.find_team_counters(team, top_n=5)
    
    print(f"\nMeilleurs counters pour l'équipe:")
    print(team_counters) 