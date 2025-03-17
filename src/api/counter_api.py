from flask import Flask, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Importer le service de prédiction
from src.services.prediction_service import PredictionService

app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

# Variables globales
pokemon_df = None
matchups_df = None
usage_df = None
prediction_service = None

def load_data_and_model():
    """Charge les données et le modèle optimisé."""
    global pokemon_df, matchups_df, usage_df, prediction_service
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Chemins pour les données
    pokemon_path = os.path.join(base_dir, 'data', 'pokemon.csv')
    matchups_path = os.path.join(base_dir, 'data', 'pokemon_matchups.csv')
    usage_path = os.path.join(base_dir, 'data', 'pokemon_usage_stats.csv')
    
    # Vérifier si le fichier pokemon.csv existe
    if not os.path.exists(pokemon_path):
        print(f"Fichier Pokémon non trouvé: {pokemon_path}")
        # Chercher dans d'autres emplacements possibles
        alternative_paths = [
            os.path.join(os.getcwd(), 'data', 'pokemon.csv'),
            'data/pokemon.csv',
            'D:/Dev/ML/pokefight/data/pokemon.csv'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                pokemon_path = alt_path
                print(f"Fichier Pokémon trouvé à l'emplacement alternatif: {pokemon_path}")
                break
        
        if not os.path.exists(pokemon_path):
            return False
    
    # Charger les données
    try:
        pokemon_df = pd.read_csv(pokemon_path)
        print(f"Fichier Pokémon chargé avec succès: {len(pokemon_df)} entrées")
        
        if os.path.exists(matchups_path):
            matchups_df = pd.read_csv(matchups_path)
        else:
            print(f"Fichier de matchups non trouvé: {matchups_path}")
            matchups_df = pd.DataFrame()
        
        if os.path.exists(usage_path):
            usage_df = pd.read_csv(usage_path)
        else:
            print(f"Fichier d'utilisation non trouvé: {usage_path}")
            usage_df = pd.DataFrame()
        
        print("Données chargées avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return False
    
    # Initialiser le service de prédiction
    try:
        # Chemins pour le modèle et les métadonnées
        model_path = os.path.join(base_dir, 'models', 'final_model_complete_dataset.joblib')
        metadata_path = os.path.join(base_dir, 'models', 'final_model_complete_dataset_metadata.joblib')
        
        # Vérifier si les fichiers existent
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            # Chercher dans d'autres emplacements possibles
            alternative_model_paths = [
                os.path.join(os.getcwd(), 'models', 'final_model_complete_dataset.joblib'),
                'models/final_model_complete_dataset.joblib',
                'D:/Dev/ML/pokefight/models/final_model_complete_dataset.joblib'
            ]
            
            alternative_metadata_paths = [
                os.path.join(os.getcwd(), 'models', 'final_model_complete_dataset_metadata.joblib'),
                'models/final_model_complete_dataset_metadata.joblib',
                'D:/Dev/ML/pokefight/models/final_model_complete_dataset_metadata.joblib'
            ]
            
            for alt_path in alternative_model_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    print(f"Modèle trouvé à l'emplacement alternatif: {model_path}")
                    break
            
            for alt_path in alternative_metadata_paths:
                if os.path.exists(alt_path):
                    metadata_path = alt_path
                    print(f"Métadonnées trouvées à l'emplacement alternatif: {metadata_path}")
                    break
            
            if not os.path.exists(model_path) or not os.path.exists(metadata_path):
                print(f"Modèle optimisé non trouvé. Veuillez entraîner le modèle d'abord.")
                return False
        
        prediction_service = PredictionService(model_path, metadata_path)
        print("Service de prédiction initialisé avec succès")
        return True
    except Exception as e:
        print(f"Erreur lors de l'initialisation du service de prédiction: {e}")
        return False

def calculate_type_advantage(pokemon1, pokemon2):
    """Calcule l'avantage de type entre deux Pokémon sans utiliser type_chart.csv."""
    
    # Matrice d'efficacité de type codée en dur - dictionnaire avec les multiplicateurs d'efficacité
    type_effectiveness = {
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
    
    # Obtenir les types des Pokémon
    p1_type1 = pokemon1.get('type1', 'normal').lower()
    p1_type2 = pokemon1.get('type2', None)
    if p1_type2 in [None, 'none', '', 'nan', np.nan]:
        p1_type2 = None
    else:
        p1_type2 = p1_type2.lower()
    
    p2_type1 = pokemon2.get('type1', 'normal').lower()
    p2_type2 = pokemon2.get('type2', None)
    if p2_type2 in [None, 'none', '', 'nan', np.nan]:
        p2_type2 = None
    else:
        p2_type2 = p2_type2.lower()
    
    # Calculer l'avantage de type du Pokémon 2 contre le Pokémon 1
    advantage = 1.0
    
    # Type 1 de P2 contre les types de P1
    if p1_type1 and p2_type1:
        if p2_type1 in type_effectiveness and p1_type1 in type_effectiveness.get(p2_type1, {}):
            advantage *= type_effectiveness[p2_type1][p1_type1]
    
    if p1_type2 and p2_type1:
        if p2_type1 in type_effectiveness and p1_type2 in type_effectiveness.get(p2_type1, {}):
            advantage *= type_effectiveness[p2_type1][p1_type2]
    
    # Type 2 de P2 contre les types de P1 (si P2 a un deuxième type)
    if p2_type2:
        if p1_type1:
            if p2_type2 in type_effectiveness and p1_type1 in type_effectiveness.get(p2_type2, {}):
                advantage *= type_effectiveness[p2_type2][p1_type1]
        
        if p1_type2:
            if p2_type2 in type_effectiveness and p1_type2 in type_effectiveness.get(p2_type2, {}):
                advantage *= type_effectiveness[p2_type2][p1_type2]
    
    return advantage

@app.route('/api/pokemon', methods=['GET'])
def get_pokemon_list():
    """Retourne la liste de tous les Pokémon disponibles."""
    global pokemon_df
    
    if pokemon_df is None:
        # Tentative de rechargement des données
        if not load_data_and_model():
            return jsonify({"error": "Données non chargées"}), 500
    
    try:
        # Simplifier la réponse pour éviter les problèmes
        pokemon_list = pokemon_df['name'].tolist()
        return jsonify(pokemon_list)
    except Exception as e:
        print(f"Erreur lors de la récupération de la liste des Pokémon: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/pokemon/<pokemon_name>', methods=['GET'])
def get_pokemon_details(pokemon_name):
    """Retourne les détails d'un Pokémon spécifique."""
    if pokemon_df is None:
        return jsonify({"error": "Données non chargées"}), 500
    
    pokemon = pokemon_df[pokemon_df['name'] == pokemon_name]
    
    if pokemon.empty:
        return jsonify({"error": f"Pokémon {pokemon_name} non trouvé"}), 404
    
    # Convertir en dictionnaire pour la réponse JSON
    pokemon_data = pokemon.iloc[0].to_dict()
    
    return jsonify({"pokemon": pokemon_data})

@app.route('/api/counters/<pokemon_name>', methods=['GET'])
def find_counters(pokemon_name):
    """Trouve les meilleurs contre-Pokémon pour un Pokémon donné."""
    global pokemon_df, prediction_service
    
    if pokemon_df is None or prediction_service is None:
        print(f"Erreur critique: données ou modèle non chargés (pokemon_df={pokemon_df is not None}, prediction_service={prediction_service is not None})")
        # Tentative de rechargement des données
        if not load_data_and_model():
            return jsonify({"error": "Données ou modèle non chargés"}), 500
    
    # Récupérer le nombre de contre-Pokémon à retourner
    top_k = request.args.get('top_k', default=5, type=int)
    min_probability = request.args.get('min_probability', default=0.5, type=float)
    
    print(f"Recherche de contre-Pokémon pour {pokemon_name} avec top_k={top_k}, min_probability={min_probability}")
    
    try:
        # Normaliser le nom du Pokémon (première lettre en majuscule, reste en minuscule)
        normalized_name = pokemon_name.strip().title()
        
        # Vérifier si le Pokémon existe
        if normalized_name not in pokemon_df['name'].values:
            print(f"Pokémon {normalized_name} non trouvé dans la base de données")
            # Essayer de trouver un Pokémon similaire
            similar_pokemon = pokemon_df[pokemon_df['name'].str.lower().str.contains(pokemon_name.lower())]
            if not similar_pokemon.empty:
                normalized_name = similar_pokemon.iloc[0]['name']
                print(f"Pokémon similaire trouvé: {normalized_name}")
            else:
                return jsonify({"error": f"Pokémon {pokemon_name} non trouvé"}), 404
        
        # Récupérer les données du Pokémon cible
        target_pokemon = pokemon_df[pokemon_df['name'] == normalized_name].iloc[0].to_dict()
        print(f"Pokémon cible trouvé: {normalized_name} (Types: {target_pokemon['type1']}/{target_pokemon.get('type2', 'none')})")
        
        # Liste pour stocker les résultats
        counter_results = []
        
        # OPTIMISATION: Limiter le nombre de Pokémon à évaluer pour accélérer le traitement
        # Sélectionner les 100 meilleurs Pokémon par BST (Base Stat Total)
        pokemon_df['bst'] = pokemon_df['hp'] + pokemon_df['attack'] + pokemon_df['defense'] + \
                           pokemon_df['sp_attack'] + pokemon_df['sp_defense'] + pokemon_df['speed']
        
        # Exclure le Pokémon cible
        potential_counters = pokemon_df[pokemon_df['name'] != normalized_name]
        
        # Trier par BST décroissant et prendre les 100 premiers
        potential_counters = potential_counters.sort_values('bst', ascending=False).head(100)
        
        # Compteur pour suivre le nombre de counters trouvés
        counter_count = 0
        
        # Parcourir les Pokémon potentiels comme counters
        for _, counter_row in potential_counters.iterrows():
            counter_pokemon = counter_row.to_dict()
            
            try:
                # Calculer l'avantage de type
                type_advantage = calculate_type_advantage(target_pokemon, counter_pokemon)
                
                # Prédire avec notre modèle optimisé
                try:
                    win_probability, additional_info = prediction_service.predict_win_probability(
                        target_pokemon, counter_pokemon)
                    
                    print(f"Résultat: {counter_pokemon['name']} a {win_probability:.2%} de chances de gagner contre {normalized_name}")
                    
                    # Si la probabilité est suffisante, ajouter aux résultats
                    if win_probability >= min_probability:
                        # Ajouter des raisons (avantages)
                        reasons = additional_info['advantages'].copy()
                        
                        # Ajouter l'avantage de type numérique si significatif
                        if type_advantage > 1.0:
                            if type_advantage >= 4.0:
                                reasons.append(f"Avantage de type x{type_advantage} (super efficace)")
                            else:
                                reasons.append(f"Avantage de type x{type_advantage}")
                        
                        counter_result = {
                            'pokemon': counter_pokemon['name'],
                            'win_probability': float(win_probability),
                            'original_win_probability': float(win_probability),
                            'source': "Modèle optimisé sur dataset complet",
                            'reasons': reasons,
                            'stats': {
                                'hp': int(counter_pokemon['hp']),
                                'attack': int(counter_pokemon['attack']),
                                'defense': int(counter_pokemon['defense']),
                                'sp_attack': int(counter_pokemon['sp_attack']),
                                'sp_defense': int(counter_pokemon['sp_defense']),
                                'speed': int(counter_pokemon['speed']),
                            },
                            'types': [counter_pokemon['type1'], 
                                    counter_pokemon['type2'] if pd.notna(counter_pokemon['type2']) else None]
                        }
                        counter_results.append(counter_result)
                        counter_count += 1
                        
                        # OPTIMISATION: Arrêter si nous avons trouvé suffisamment de counters
                        if counter_count >= top_k * 2:  # Prendre le double pour avoir une marge
                            break
                except Exception as prediction_error:
                    print(f"Erreur lors de la prédiction pour {counter_pokemon['name']}: {repr(prediction_error)}")
                    continue
            except Exception as e:
                print(f"Erreur lors du traitement de {counter_pokemon['name']}: {repr(e)}")
                continue
        
        # Trier par probabilité de victoire décroissante
        counter_results.sort(key=lambda x: x['win_probability'], reverse=True)
        
        # Limiter les résultats
        counter_results = counter_results[:top_k]
        
        # Si aucun résultat n'a été trouvé, ajouter des données de secours
        if not counter_results:
            print(f"Aucun contre-Pokémon trouvé pour {normalized_name}, utilisation de données de secours")
            counter_results = [{
                'pokemon': "Garchomp",
                'win_probability': 0.85,
                'original_win_probability': 0.78,
                'source': "⚠️ Secours (aucun résultat trouvé)",
                'reasons': ["⚠️ Mode secours - Aucun résultat trouvé", "Essayez un autre Pokémon"],
                'types': ["Dragon", "Ground"],
                'stats': {
                    'hp': 108,
                    'attack': 130,
                    'defense': 95,
                    'sp_attack': 80,
                    'sp_defense': 85,
                    'speed': 102,
                }
            }]
        
        print(f"Recherche terminée: {counter_count} contre-Pokémon trouvés pour {normalized_name}")
        print(f"Top counters: {[c['pokemon'] for c in counter_results[:5]]}")
        
        return jsonify({
            "target_pokemon": normalized_name,
            "counters": counter_results
        })
        
    except Exception as e:
        import traceback
        print(f"Erreur lors de la recherche de contre-Pokémon: {repr(e)}")
        print(traceback.format_exc())
        
        # Retourner des données de secours en cas d'erreur
        fallback_data = {
            "target_pokemon": pokemon_name,
            "counters": [{
                'pokemon': "Garchomp",
                'win_probability': 0.85,
                'original_win_probability': 0.78,
                'source': "⚠️ Secours (erreur API)",
                'reasons': ["⚠️ Mode secours - Erreur API", "Essayez de redémarrer l'application"],
                'types': ["Dragon", "Ground"],
                'stats': {
                    'hp': 108,
                    'attack': 130,
                    'defense': 95,
                    'sp_attack': 80,
                    'sp_defense': 85,
                    'speed': 102,
                }
            }]
        }
        return jsonify(fallback_data)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérifie l'état de santé de l'API."""
    global pokemon_df, prediction_service
    
    status = "ok"
    message = "API opérationnelle"
    details = {}
    
    if pokemon_df is None:
        status = "warning"
        message = "Données Pokémon non chargées"
        # Tentative de rechargement des données
        if load_data_and_model():
            message += " (rechargées avec succès)"
        else:
            status = "error"
            message = "Impossible de charger les données Pokémon"
    else:
        details["pokemon_count"] = len(pokemon_df)
    
    if prediction_service is None:
        status = "error" if status != "error" else status
        message += ", Service de prédiction non initialisé"
    else:
        details["model"] = "final_model_complete_dataset"
        details["model_accuracy"] = prediction_service.metadata['metrics']['accuracy']
        details["model_auc"] = prediction_service.metadata['metrics']['auc_roc']
    
    # Vérifier l'accès aux fichiers importants
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    details["file_access"] = {
        "pokemon_csv": os.path.exists(os.path.join(base_dir, 'data', 'pokemon.csv')),
        "model": os.path.exists(os.path.join(base_dir, 'models', 'final_model_complete_dataset.joblib')),
        "metadata": os.path.exists(os.path.join(base_dir, 'models', 'final_model_complete_dataset_metadata.joblib'))
    }
    
    return jsonify({
        "status": status,
        "message": message,
        "details": details
    })

if __name__ == '__main__':
    # Charger les données et le modèle
    if load_data_and_model():
        # Démarrer l'API
        port = int(os.environ.get("PORT", 5001))
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        print("Erreur lors du chargement des données ou du modèle. L'API ne peut pas démarrer.")
else:
    # S'assurer que les données sont chargées même si le module est importé
    print("Chargement des données pour l'API...")
    load_data_and_model() 