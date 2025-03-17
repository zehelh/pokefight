import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import sys
import requests
import json
import pandas as pd
import os.path
import time

# Garantir que le répertoire de travail est correct
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')

app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

# Vérifier et afficher les chemins
print(f"Répertoire des templates: {template_dir}")
print(f"Répertoire des fichiers statiques: {static_dir}")
print(f"Templates disponibles: {os.listdir(template_dir) if os.path.exists(template_dir) else 'Aucun'}")
print(f"Fichiers JS disponibles: {os.listdir(os.path.join(static_dir, 'js')) if os.path.exists(os.path.join(static_dir, 'js')) else 'Aucun'}")

# Configuration
API_URL = os.environ.get('API_URL', 'http://localhost:5001')

# Ajouter une fonction de requête avec retry
def make_api_request(url, method='get', max_retries=3, **kwargs):
    """Effectue une requête API avec retry en cas d'échec"""
    from requests.exceptions import RequestException
    
    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                response = requests.get(url, timeout=5, **kwargs)
            elif method.lower() == 'post':
                response = requests.post(url, timeout=5, **kwargs)
            else:
                raise ValueError(f"Méthode non supportée: {method}")
            
            response.raise_for_status()
            return response
        except RequestException as e:
            app.logger.warning(f"Tentative {attempt+1}/{max_retries} échouée: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(0.5)  # Attendre un peu avant de réessayer

# Création des répertoires pour les fichiers statiques manquants
import os

# Vérifier si le répertoire CSS existe, sinon le créer
css_dir = os.path.join(current_dir, 'static', 'css')
if not os.path.exists(css_dir):
    os.makedirs(css_dir, exist_ok=True)
    print(f"Création du répertoire CSS: {css_dir}")

# Créer style.css s'il n'existe pas
style_css_path = os.path.join(css_dir, 'style.css')
if not os.path.exists(style_css_path):
    print(f"Création du fichier style.css")
    with open(style_css_path, 'w') as f:
        f.write("""/* Styles minimalistes pour l'application */
body { font-family: sans-serif; margin: 0; padding: 0; }
.container { width: 90%; max-width: 1200px; margin: 0 auto; }
nav { background-color: #e3350d; color: white; padding: 1rem 0; }
.button { background-color: #e3350d; color: white; padding: 0.5rem 1rem; text-decoration: none; }
footer { background-color: #333; color: white; padding: 1rem 0; text-align: center; }
.loader { border: 5px solid #f3f3f3; border-top: 5px solid #e3350d; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
""")

# Créer le dossier img si nécessaire (utilisé dans les templates)
img_dir = os.path.join(current_dir, 'static', 'img')
if not os.path.exists(img_dir):
    os.makedirs(img_dir, exist_ok=True)
    print(f"Création du répertoire img: {img_dir}")

@app.route('/')
def index():
    """
    Page d'accueil
    """
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Erreur lors du rendu de index.html: {e}")
        return "<h1>Erreur lors du chargement de la page d'accueil</h1><p>Vérifiez les logs pour plus d'informations.</p>"

@app.route('/search')
def search():
    """
    Page de recherche de contre-Pokémon
    """
    try:
        app.logger.info("Tentative de rendu de search.html")
        return render_template('search.html')
    except Exception as e:
        app.logger.error(f"Erreur lors du rendu de search.html: {e}")
        # Créer une page de recherche minimale en cas d'erreur
        return """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Recherche de Contre-Pokémon - PokéFight</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100">
            <nav class="bg-red-600 text-white p-4">
                <div class="container mx-auto flex justify-between">
                    <a href="/" class="font-bold text-xl">PokéFight</a>
                    <div>
                        <a href="/" class="mr-4">Accueil</a>
                        <a href="/search" class="mr-4 underline">Recherche</a>
                        <a href="/compare">Comparaison</a>
                    </div>
                </div>
            </nav>
            <div class="container mx-auto p-4">
                <h1 class="text-2xl font-bold mb-4">Recherche de contre-Pokémon</h1>
                <div class="mb-4">
                    <input type="text" id="pokemon-search" placeholder="Nom du Pokémon..." class="p-2 border rounded">
                    <button onclick="searchCounters()" class="bg-red-600 text-white p-2 rounded">Rechercher</button>
                </div>
                <div id="results-container"></div>
            </div>
            <script>
            function searchCounters() {
                const pokemonName = document.getElementById('pokemon-search').value;
                if (!pokemonName) return;
                
                fetch(`/api/counters/${pokemonName}?top_k=5`)
                    .then(response => response.json())
                    .then(data => {
                        let html = '<h2>Résultats:</h2>';
                        if (data.length === 0) {
                            html += '<p>Aucun résultat trouvé</p>';
                        } else {
                            data.forEach(counter => {
                                html += `<div class="p-2 border mb-2">
                                    <h3>${counter.pokemon}</h3>
                                    <p>Probabilité: ${Math.round(counter.win_probability * 100)}%</p>
                                </div>`;
                            });
                        }
                        document.getElementById('results-container').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('results-container').innerHTML = 
                            `<p class="text-red-600">Erreur: ${error.message}</p>`;
                    });
            }
            </script>
        </body>
        </html>
        """

@app.route('/compare')
def compare():
    """
    Page de comparaison de deux Pokémon
    """
    try:
        return render_template('compare.html')
    except Exception as e:
        app.logger.error(f"Erreur lors du rendu de compare.html: {e}")
        return "<h1>Erreur lors du chargement de la page de comparaison</h1><p>Vérifiez les logs pour plus d'informations.</p>"

@app.route('/api/pokemon')
def get_pokemon_list():
    """
    Proxy pour l'API de liste des Pokémon
    """
    try:
        app.logger.info(f"Tentative de récupération de la liste des Pokémon depuis {API_URL}/api/pokemon")
        response = requests.get(f"{API_URL}/api/pokemon")
        
        # Afficher des informations sur la réponse pour le débogage
        app.logger.info(f"Statut de la réponse: {response.status_code}")
        app.logger.info(f"Headers: {response.headers}")
        
        # Vérifier si la réponse est au format JSON
        content_type = response.headers.get('Content-Type', '')
        app.logger.info(f"Content-Type: {content_type}")
        
        if 'application/json' not in content_type:
            app.logger.warning(f"La réponse n'est pas au format JSON: {content_type}")
            # Tenter de convertir la réponse en JSON quand même
            try:
                data = response.json()
                app.logger.info(f"Conversion en JSON réussie malgré le Content-Type incorrect")
            except:
                app.logger.error(f"Impossible de convertir la réponse en JSON: {response.text[:200]}...")
                return jsonify({"error": "Format de réponse invalide"}), 500
        else:
            data = response.json()
        
        # Vérifier le type de données reçu
        if isinstance(data, list):
            app.logger.info(f"Liste de {len(data)} Pokémon reçue")
            return jsonify(data)
        elif isinstance(data, dict) and 'pokemon' in data:
            # Si l'API renvoie un dictionnaire avec une clé 'pokemon' contenant la liste
            pokemon_list = data['pokemon']
            app.logger.info(f"Liste de {len(pokemon_list)} Pokémon extraite du dictionnaire")
            return jsonify(pokemon_list)
        else:
            # Essayer de trouver une liste dans la réponse
            app.logger.warning(f"Format de données inattendu: {type(data)}")
            app.logger.info(f"Contenu de la réponse: {str(data)[:200]}...")
            
            # Créer une liste de secours avec les Pokémon les plus populaires
            fallback_list = [
                "Pikachu", "Charizard", "Bulbasaur", "Squirtle", "Mewtwo", 
                "Gengar", "Gyarados", "Dragonite", "Snorlax", "Eevee",
                "Venusaur", "Blastoise", "Garchomp", "Lucario", "Tyranitar"
            ]
            app.logger.info(f"Utilisation d'une liste de secours de {len(fallback_list)} Pokémon")
            return jsonify(fallback_list)
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération de la liste des Pokémon: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/counters/<pokemon_name>')
def get_counters(pokemon_name):
    """
    Proxy pour l'API de recherche de contre-Pokémon avec priorité au modèle
    """
    try:
        top_k = request.args.get('top_k', 5)
        min_probability = request.args.get('min_probability', 0.5)
        app.logger.info(f"Recherche de contre-Pokémon pour {pokemon_name} avec top_k={top_k}")
        
        # ÉTAPE 1: Essayer d'abord le modèle simplifié intégré
        try:
            app.logger.info("Tentative d'utilisation du modèle simplifié intégré")
            model_response = get_model_counters(pokemon_name)
            
            # Vérifier si la réponse est valide (pas d'erreur)
            if model_response.status_code == 200:
                app.logger.info("Modèle simplifié utilisé avec succès")
                return model_response
            else:
                app.logger.warning("Échec du modèle simplifié, passage à l'API principale")
        except Exception as model_error:
            app.logger.error(f"Erreur avec le modèle simplifié: {model_error}")
        
        # ÉTAPE 2: Essayer ensuite l'API principale
        try:
            # S'assurer que le top_k est passé comme paramètre de requête
            url = f"{API_URL}/api/counters/{pokemon_name}?top_k={top_k}&min_probability={min_probability}"
            
            app.logger.info(f"Envoi de la requête à {url} avec un timeout de 30 secondes")
            response = requests.get(url, timeout=30)  # Réduire le timeout à 30 secondes
            
            # Traiter la réponse
            data = response.json()
            app.logger.info(f"Données reçues avec succès: {type(data)}")
            
            # Vérifier le format des données et les transformer si nécessaire
            if isinstance(data, dict) and 'counters' in data:
                counters = data['counters']
                app.logger.info(f"Liste de {len(counters)} contre-Pokémon extraite du dictionnaire")
                
                # Ajouter le champ original_win_probability si absent et s'assurer que la source est bien indiquée
                for counter in counters:
                    if 'original_win_probability' not in counter and 'win_probability' in counter:
                        counter['original_win_probability'] = counter['win_probability']
                    
                    # S'assurer que la source est bien indiquée
                    if 'source' not in counter:
                        counter['source'] = "Modèle prédictif (API)"
                
                return jsonify(data)  # Retourner l'objet complet avec target_pokemon
            elif isinstance(data, list):
                app.logger.info(f"Liste de {len(data)} contre-Pokémon reçue directement")
                
                # Ajouter le champ original_win_probability si absent et s'assurer que la source est bien indiquée
                for counter in data:
                    if 'original_win_probability' not in counter and 'win_probability' in counter:
                        counter['original_win_probability'] = counter['win_probability']
                    
                    # S'assurer que la source est bien indiquée
                    if 'source' not in counter:
                        counter['source'] = "Modèle prédictif (API)"
                
                return jsonify({
                    'target_pokemon': pokemon_name,
                    'counters': data
                })
            else:
                app.logger.warning(f"Format de données inattendu: {type(data)}")
                # Utiliser le mode de secours statistique
                raise Exception("Format de données inattendu, utilisation du mode de secours")
                
        except Exception as api_error:
            app.logger.error(f"Erreur API: {api_error}")
            app.logger.info("Utilisation du mode de secours basé sur les statistiques")
            
            # ÉTAPE 3: Utiliser des données de secours basées sur les statistiques brutes
            try:
                pokemon_df = get_pokemon_dataframe()
                if pokemon_df is not None:
                    # Trouver le Pokémon cible
                    target = pokemon_df[pokemon_df['name'].str.lower() == pokemon_name.lower()]
                    if not target.empty:
                        target_pokemon = target.iloc[0]
                        
                        # Calculer le BST (Base Stat Total) pour tous les Pokémon
                        pokemon_df['bst'] = pokemon_df['hp'] + pokemon_df['attack'] + pokemon_df['defense'] + \
                                           pokemon_df['sp_attack'] + pokemon_df['sp_defense'] + pokemon_df['speed']
                        
                        # Exclure le Pokémon cible
                        potential_counters = pokemon_df[pokemon_df['name'] != pokemon_name]
                        
                        # Trier par BST décroissant
                        potential_counters = potential_counters.sort_values('bst', ascending=False)
                        
                        # Limiter à top_k
                        potential_counters = potential_counters.head(int(top_k))
                        
                        # Créer la liste de counters
                        counters = []
                        for _, counter in potential_counters.iterrows():
                            # Calculer une probabilité basée sur la différence de BST
                            target_bst = target_pokemon['hp'] + target_pokemon['attack'] + target_pokemon['defense'] + \
                                        target_pokemon['sp_attack'] + target_pokemon['sp_defense'] + target_pokemon['speed']
                            counter_bst = counter['bst']
                            bst_diff = counter_bst - target_bst
                            
                            # Normaliser la probabilité entre 0.5 et 0.9
                            win_prob = min(0.9, max(0.5, 0.7 + (bst_diff / 200)))
                            
                            # Déterminer les avantages statistiques
                            advantages = []
                            
                            if counter['speed'] > target_pokemon['speed'] + 10:
                                advantages.append(f"Plus rapide (+{counter['speed'] - target_pokemon['speed']})")
                            
                            if counter['attack'] > target_pokemon['attack'] + 20:
                                advantages.append(f"Attaque supérieure (+{counter['attack'] - target_pokemon['attack']})")
                            
                            if counter['defense'] > target_pokemon['defense'] + 20:
                                advantages.append(f"Défense supérieure (+{counter['defense'] - target_pokemon['defense']})")
                            
                            if counter['sp_attack'] > target_pokemon['sp_attack'] + 20:
                                advantages.append(f"Attaque Spéciale supérieure (+{counter['sp_attack'] - target_pokemon['sp_attack']})")
                            
                            if counter['sp_defense'] > target_pokemon['sp_defense'] + 20:
                                advantages.append(f"Défense Spéciale supérieure (+{counter['sp_defense'] - target_pokemon['sp_defense']})")
                            
                            if counter['hp'] > target_pokemon['hp'] + 20:
                                advantages.append(f"PV supérieurs (+{counter['hp'] - target_pokemon['hp']})")
                            
                            if not advantages:
                                advantages.append("Statistiques globales supérieures")
                            
                            counters.append({
                                'pokemon': counter['name'],
                                'win_probability': float(win_prob),
                                'original_win_probability': float(win_prob),
                                'source': "Analyse statistique (mode secours)",
                                'reasons': advantages,
                                'types': [counter['type1'], counter['type2'] if pd.notna(counter['type2']) else None],
                                'stats': {
                                    'hp': int(counter['hp']),
                                    'attack': int(counter['attack']),
                                    'defense': int(counter['defense']),
                                    'sp_attack': int(counter['sp_attack']),
                                    'sp_defense': int(counter['sp_defense']),
                                    'speed': int(counter['speed']),
                                }
                            })
                        
                        if counters:
                            return jsonify({
                                'target_pokemon': pokemon_name,
                                'counters': counters
                            })
            except Exception as csv_error:
                app.logger.error(f"Erreur lors de la génération de counters statistiques: {csv_error}")
            
            # ÉTAPE 4: Utiliser des données de secours si tout échoue
            fallback_counters = [
                {
                    "pokemon": "Garchomp",
                    "win_probability": 0.85,
                    "original_win_probability": 0.78,
                    "source": "⚠️ Secours (API indisponible)",
                    "reasons": ["⚠️ Mode secours - API inaccessible", "Essayez de redémarrer l'application"],
                    "types": ["Dragon", "Ground"],
                    "stats": {
                        "hp": 108,
                        "attack": 130,
                        "defense": 95,
                        "sp_attack": 80,
                        "sp_defense": 85,
                        "speed": 102,
                    }
                }
            ]
            return jsonify({
                'target_pokemon': pokemon_name,
                'counters': fallback_counters
            })
            
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération des contre-Pokémon pour {pokemon_name}: {e}")
        return jsonify({
            'target_pokemon': pokemon_name,
            'counters': []
        }), 500

@app.route('/api/compare/<pokemon1>/<pokemon2>')
def compare_pokemon(pokemon1, pokemon2):
    """
    Proxy pour l'API de comparaison de Pokémon
    """
    try:
        response = requests.get(f"{API_URL}/api/compare/{pokemon1}/{pokemon2}")
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        app.logger.error(f"Erreur lors de la comparaison entre {pokemon1} et {pokemon2}: {e}")
        return jsonify({"error": str(e)}), 500

# Fonction pour déboguer l'API
@app.route('/debug/api')
def debug_api():
    try:
        # Vérifier si l'API est accessible
        health_response = requests.get(f"{API_URL}/api/health")
        health_status = health_response.json() if health_response.ok else {"status": "error", "message": health_response.text}
        
        # Récupérer la liste des Pokémon
        pokemon_response = requests.get(f"{API_URL}/api/pokemon")
        pokemon_list = pokemon_response.json() if pokemon_response.ok else {"error": pokemon_response.text}
        
        debug_info = {
            "api_url": API_URL,
            "health_status": health_status,
            "pokemon_list_status": "ok" if pokemon_response.ok else "error",
            "pokemon_count": len(pokemon_list) if pokemon_response.ok and isinstance(pokemon_list, list) else 0
        }
        
        return jsonify(debug_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ajouter une route pour récupérer la liste des Pokémon directement depuis le fichier CSV
@app.route('/api/local/pokemon')
def get_local_pokemon_list():
    """
    Récupère la liste des Pokémon directement depuis le fichier CSV local
    ou redirige vers l'API principale en cas d'échec
    """
    try:
        # Recherche du fichier pokemon.csv dans tous les répertoires possibles
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        possible_paths = [
            os.path.join(base_dir, 'data', 'pokemon.csv'),  # Chemin absolu basé sur l'emplacement du module
            os.path.join(os.getcwd(), 'data', 'pokemon.csv'),  # Chemin absolu basé sur le répertoire de travail
            'data/pokemon.csv',  # Chemin relatif au répertoire de travail
            os.path.join(os.path.dirname(os.getcwd()), 'data', 'pokemon.csv'),  # Un niveau au-dessus
            'D:/Dev/ML/pokefight/data/pokemon.csv'  # Chemin absolu spécifique
        ]
        
        # Trouver le premier chemin valide
        csv_path = None
        for path in possible_paths:
            app.logger.info(f"Essai de chemin CSV: {path}, existe: {os.path.exists(path)}")
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            app.logger.error("Impossible de trouver le fichier pokemon.csv - Utilisation d'une liste de secours")
            # Liste de secours en cas de fichier manquant
            fallback_list = [
                "Pikachu", "Charizard", "Bulbasaur", "Squirtle", "Mewtwo", 
                "Gengar", "Gyarados", "Dragonite", "Snorlax", "Eevee",
                "Venusaur", "Blastoise", "Garchomp", "Lucario", "Tyranitar",
                "Jigglypuff", "Arcanine", "Alakazam", "Machamp", "Vaporeon"
            ]
            app.logger.info(f"Retour d'une liste de secours de {len(fallback_list)} Pokémon")
            return jsonify(fallback_list)
        
        # Lire le fichier CSV
        app.logger.info(f"Lecture du fichier CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Extraire la liste des noms de Pokémon
        if 'name' in df.columns:
            pokemon_list = df['name'].tolist()
        elif 'Name' in df.columns:
            pokemon_list = df['Name'].tolist()
        else:
            app.logger.error("Colonne 'name' ou 'Name' introuvable dans le fichier CSV")
            # Liste de secours en cas d'erreur
            fallback_list = [
                "Pikachu", "Charizard", "Bulbasaur", "Squirtle", "Mewtwo", 
                "Gengar", "Gyarados", "Dragonite", "Snorlax", "Eevee"
            ]
            return jsonify(fallback_list)
        
        app.logger.info(f"Liste de {len(pokemon_list)} Pokémon extraite du fichier CSV")
        return jsonify(pokemon_list)
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération de la liste des Pokémon: {str(e)}")
        # Liste de secours en cas d'erreur
        fallback_list = [
            "Pikachu", "Charizard", "Bulbasaur", "Squirtle", "Mewtwo", 
            "Gengar", "Gyarados", "Dragonite", "Snorlax", "Eevee"
        ]
        app.logger.info(f"Retour d'une liste de secours après erreur: {str(e)}")
        return jsonify(fallback_list)

@app.route('/api/pokemon/<pokemon_name>')
def get_pokemon_details(pokemon_name):
    """
    Proxy pour l'API de détails d'un Pokémon
    """
    try:
        app.logger.info(f"Récupération des détails pour {pokemon_name}")
        
        # Essayer d'abord l'API principale
        try:
            response = make_api_request(f"{API_URL}/api/pokemon/{pokemon_name}")
            data = response.json()
            app.logger.info(f"Détails récupérés avec succès pour {pokemon_name}")
            return jsonify(data)
        except Exception as api_error:
            app.logger.warning(f"Erreur lors de la récupération depuis l'API: {api_error}")
            
            # Essayer de récupérer les données depuis le fichier CSV local
            try:
                # Recherche du fichier pokemon.csv
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                possible_paths = [
                    os.path.join(base_dir, 'data', 'pokemon.csv'),
                    os.path.join(os.getcwd(), 'data', 'pokemon.csv'),
                    'data/pokemon.csv',
                    os.path.join(os.path.dirname(os.getcwd()), 'data', 'pokemon.csv'),
                    'D:/Dev/ML/pokefight/data/pokemon.csv'
                ]
                
                # Trouver le premier chemin valide
                csv_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        csv_path = path
                        break
                
                if csv_path:
                    # Charger le CSV et trouver le Pokémon
                    df = pd.read_csv(csv_path)
                    pokemon = df[df['name'].str.lower() == pokemon_name.lower()]
                    
                    if not pokemon.empty:
                        # Convertir en dictionnaire pour la réponse JSON
                        pokemon_data = pokemon.iloc[0].to_dict()
                        app.logger.info(f"Détails récupérés depuis le CSV pour {pokemon_name}")
                        return jsonify({"pokemon": pokemon_data})
            except Exception as csv_error:
                app.logger.error(f"Erreur lors de la récupération depuis le CSV: {csv_error}")
            
            # Si toutes les tentatives échouent, renvoyer des données de secours
            app.logger.warning(f"Utilisation de données de secours pour {pokemon_name}")
            fallback_data = {
                "pokemon": {
                    "name": pokemon_name,
                    "type1": "normal",
                    "type2": None,
                    "hp": 100,
                    "attack": 100,
                    "defense": 100,
                    "sp_attack": 100,
                    "sp_defense": 100,
                    "speed": 100
                }
            }
            return jsonify(fallback_data)
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération des détails pour {pokemon_name}: {e}")
        return jsonify({"error": str(e), "pokemon": None}), 500

def get_pokemon_dataframe():
    """Récupère le dataframe des Pokémon depuis le fichier CSV"""
    try:
        # Recherche du fichier pokemon.csv dans tous les répertoires possibles
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        possible_paths = [
            os.path.join(base_dir, 'data', 'pokemon.csv'),  # Chemin absolu basé sur l'emplacement du module
            os.path.join(os.getcwd(), 'data', 'pokemon.csv'),  # Chemin absolu basé sur le répertoire de travail
            'data/pokemon.csv',  # Chemin relatif au répertoire de travail
            os.path.join(os.path.dirname(os.getcwd()), 'data', 'pokemon.csv'),  # Un niveau au-dessus
            'D:/Dev/ML/pokefight/data/pokemon.csv'  # Chemin absolu spécifique
        ]
        
        # Trouver le premier chemin valide
        csv_path = None
        for path in possible_paths:
            app.logger.info(f"Essai de chemin CSV: {path}, existe: {os.path.exists(path)}")
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            app.logger.error("Impossible de trouver le fichier pokemon.csv")
            return None
        
        # Lire le fichier CSV
        app.logger.info(f"Lecture du fichier CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        app.logger.error(f"Erreur lors de la lecture du fichier CSV: {str(e)}")
        return None

@app.route('/api/pokemon/types')
def get_pokemon_types():
    """Retourne les types de tous les Pokémon en une seule requête."""
    try:
        # Vérifier si le fichier pokemon.csv existe
        pokemon_df = get_pokemon_dataframe()
        
        if pokemon_df is None:
            return jsonify({"error": "Fichier Pokémon non trouvé"}), 404
        
        # Créer un dictionnaire avec les types de chaque Pokémon
        types_dict = {}
        for _, row in pokemon_df.iterrows():
            pokemon_name = row['name']
            type1 = row['type1'] if pd.notna(row['type1']) else None
            type2 = row['type2'] if pd.notna(row['type2']) else None
            
            # Ajouter les types au dictionnaire
            types_dict[pokemon_name] = [type1, type2]
        
        return jsonify(types_dict)
    except Exception as e:
        app.logger.error(f"Erreur lors de la récupération des types: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/local/counters/<pokemon_name>')
def get_local_counters(pokemon_name):
    """
    Génère des counters basés uniquement sur les statistiques, sans passer par l'API principale
    """
    try:
        top_k = request.args.get('top_k', 5)
        app.logger.info(f"Génération locale de counters pour {pokemon_name} avec top_k={top_k}")
        
        # Récupérer les données Pokémon
        pokemon_df = get_pokemon_dataframe()
        if pokemon_df is None:
            return jsonify({"error": "Fichier Pokémon non trouvé"}), 404
        
        # Trouver le Pokémon cible
        target = pokemon_df[pokemon_df['name'].str.lower() == pokemon_name.lower()]
        if target.empty:
            return jsonify({"error": f"Pokémon {pokemon_name} non trouvé"}), 404
        
        target_pokemon = target.iloc[0]
        
        # Calculer le BST (Base Stat Total) pour tous les Pokémon
        pokemon_df['bst'] = pokemon_df['hp'] + pokemon_df['attack'] + pokemon_df['defense'] + \
                           pokemon_df['sp_attack'] + pokemon_df['sp_defense'] + pokemon_df['speed']
        
        # Exclure le Pokémon cible
        potential_counters = pokemon_df[pokemon_df['name'] != pokemon_name]
        
        # Trier par BST décroissant
        potential_counters = potential_counters.sort_values('bst', ascending=False)
        
        # Limiter à top_k
        potential_counters = potential_counters.head(int(top_k))
        
        # Créer la liste de counters
        counters = []
        for _, counter in potential_counters.iterrows():
            # Calculer une probabilité basée sur la différence de BST
            target_bst = target_pokemon['hp'] + target_pokemon['attack'] + target_pokemon['defense'] + \
                        target_pokemon['sp_attack'] + target_pokemon['sp_defense'] + target_pokemon['speed']
            counter_bst = counter['bst']
            bst_diff = counter_bst - target_bst
            
            # Normaliser la probabilité entre 0.5 et 0.9
            win_prob = min(0.9, max(0.5, 0.7 + (bst_diff / 200)))
            
            # Déterminer les avantages statistiques
            advantages = []
            
            if counter['speed'] > target_pokemon['speed'] + 10:
                advantages.append(f"Plus rapide (+{counter['speed'] - target_pokemon['speed']})")
            
            if counter['attack'] > target_pokemon['attack'] + 20:
                advantages.append(f"Attaque supérieure (+{counter['attack'] - target_pokemon['attack']})")
            
            if counter['defense'] > target_pokemon['defense'] + 20:
                advantages.append(f"Défense supérieure (+{counter['defense'] - target_pokemon['defense']})")
            
            if counter['sp_attack'] > target_pokemon['sp_attack'] + 20:
                advantages.append(f"Attaque Spéciale supérieure (+{counter['sp_attack'] - target_pokemon['sp_attack']})")
            
            if counter['sp_defense'] > target_pokemon['sp_defense'] + 20:
                advantages.append(f"Défense Spéciale supérieure (+{counter['sp_defense'] - target_pokemon['sp_defense']})")
            
            if counter['hp'] > target_pokemon['hp'] + 20:
                advantages.append(f"PV supérieurs (+{counter['hp'] - target_pokemon['hp']})")
            
            if not advantages:
                advantages.append("Statistiques globales supérieures")
            
            counters.append({
                'pokemon': counter['name'],
                'win_probability': float(win_prob),
                'original_win_probability': float(win_prob),
                'source': "Analyse statistique (mode rapide)",
                'reasons': advantages,
                'types': [counter['type1'], counter['type2'] if pd.notna(counter['type2']) else None],
                'stats': {
                    'hp': int(counter['hp']),
                    'attack': int(counter['attack']),
                    'defense': int(counter['defense']),
                    'sp_attack': int(counter['sp_attack']),
                    'sp_defense': int(counter['sp_defense']),
                    'speed': int(counter['speed']),
                }
            })
        
        return jsonify({
            'target_pokemon': pokemon_name,
            'counters': counters
        })
    except Exception as e:
        app.logger.error(f"Erreur lors de la génération locale de counters pour {pokemon_name}: {e}")
        return jsonify({
            'target_pokemon': pokemon_name,
            'counters': []
        }), 500

@app.route('/api/model/counters/<pokemon_name>')
def get_model_counters(pokemon_name):
    """
    Implémentation simplifiée du modèle directement dans l'application web
    pour éviter les problèmes de timeout avec l'API externe
    """
    try:
        top_k = request.args.get('top_k', 5)
        app.logger.info(f"Génération de counters avec modèle simplifié pour {pokemon_name} avec top_k={top_k}")
        
        # Récupérer les données Pokémon
        pokemon_df = get_pokemon_dataframe()
        if pokemon_df is None:
            return jsonify({"error": "Fichier Pokémon non trouvé"}), 404
        
        # Trouver le Pokémon cible
        target = pokemon_df[pokemon_df['name'].str.lower() == pokemon_name.lower()]
        if target.empty:
            return jsonify({"error": f"Pokémon {pokemon_name} non trouvé"}), 404
        
        target_pokemon = target.iloc[0]
        
        # Calculer le BST (Base Stat Total) pour tous les Pokémon
        pokemon_df['bst'] = pokemon_df['hp'] + pokemon_df['attack'] + pokemon_df['defense'] + \
                           pokemon_df['sp_attack'] + pokemon_df['sp_defense'] + pokemon_df['speed']
        
        # Exclure le Pokémon cible
        potential_counters = pokemon_df[pokemon_df['name'] != pokemon_name]
        
        # Créer un modèle simplifié basé sur les statistiques et les types
        # Calculer un score pour chaque Pokémon potentiel
        counters = []
        
        for _, counter in potential_counters.iterrows():
            # Calculer la différence de BST
            target_bst = target_pokemon['hp'] + target_pokemon['attack'] + target_pokemon['defense'] + \
                        target_pokemon['sp_attack'] + target_pokemon['sp_defense'] + target_pokemon['speed']
            counter_bst = counter['bst']
            bst_diff = counter_bst - target_bst
            
            # Calculer l'avantage de type (simplifié)
            type_advantage = 1.0
            
            # Vérifier les avantages de type connus
            target_types = [target_pokemon['type1']]
            if pd.notna(target_pokemon['type2']):
                target_types.append(target_pokemon['type2'])
            
            counter_types = [counter['type1']]
            if pd.notna(counter['type2']):
                counter_types.append(counter['type2'])
            
            # Dictionnaire simplifié des avantages de type
            type_chart = {
                'fire': {'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'steel': 2.0},
                'water': {'fire': 2.0, 'ground': 2.0, 'rock': 2.0},
                'electric': {'water': 2.0, 'flying': 2.0},
                'grass': {'water': 2.0, 'ground': 2.0, 'rock': 2.0},
                'ice': {'grass': 2.0, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0},
                'fighting': {'normal': 2.0, 'ice': 2.0, 'rock': 2.0, 'dark': 2.0, 'steel': 2.0},
                'poison': {'grass': 2.0, 'fairy': 2.0},
                'ground': {'fire': 2.0, 'electric': 2.0, 'poison': 2.0, 'rock': 2.0, 'steel': 2.0},
                'flying': {'grass': 2.0, 'fighting': 2.0, 'bug': 2.0},
                'psychic': {'fighting': 2.0, 'poison': 2.0},
                'bug': {'grass': 2.0, 'psychic': 2.0, 'dark': 2.0},
                'rock': {'fire': 2.0, 'ice': 2.0, 'flying': 2.0, 'bug': 2.0},
                'ghost': {'psychic': 2.0, 'ghost': 2.0},
                'dragon': {'dragon': 2.0},
                'dark': {'psychic': 2.0, 'ghost': 2.0},
                'steel': {'ice': 2.0, 'rock': 2.0, 'fairy': 2.0},
                'fairy': {'fighting': 2.0, 'dragon': 2.0, 'dark': 2.0}
            }
            
            # Calculer l'avantage de type
            for c_type in counter_types:
                if c_type and c_type.lower() in type_chart:
                    for t_type in target_types:
                        if t_type and t_type.lower() in type_chart[c_type.lower()]:
                            type_advantage *= type_chart[c_type.lower()][t_type.lower()]
            
            # Calculer un score basé sur les statistiques et l'avantage de type
            # Formule: (BST diff / 100) + (type advantage - 1) * 2 + (speed diff / 50)
            speed_diff = counter['speed'] - target_pokemon['speed']
            score = (bst_diff / 100) + (type_advantage - 1) * 2 + (speed_diff / 50)
            
            # Convertir le score en probabilité de victoire (entre 0.5 et 0.95)
            win_prob = min(0.95, max(0.5, 0.7 + score / 10))
            
            # Déterminer les avantages
            advantages = []
            
            # Avantage de type
            if type_advantage > 1.0:
                advantages.append(f"Avantage de type (x{type_advantage:.1f})")
            
            # Avantages statistiques
            if counter['speed'] > target_pokemon['speed'] + 10:
                advantages.append(f"Plus rapide (+{counter['speed'] - target_pokemon['speed']})")
            
            if counter['attack'] > target_pokemon['attack'] + 20:
                advantages.append(f"Attaque supérieure (+{counter['attack'] - target_pokemon['attack']})")
            
            if counter['defense'] > target_pokemon['defense'] + 20:
                advantages.append(f"Défense supérieure (+{counter['defense'] - target_pokemon['defense']})")
            
            if counter['sp_attack'] > target_pokemon['sp_attack'] + 20:
                advantages.append(f"Attaque Spéciale supérieure (+{counter['sp_attack'] - target_pokemon['sp_attack']})")
            
            if counter['sp_defense'] > target_pokemon['sp_defense'] + 20:
                advantages.append(f"Défense Spéciale supérieure (+{counter['sp_defense'] - target_pokemon['sp_defense']})")
            
            if counter['hp'] > target_pokemon['hp'] + 20:
                advantages.append(f"PV supérieurs (+{counter['hp'] - target_pokemon['hp']})")
            
            if not advantages:
                advantages.append("Bon équilibre général contre ce Pokémon")
            
            counters.append({
                'pokemon': counter['name'],
                'win_probability': float(win_prob),
                'original_win_probability': float(win_prob),
                'source': "Modèle simplifié (intégré)",
                'reasons': advantages,
                'types': [counter['type1'], counter['type2'] if pd.notna(counter['type2']) else None],
                'stats': {
                    'hp': int(counter['hp']),
                    'attack': int(counter['attack']),
                    'defense': int(counter['defense']),
                    'sp_attack': int(counter['sp_attack']),
                    'sp_defense': int(counter['sp_defense']),
                    'speed': int(counter['speed']),
                }
            })
        
        # Trier par probabilité de victoire
        counters.sort(key=lambda x: x['win_probability'], reverse=True)
        
        # Limiter au nombre demandé
        counters = counters[:int(top_k)]
        
        return jsonify({
            'target_pokemon': pokemon_name,
            'counters': counters
        })
    except Exception as e:
        app.logger.error(f"Erreur lors de la génération de counters avec modèle simplifié pour {pokemon_name}: {e}")
        return jsonify({
            'target_pokemon': pokemon_name,
            'counters': []
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 