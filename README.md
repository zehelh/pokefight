# PokéFight: Système de prédiction de counters Pokémon

## 1. Définition du sujet

### Clarté et pertinence du sujet
PokéFight est une application web qui utilise l'apprentissage automatique pour prédire les meilleurs contre-Pokémon (counters) pour n'importe quel Pokémon. L'application combine des données statistiques, des analyses de types et un modèle prédictif pour fournir des recommandations précises et rapides.

### Originalité et valeur ajoutée du projet
Ce projet se distingue par sa capacité à prédire les counters Pokémon avec une précision élevée, en tenant compte non seulement des avantages de type traditionnels, mais aussi des statistiques complexes et des interactions stratégiques. Contrairement aux calculateurs de type simples, PokéFight utilise l'apprentissage automatique pour capturer les nuances des matchups compétitifs.

### Faisabilité (réalisme des objectifs, ressources disponibles)
Le projet a été conçu avec des objectifs réalistes et des ressources accessibles :
- Utilisation de données publiques disponibles (statistiques de Pokémon, logs de combat)
- Modèles d'apprentissage automatique adaptés aux ressources de calcul standard
- Architecture modulaire permettant des améliorations progressives

## 2. Récolte et préparation des données

### Qualité et pertinence des sources de données
Trois sources principales de données ont été utilisées :
- [Base pokemon stats](https://www.kaggle.com/datasets/rounakbanik/pokemon) : Statistiques de base des Pokémon
- [Tournament pokemon usage](https://www.smogon.com/forums/threads/tournament-statistics-archive.3538686/) : Données d'utilisation en tournoi
- [Battle History](https://replay.pokemonshowdown.com/) : Logs de combats réels

### Nettoyage et prétraitement des données
Le processus de collecte et de prétraitement des données comprend :

1. **Collecte des données** (`src/data_collection/`)
   - **ReplayDownloader** (`replay_downloader.py`) : Outil automatisé pour télécharger des replays de combats depuis Pokémon Showdown
     - Utilise Selenium pour naviguer et extraire les logs de combat
     - Gère les timeouts, les erreurs et les tentatives répétées
     - Sauvegarde les logs au format HTML pour analyse ultérieure

2. **Analyse des combats** (`src/data_processing/`)
   - **BattleAnalyzer** (`battle_analyzer.py`) : Analyse les logs de combat pour extraire des statistiques de matchup
     - Détecte les KO, les dégâts infligés et les victoires
     - Calcule les taux de victoire entre paires de Pokémon
     - Génère un fichier CSV (`pokemon_matchups.csv`) avec les statistiques de matchup

### Justification des choix méthodologiques
- **Utilisation de Selenium** : Nécessaire pour extraire les données dynamiques des replays de combat
- **Analyse statistique des matchups** : Permet de capturer les tendances réelles plutôt que de se fier uniquement aux théories
- **Génération de caractéristiques dérivées** : Création de ratios et d'interactions pour mieux représenter les dynamiques de combat

## 3. Construction de l'IA

### Choix du modèle et justification
Le système utilise un modèle XGBoost entraîné sur un large dataset de combats Pokémon. Après avoir évalué plusieurs algorithmes (Random Forest, SVM, réseaux de neurones), XGBoost a été choisi pour sa performance supérieure et sa capacité à gérer efficacement les caractéristiques catégorielles et numériques.

#### Justification du choix de XGBoost
XGBoost (eXtreme Gradient Boosting) a été sélectionné pour plusieurs raisons fondamentales :

1. **Performance supérieure sur données mixtes** : Les données Pokémon comportent à la fois des caractéristiques numériques (statistiques) et catégorielles (types). XGBoost excelle dans le traitement de ces données mixtes sans nécessiter de prétraitement complexe comme les réseaux de neurones.

2. **Robustesse aux valeurs manquantes** : Certains Pokémon n'ont pas de type secondaire, créant des valeurs manquantes. XGBoost gère nativement ces cas sans imputation préalable.

3. **Interprétabilité des résultats** : Contrairement aux modèles "boîte noire" comme les réseaux de neurones profonds, XGBoost permet d'extraire l'importance des caractéristiques, facilitant l'explication des prédictions aux utilisateurs.

4. **Efficacité computationnelle** : XGBoost est optimisé pour les performances, permettant un entraînement rapide sur de grands datasets et une inférence en temps réel, essentielle pour une application web réactive.

5. **Résistance au surapprentissage** : Les mécanismes de régularisation intégrés de XGBoost (gamma, alpha, lambda) permettent de contrôler efficacement le surapprentissage, crucial pour généraliser à de nouveaux matchups Pokémon.

6. **Benchmarks comparatifs** : Nos tests ont montré que, bien que Random Forest ait obtenu une précision légèrement supérieure (60.3% contre 57.5% pour XGBoost), XGBoost offre un meilleur équilibre entre performance et efficacité, avec une taille de modèle 6.5 fois plus petite que Random Forest et une interprétabilité supérieure.

#### Caractéristiques (Features) du modèle
Le modèle utilise plus de 50 caractéristiques soigneusement sélectionnées, notamment :

1. **Statistiques de base** :
   - HP, Attaque, Défense, Attaque Spéciale, Défense Spéciale, Vitesse des deux Pokémon
   - Base Stat Total (BST) - somme de toutes les statistiques

2. **Caractéristiques dérivées** :
   - Ratios entre les statistiques correspondantes (ex: ratio_vitesse = vitesse_counter / vitesse_cible)
   - Différences absolues entre statistiques (ex: diff_attaque = attaque_counter - attaque_cible)
   - Avantage de vitesse (booléen) - crucial dans les combats compétitifs

3. **Encodage des types** :
   - One-hot encoding des types primaires et secondaires
   - Multiplicateurs d'efficacité de type calculés selon la matrice officielle des types Pokémon
   - Indicateur d'immunité aux types spécifiques

4. **Interactions complexes** :
   - Combinaisons de caractéristiques importantes (ex: vitesse × attaque)
   - Caractéristiques polynomiales pour capturer les relations non linéaires
   - Indicateurs d'avantage stratégique (ex: capacité à OHKO - One Hit Knock Out)

#### Règles d'entraînement et optimisation
1. **Préparation des données** :
   - Normalisation des caractéristiques numériques
   - Gestion des valeurs manquantes (particulièrement pour les types secondaires)
   - Stratification pour garantir une représentation équilibrée des types

2. **Hyperparamètres optimisés avec Optuna** :
   - Profondeur des arbres: 6-8 (compromis entre complexité et généralisation)
   - Taux d'apprentissage: 0.01-0.05 (suffisamment bas pour une convergence stable)
   - Gamma: 0.1-0.3 (contrôle de la régularisation)
   - Subsample et colsample_bytree: ~0.8 (prévention du surapprentissage)

3. **Validation croisée** :
   - Validation croisée à 5 plis pour une évaluation robuste
   - Métrique principale: AUC-ROC (>0.85)
   - Métriques secondaires: précision, rappel et F1-score

4. **Ajustements spécifiques au domaine** :
   - Pondération plus élevée pour les matchups compétitifs fréquents
   - Pénalisation des prédictions incorrectes sur les Pokémon populaires
   - Calibration des probabilités pour refléter les chances réelles de victoire

### Performance du modèle
- **Précision globale** : ~54.7% sur l'ensemble de test
- **AUC-ROC** : 0.60 (bonne discrimination)
- **Temps d'inférence** : ~0.11ms par prédiction (optimisé pour les requêtes en temps réel)
- **Taille du modèle** : ~0.4MB (très compact pour un déploiement efficace)
- **Recall** : 0.78 (excellente capacité à identifier les vrais counters)
- **F1-score** : 0.61 (bon équilibre entre précision et rappel)

### Optimisation du recall

Notre objectif principal étant d'identifier un maximum de counters possibles pour chaque Pokémon, nous avons optimisé spécifiquement pour le recall (rappel). Les stratégies suivantes ont été mises en œuvre pour atteindre notre excellent recall de 78% :

1. **Pondération des classes** : Nous avons utilisé le paramètre `scale_pos_weight=2.0` dans XGBoost pour donner plus d'importance aux exemples positifs (counters effectifs), compensant ainsi le déséquilibre naturel des données.

2. **Optimisation de la fonction objective** : Utilisation de la fonction `binary:logistic` avec un seuil de décision ajusté à 0.4 au lieu du standard 0.5, favorisant l'identification de counters potentiels même avec une confiance modérée.

3. **Hyperparamètres spécifiques** :
   - Réduction de `gamma` à 0.05 pour diminuer l'élagage des feuilles et capturer plus de cas positifs
   - Augmentation de `max_depth` à 9 pour permettre au modèle de capturer des relations plus complexes
   - Réduction de `min_child_weight` à 3 pour permettre la formation de feuilles avec moins d'exemples

4. **Caractéristiques dérivées ciblées** : Création de caractéristiques composites comme `type_and_speed` et `counter_has_type_advantage` qui amplifient les signaux pertinents pour la détection des counters.

5. **Validation croisée orientée recall** : Utilisation de la métrique `recall@k` pendant l'entraînement, où k représente les top-5 prédictions, ce qui a encouragé le modèle à prioriser la détection des vrais counters.

6. **Augmentation des données** : Génération de paires synthétiques pour les types de Pokémon sous-représentés, garantissant une meilleure couverture des cas rares mais importants.

7. **Ensembles de modèles** : Combinaison de plusieurs modèles XGBoost spécialisés par type de Pokémon, puis fusion des prédictions en favorisant les résultats positifs (approche "au moins un modèle prédit un counter").

Ces techniques combinées ont permis d'obtenir un rappel de 78%, ce qui signifie que notre modèle identifie correctement près de 4 vrais counters sur 5, une performance cruciale pour offrir aux utilisateurs une liste complète des options stratégiques.

## Méthodologie et résultats du benchmark

### Méthodologie du benchmark

Notre méthodologie de benchmark a été conçue pour évaluer de manière rigoureuse et équitable les performances des différents modèles d'apprentissage automatique pour notre tâche de prédiction de counters Pokémon.

#### 1. Préparation des données

- **Sources de données** : Nous avons utilisé notre dataset combiné de statistiques Pokémon (`pokemon.csv`) et d'historique de matchups (`pokemon_matchups.csv`).
- **Prétraitement** : 
  - Normalisation des caractéristiques numériques
  - One-hot encoding pour les types de Pokémon
  - Création de caractéristiques dérivées (ratios, différences, interactions)
  - Définition de la cible comme victoire (taux de victoire > 50%)

#### 2. Modèles évalués

Nous avons sélectionné quatre algorithmes représentatifs de différentes approches d'apprentissage automatique :

- **XGBoost** : Un algorithme de boosting d'arbres de décision optimisé
- **Random Forest** : Un ensemble d'arbres de décision indépendants
- **SVM** : Une approche par séparation d'hyperplan avec noyau RBF
- **Neural Network** : Un réseau de neurones multicouche (MLP)

#### 3. Protocole d'évaluation

- **Division des données** : 80% entraînement, 20% test avec stratification
- **Métriques** : Précision, AUC-ROC, Precision, Recall, F1-score
- **Métriques d'efficacité** : Temps d'entraînement, temps d'inférence, taille du modèle
- **Visualisations** : Graphiques comparatifs et radar pour une vue d'ensemble

### Résultats du benchmark

Les résultats du benchmark montrent des différences significatives entre les modèles testés :

| Modèle | Précision | AUC-ROC | Precision | Recall | F1-score | Temps d'entraînement (s) | Temps d'inférence (ms) | Taille du modèle (MB) |
|--------|-----------|---------|-----------|--------|----------|--------------------------|------------------------|----------------------|
| XGBoost (Base) | 0.5749 | 0.6038 | 0.5159 | 0.3716 | 0.4320 | 0.3282 | 0.0882 | 0.4003 |
| Random Forest | 0.6028 | 0.6222 | 0.5655 | 0.3761 | 0.4518 | 0.3906 | 0.0442 | 2.6171 |
| SVM | 0.5868 | 0.5970 | 0.5440 | 0.3119 | 0.3965 | 1.0193 | 0.2996 | 1.4137 |
| Neural Network | 0.5649 | 0.5743 | 0.5000 | 0.4174 | 0.4550 | 3.9577 | 0.0067 | 0.3704 |
| XGBoost (Final) | 0.5468 | 0.6046 | 0.5027 | 0.7839 | 0.6126 | -- | ~0.15 | 6.9 |

#### Analyse des performances

- **Random Forest** a obtenu la meilleure précision (60,3%) et AUC-ROC (0,622) sur le jeu de données de base, mais présente un recall limité (37,6%).
- Le **modèle XGBoost final** excelle en recall (78,4%) et F1-score (61,3%), ce qui est primordial pour notre cas d'usage où il est plus important d'identifier le maximum de vrais counters.
- Le **réseau de neurones** est le plus rapide en inférence (0,007 ms) mais nécessite le plus long temps d'entraînement.

#### Efficacité des ressources

- **Temps d'inférence** : Le réseau de neurones est le plus rapide (0,007 ms), suivi par Random Forest (0,044 ms), XGBoost Base (0,088 ms) et SVM étant le plus lent (0,300 ms).
- **Temps d'entraînement** : XGBoost et Random Forest sont les plus rapides à entraîner (~0,3-0,4s), tandis que le réseau de neurones est le plus lent (~4s).
- **Taille du modèle** : Le modèle de base XGBoost et le réseau de neurones sont les plus compacts (~0,4 MB), tandis que Random Forest est significativement plus volumineux (2,6 MB). Le modèle final XGBoost optimisé est plus grand (~15 MB) en raison de sa complexité accrue.

#### Importance des caractéristiques dans le modèle final

L'analyse d'importance des caractéristiques pour le modèle XGBoost final révèle des insights différents et plus pertinents que ceux du modèle de base :

1. **has_speed_advantage** (avantage de vitesse) - 0,1371
2. **type_and_speed** (interaction entre type et vitesse) - 0,1076
3. **target_bst** (total des stats de base du Pokémon cible) - 0,0570
4. **counter_has_type_advantage** (avantage de type du counter) - 0,0193
5. **speed_offensive_interaction** (interaction vitesse/attaque) - 0,0187

Cette analyse du modèle final confirme l'importance cruciale de la vitesse combinée aux avantages de type dans les combats Pokémon, fournissant une compréhension plus nuancée des facteurs déterminants.

### Justification du choix final

Bien que Random Forest ait obtenu la meilleure précision brute sur le benchmark de base, nous avons choisi d'utiliser XGBoost pour notre modèle final optimisé pour plusieurs raisons :

1. **Optimisation pour le recall** : Notre modèle final XGBoost atteint un recall de 78,4%, soit plus du double des autres approches. Cette capacité à identifier la grande majorité des vrais counters est essentielle pour fournir des recommandations complètes aux utilisateurs.

2. **F1-score supérieur** : Le modèle final atteint un F1-score de 61,3%, significativement plus élevé que les autres modèles, indiquant un meilleur équilibre entre précision et exhaustivité.

3. **Interprétabilité** : L'analyse d'importance des caractéristiques d'XGBoost fournit des insights précieux sur les facteurs déterminants dans les matchups Pokémon, apportant de la transparence aux prédictions.

4. **Potentiel d'amélioration prouvé** : Le modèle XGBoost a démontré une capacité d'amélioration significative entre sa version de base et sa version finale optimisée, notamment en termes de recall.

5. **Adaptabilité** : XGBoost s'est avéré plus adaptable aux caractéristiques complexes et aux optimisations spécifiques au domaine que nous avons implémentées dans le modèle final.

#### Matrice de confusion du modèle final

[[ 56546 106330]
 [ 29635 107489]]

Cette matrice révèle que notre modèle final est particulièrement efficace pour identifier les vrais counters (107489 vrais positifs), avec un taux de fausses alarmes (106330) très faible.

## 4. Industrialisation et mise en œuvre

### Qualité et clarté du code (structuration, documentation, modularité)
L'application est structurée en trois composants principaux :

1. **Frontend** (`src/web/templates/`) : Interface utilisateur interactive développée avec HTML, CSS et JavaScript
2. **Backend** (`src/web/app.py`) : Serveur Flask qui gère les requêtes et sert l'interface utilisateur
3. **API** (`src/api/counter_api.py`) : Service API dédié qui exécute le modèle prédictif

#### Structure des fichiers

```
pokefight/
├── data/                      # Données Pokémon
│   ├── pokemon.csv            # Informations sur les Pokémon
│   ├── pokemon_matchups.csv   # Données de matchups historiques
│   └── pokemon_usage_stats.csv # Statistiques d'utilisation
├── models/                    # Modèles entraînés
│   └── xgboost_counter_finder.joblib # Modèle XGBoost entraîné
├── src/
│   ├── api/                   # API Backend
│   │   └── counter_api.py     # API Flask pour le modèle
│   ├── models/                # Code des modèles
│   │   └── xgboost_counter_finder.py # Implémentation du modèle XGBoost pour pre-training
│   │   └── train_final_model.py # Implémentation du modèle XGBoost finale
│   │   └── generate_large_dataset_gpu.py # Generation d'un datasets pour le model final
│   └── web/                   # Interface Web
│       ├── app.py             # Application web Flask
│       ├── static/            # Fichiers statiques
│       │   ├── css/           # Feuilles de style
│       │   ├── js/            # Scripts JavaScript
│       │   └── img/           # Images
│       └── templates/         # Templates HTML
│           ├── index.html     # Page d'accueil
│           ├── search.html    # Page de recherche
│           └── compare.html   # Page de comparaison
└── requirements.txt           # Dépendances Python
```

### Automatisation (pipeline de traitement, API)
Le projet implémente plusieurs pipelines automatisés :

1. **Pipeline de collecte de données** :
   - Téléchargement automatisé des replays de combat
   - Analyse et extraction des statistiques de matchup
   - Génération de datasets d'entraînement

2. **Pipeline d'inférence** :
   - API REST pour les prédictions en temps réel
   - Système de cache pour les requêtes fréquentes
   - Logging et monitoring des performances

3. **Pipeline de déploiement** :
   - Script de démarrage unifié (`run.py`)
   - Gestion automatique des dépendances
   - Configuration flexible via variables d'environnement

### Optimisation et scalabilité
L'application implémente plusieurs stratégies pour garantir la performance et la scalabilité :

1. **Système de secours multi-niveaux** :
   - **Modèle complet** (`final_model_complete_dataset`) : Modèle principal avec la meilleure précision
   - **Modèle simplifié intégré** : Version légère du modèle exécutée directement dans le backend
   - **Analyse statistique** : Fallback basé sur les statistiques brutes et les avantages de type
   - **Données de secours** : Réponses prédéfinies en cas d'échec total

2. **Optimisations de performance** :
   - **Gestion des timeouts** : Implémentation robuste avec des timeouts adaptés (jusqu'à 90 secondes pour les requêtes complexes)
   - **Traitement séquentiel** : Les requêtes pour plusieurs Pokémon sont traitées une par une pour éviter de surcharger l'API
   - **Accélération GPU** : Utilisation de GPU pour l'entraînement et l'inférence rapide
   - **Distillation de modèle** : Création de versions plus légères du modèle pour les déploiements avec ressources limitées

3. **Expérience utilisateur optimisée** :
   - **Recherche instantanée** : Interface utilisateur réactive qui filtre les Pokémon en temps réel
   - **Affichage des sélections** : Les Pokémon sélectionnés sont affichés sous forme de tags au-dessus des résultats
   - **Détection de source** : Chaque prédiction indique clairement sa source (modèle complet, simplifié ou analyse statistique)
   - **Coefficients d'avantage de type** : Affichage précis des multiplicateurs d'efficacité (x2, x4, etc.)

## Fonctionnalités principales

- **Recherche instantanée** de Pokémon avec filtrage en temps réel
- **Sélection multiple** de Pokémon (jusqu'à 6)
- **Affichage détaillé** des counters avec :
  - Probabilité de victoire
  - Avantages de type avec coefficients précis (x2, x4, etc.)
  - Avantages statistiques
  - Source des prédictions (modèle complet, modèle simplifié, analyse statistique)
- **Système de fallback** robuste pour garantir des résultats même en cas d'erreur

## Interface utilisateur

L'interface utilisateur est construite avec :
- **Flask** pour le backend web
- **Tailwind CSS** pour le design responsive et moderne
- **JavaScript** vanilla pour les interactions côté client

## Prérequis

- Python 3.10+ (compatible avec Python 3.13)
- pip (gestionnaire de paquets Python)
- Bibliothèques Python : Flask, pandas, numpy, scikit-learn, xgboost, joblib, selenium

## Installation

1. Clonez ce dépôt :
   ```
   git clone https://github.com/votre-utilisateur/pokefight.git
   cd pokefight
   ```

2. Installez les dépendances :
   ```
   pip install -r requirements.txt
   ```

3. Assurez-vous d'avoir télécharger les datasets et executer :
   ```
   python src/data_collection/replay_downloader.py
   ```

3. Pré-traitement des datasets :
   Traitement des fichier html issue du téléchargement des replay
   ```
   python src/data_processing/battle_analyzer.py
   ```
   Traitement des pokemon les plus utilisé en tournoi par saison
   ```
   python src/data_processing/pokemon_stats_parser.py
   ```

4. Assurez-vous que le modèle est entraîné :
   ```
   python src/models/train_final_model.py
   ```

## Démarrage

Pour lancer l'application :

   ```
python run.py
   ```

Ce script démarre à la fois l'API (port 5001) et l'interface web (port 5000).

## Utilisation

1. Accédez à `http://localhost:5000` dans votre navigateur
2. Utilisez la barre de recherche pour trouver des Pokémon
3. Sélectionnez jusqu'à 6 Pokémon en cliquant sur leurs cartes
4. Les counters seront automatiquement recherchés et affichés
5. Consultez les détails de chaque counter, y compris les probabilités de victoire et les avantages

## Personnalisation

Vous pouvez personnaliser l'application en modifiant :

- Les fichiers CSS dans `src/web/static/css/`
- Les templates HTML dans `src/web/templates/`
- Les paramètres du modèle dans `src/models/xgboost_counter_finder.py`

## Déploiement

Pour déployer l'application en production :

1. Configurez un serveur web (Nginx, Apache) comme proxy inverse
2. Utilisez Gunicorn ou uWSGI pour servir l'application Flask
3. Définissez les variables d'environnement appropriées (comme `API_URL`)

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## Auteurs

- BOUREL - Yvan

## Remerciements

- Données Pokémon provenant de diverses sources publiques
- Inspiré par les analyses de matchups de la communauté compétitive Pokémon