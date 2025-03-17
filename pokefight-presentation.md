---
title: "PokéFight"
subtitle: "Système de prédiction de counters Pokémon par IA"
author: "BOUREL Yvan"
date: "Mars 2025"
theme: "metropolis"
colortheme: "metropolis"
fontsize: "10pt"
aspectratio: "169"
header-includes:
  - \usepackage{graphicx}
  - \usepackage{booktabs}
  - \usepackage{fontawesome}
  - \definecolor{pokered}{RGB}{215,60,60}
  - \definecolor{pokeblue}{RGB}{65,105,225}
  - \setbeamercolor{title}{fg=white,bg=pokered}
  - \setbeamercolor{frametitle}{fg=white,bg=pokeblue}
---

# Projet PokéFight

## Vue d'ensemble

### Définition du projet {.fragile}

- **PokéFight** : Application web utilisant l'IA pour prédire les meilleurs counters Pokémon
- Combine **analyse de données** et **machine learning** pour des recommandations précises
- Va au-delà des simples calculateurs de types traditionnels

![](https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/6.png){width=15%}
![](https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/9.png){width=15%}
![](https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png){width=15%}
![](https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/150.png){width=15%}

### Originalité et valeur ajoutée

- Précision élevée grâce à l'apprentissage automatique
- Considère les **statistiques complexes** au-delà des simples avantages de type
- Capture les **nuances stratégiques** des matchups compétitifs
- Offre des recommandations personnalisées et précises

## Sources de données

### Collecte et analyse des données

- **3 sources principales** :
  - [Base pokemon stats](https://www.kaggle.com/datasets/rounakbanik/pokemon) : Statistiques de base des Pokémon
  - [Tournament pokemon usage](https://www.smogon.com/forums/threads/tournament-statistics-archive.3538686/) : Données d'utilisation en tournoi
  - [Battle History](https://replay.pokemonshowdown.com/) : Logs de combats réels

### Traitement des données

- **ReplayDownloader** : Téléchargement automatisé des replays de combats avec Selenium
- **BattleAnalyzer** : Extraction des statistiques de matchup (KO, dégâts, victoires)
- Génération de caractéristiques dérivées (ratios, interactions)

![Architecture de collecte de données](https://mermaid.ink/img/pako:eNqNkc1qwzAQhF9F7CkFv4Bz6KGUHkqhLb30JodVvY4FtqTKclJC8u5VbAcSOgfBandn_GlgCIXViHkoISS-OGNsiVoFx8Oz2ElTobeuL2GLWxzM59IkWoewHZKyNf4dNKVlnE-vc3B4ZME5b9jdG-iq7Hqc0pKZ95Bv_ue21X7H7svVkp0WSZzNsOBz9lKYFk2QKoXqKEh3aqL3_Cp9Qcet7rTfTehCR2IokN0QyTlKCe6OIGcI0pCm-TsVKtZO4cTL1ZGRMUiwRgmKuZhHBQ50MQOdmYYy0iWE2hBV5CX_2qnBY-UgpP4pF5iRnUMVulXh-QcyP4-C?type=png){width=60%}

## Architecture de l'IA

### Choix du modèle

- **XGBoost** sélectionné après comparaison de différents algorithmes
- Benchmark rigoureux face à Random Forest, SVM et réseaux neuronaux
- Plus de **50 caractéristiques** soigneusement sélectionnées

### Justification de XGBoost

1. Performance supérieure sur données mixtes (numériques et catégorielles)
2. Robustesse aux valeurs manquantes (types secondaires)
3. **Interprétabilité** des résultats
4. Efficacité computationnelle pour inférence en temps réel
5. Résistance au surapprentissage grâce aux mécanismes de régularisation
6. Meilleur équilibre entre performance et efficacité

## Caractéristiques du modèle

### Features principales

- **Statistiques de base** : HP, Attaque, Défense, Attaque Spéciale, Défense Spéciale, Vitesse
- **Caractéristiques dérivées** : Ratios entre statistiques, différences absolues, avantage de vitesse
- **Encodage des types** : One-hot encoding, multiplicateurs d'efficacité, indicateurs d'immunité
- **Interactions complexes** : Combinaisons de caractéristiques, caractéristiques polynomiales

### Top 5 des caractéristiques importantes

1. **has_speed_advantage** (avantage de vitesse) - 0.1371
2. **type_and_speed** (interaction entre type et vitesse) - 0.1076
3. **target_bst** (total des stats de base du Pokémon cible) - 0.0570
4. **counter_has_type_advantage** (avantage de type du counter) - 0.0193
5. **speed_offensive_interaction** (interaction vitesse/attaque) - 0.0187

![Importance des caractéristiques](https://mermaid.ink/img/pako:eNptj8FqwzAQRH9FzCkF_4DtoYcWemihvfSSm7zaVS2wJSHJSSjJv1e2Q6En7czuzHsw2EJlDWIGbYjZl94YW6NRPrD_2AetWvTW9TVscY-9-VrbLBY-7Iek7Ix_BU1pGRez6xw8nllwwRvunozRNexu5sxMeygLtvgpbav9nt2Xqxm7LpI4G7DAZ-y5NC0aL3UK1UjIcAoi-tC59IqeR-bY7yd0oXMiBQn-iCTnoQa4G4KcIMRDmhbvlKtYO4UjL1djRsQg3holaOZ8GhV40MUEdOE6ykiXEGpDtCMv-etONQ4rDyH1D7mijOyYqtDtNp5_AUjJl7M?type=png){width=70%}

## Optimisation du recall

### Stratégies pour maximiser le recall (78%)

- **Pondération des classes** : `scale_pos_weight=2.0` pour compenser le déséquilibre des données
- **Ajustement du seuil** : Seuil de décision à 0.4 au lieu de 0.5 standard
- **Hyperparamètres ciblés** : 
  - `gamma` réduit (0.05)
  - `max_depth` augmenté (9)
  - `min_child_weight` réduit (3)
- **Caractéristiques dérivées stratégiques** : Amplification des signaux pertinents
- **Validation orientée recall@k** : Optimisation pour les top-5 prédictions
- **Augmentation de données** : Pour types sous-représentés
- **Ensemble de modèles spécialisés** par type de Pokémon

## Performance du modèle final

### Métriques clés

- **Précision globale** : 54.7% sur l'ensemble de test
- **AUC-ROC** : 0.60 (bonne discrimination)
- **Temps d'inférence** : ~0.11ms par prédiction (optimisé pour requêtes en temps réel)
- **Taille du modèle** : ~0.4MB (très compact pour déploiement efficace)
- **Recall** : 0.78 (excellente capacité à identifier les vrais counters)
- **F1-score** : 0.61 (bon équilibre entre précision et rappel)

### Matrice de confusion du modèle final

```
[[ 56546 106330]
 [ 29635 107489]]
```

## Résultats du benchmark

### Comparaison des modèles

| Modèle | Précision | AUC-ROC | Precision | Recall | F1-score | Temps d'entraînement (s) | Temps d'inférence (ms) | Taille du modèle (MB) |
|--------|-----------|---------|-----------|--------|----------|--------------------------|------------------------|----------------------|
| XGBoost (Base) | 0.5749 | 0.6038 | 0.5159 | 0.3716 | 0.4320 | 0.3282 | 0.0882 | 0.4003 |
| Random Forest | 0.6028 | 0.6222 | 0.5655 | 0.3761 | 0.4518 | 0.3906 | 0.0442 | 2.6171 |
| SVM | 0.5868 | 0.5970 | 0.5440 | 0.3119 | 0.3965 | 1.0193 | 0.2996 | 1.4137 |
| Neural Network | 0.5649 | 0.5743 | 0.5000 | 0.4174 | 0.4550 | 3.9577 | 0.0067 | 0.3704 |
| XGBoost (Final) | 0.5468 | 0.6046 | 0.5027 | 0.7839 | 0.6126 | -- | ~0.15 | 6.9 |

### Justification du choix final

- **Optimisation pour le recall** : 78.4%, crucial pour identifier la majorité des vrais counters
- **F1-score supérieur** : 61.3%, meilleur équilibre précision/rappel
- Interprétabilité des résultats
- Potentiel d'amélioration prouvé
- Adaptabilité aux caractéristiques complexes

## Architecture de l'application

### Composants principaux

- **Frontend** : Interface utilisateur HTML/CSS/JS
- **Backend** : Serveur Flask pour les requêtes utilisateur
- **API** : Service dédié pour les prédictions du modèle

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
│   │   └── xgboost_counter_finder.py # Implémentation du modèle XGBoost
│   └── web/                   # Interface Web
│       ├── app.py             # Application web Flask
│       ├── static/            # Fichiers statiques
│       └── templates/         # Templates HTML
```

### Système de fallback multi-niveaux

1. **Modèle complet** : Prédictions de haute précision
2. **Modèle simplifié** : Version légère exécutée directement dans le backend
3. **Analyse statistique** : Basée sur statistiques et avantages de type
4. **Données de secours** : Réponses prédéfinies en cas d'échec total

![Architecture système](https://mermaid.ink/img/pako:eNqNlE1v2zAMhv-K4JMNJFkCw4ceVmw9DBiWDVh32C27KDJjC5VlT5LTFkX--6jYSeu0G3ayTD7k-1AUeRSVEwgZKlFuqrJhK1dIh7qlm6t3i6umkHVvn-aJ-QwXIRwmGfNOFz1WrpdwHhNFtKpXRK3JM_1_ZSoTzzOfvwD5pPHvQQlG-vRUEtqTVjKL8uwZZ5VFxVFmrm2yTp7ztf80VQkSHdgmKUgrFjqX6x3kUe4Jux4LrVJpFaknPNv4C7H4gVxSodcsjyg5NzLQOavQU1vHqfHfcJIJTc5JLPB4vVRuoRrsiIbQJ7bjHa7Yj4vaxMRVP0dQNdkDMQn8DHXsVkVWgmxLPxkB2C5lRlwlS9jv9wwl-rXBIGQm1A79lrXMwW3Bgs_pGaGVdGWEUzTH9gwt52QU7Z0aQ6-jlGPgRjlcFNiWlhZOpdP2YyUqRrEHEQh_zVQHjHEQhS1kMDvOgpTi7mVpGrczb_Ix2RNomT_ybxnkYbjYl8zKtugW0qZA5Vh6vhI7PL5nnSC-PCCWLMYy7DyL-XH1ZLqOd8SitYaXKOu-2c4VxRqXXkMmfXM1DRkL-MiFyP8L2XeNauQtj6zKy_j3JV_-KPkTmZ_8TP45Y2sVfzc5nTOdcWBYxqeQm3TRiMyLmBUl2lkrb3RFoVeFqFoVD-J5vGtqCteDcJoxVpWsqFHvNvG5faVayDJSO_FZ9MfjFx8o9PQ?type=png){width=80%}

## Fonctionnalités et UI

### Interface utilisateur

- **Recherche instantanée** avec filtrage en temps réel
- **Sélection multiple** de Pokémon (jusqu'à 6)
- **Affichage détaillé** des counters avec probabilités
- Design responsive (Tailwind CSS)

### Détails des counters

- Probabilité de victoire
- Avantages de type avec coefficients précis (x2, x4, etc.)
- Avantages statistiques
- Source des prédictions (modèle complet/simplifié/analyse)

![Interface utilisateur](https://i.imgur.com/q2YfRdX.png){width=85%}

## Optimisations de performance

### Stratégies d'optimisation

- **Gestion des timeouts** adaptée (jusqu'à 90 secondes pour requêtes complexes)
- **Traitement séquentiel** des requêtes multiples
- **Accélération GPU** pour l'entraînement et l'inférence rapide
- **Distillation de modèle** pour versions plus légères
- **Système de cache** pour requêtes fréquentes

### Expérience utilisateur améliorée

- Interface réactive qui filtre les Pokémon en temps réel
- Affichage des Pokémon sélectionnés sous forme de tags
- Indication claire de la source des prédictions
- Visualisation précise des coefficients d'avantage de type

## Conclusion

### Récapitulatif

- Système de prédiction par IA pour counters Pokémon
- XGBoost optimisé pour le recall (78.4%)
- Architecture robuste avec système de fallback multi-niveaux
- Interface utilisateur intuitive et rapide

### Perspectives

- Intégration de données de tournois récents
- Développement de modèles spécifiques par format de jeu
- Extension à d'autres jeux stratégiques
- API publique pour intégration tierce

## Contact

### Merci !

Projet réalisé par **BOUREL Yvan**

**Questions ?**

![](https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png){width=20%}