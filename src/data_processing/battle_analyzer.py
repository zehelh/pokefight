import os
import re
import csv
import pandas as pd
from collections import defaultdict

class BattleAnalyzer:
    """
    Classe pour analyser les logs de combat Pokémon et extraire des statistiques.
    """
    
    def __init__(self, battle_dir="data/battle_history", output_file="data/pokemon_matchups.csv"):
        """
        Initialise l'analyseur de combats.
        
        Args:
            battle_dir: Répertoire contenant les fichiers HTML des combats
            output_file: Fichier de sortie pour les statistiques
        """
        self.battle_dir = battle_dir
        self.output_file = output_file
        self.matchups = defaultdict(lambda: {"wins": 0, "losses": 0, "total": 0})
        self.pokemon_names = self._load_pokemon_names()
        self.damage_threshold = 40.0  # Seuil de dégâts pour compter une victoire (40%)
    
    def _load_pokemon_names(self):
        """Charge la liste des noms de Pokémon depuis le fichier CSV."""
        pokemon_names = set()
        try:
            # Essayer de charger depuis le fichier pokemon.csv
            df = pd.read_csv("data/pokemon.csv")
            for name in df["name"]:
                pokemon_names.add(name.lower())
            print(f"Chargé {len(pokemon_names)} noms de Pokémon depuis pokemon.csv")
        except FileNotFoundError:
            print("Fichier pokemon.csv non trouvé")
        
        return pokemon_names
    
    def analyze_battles(self):
        """Analyse tous les fichiers de combat dans le répertoire spécifié."""
        battle_files = [f for f in os.listdir(self.battle_dir) if f.endswith(".html")]
        print(f"Analyse de {len(battle_files)} logs de combat HTML...")
        
        for filename in battle_files:
            self._analyze_battle_file(os.path.join(self.battle_dir, filename))
        
        # Sauvegarder les résultats
        self._save_results()
        
        # Afficher quelques statistiques
        total_matchups = sum(data["total"] for data in self.matchups.values())
        print(f"Résultats sauvegardés dans {self.output_file}")
        print(f"Nombre total de matchups valides analysés: {total_matchups}")
    
    def _analyze_battle_file(self, file_path):
        """Analyse un fichier de combat spécifique."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Extraire les équipes
            teams = self._extract_teams(content)
            if not teams:
                return
            
            # Analyser le combat
            self._analyze_battle(content, teams, os.path.basename(file_path))
            
        except Exception as e:
            print(f"Erreur lors de l'analyse de {os.path.basename(file_path)}: {e}")
    
    def _extract_teams(self, content):
        """Extrait les équipes et les noms des Pokémon du contenu HTML."""
        # Extraire les noms des Pokémon de chaque équipe
        team1_raw = []
        team2_raw = []
        
        for line in content.split('\n'):
            if '|poke|p1|' in line:
                pokemon = line.split('|')[3].strip()
                team1_raw.append(pokemon)
            elif '|poke|p2|' in line:
                pokemon = line.split('|')[3].strip()
                team2_raw.append(pokemon)
        
        if not team1_raw or not team2_raw:
            return None
        
        print(f"Équipe 1 brute: {team1_raw}")
        print(f"Équipe 2 brute: {team2_raw}")
        
        # Convertir les noms bruts en noms réels
        team1 = {}
        team2 = {}
        
        for pokemon in team1_raw:
            display_name = pokemon.split(',')[0].strip()
            real_name = self._clean_pokemon_name(display_name)
            if real_name:
                print(f"Ajout à l'équipe 1: {display_name} -> {real_name}")
                team1[display_name] = real_name
        
        for pokemon in team2_raw:
            display_name = pokemon.split(',')[0].strip()
            real_name = self._clean_pokemon_name(display_name)
            if real_name:
                print(f"Ajout à l'équipe 2: {display_name} -> {real_name}")
                team2[display_name] = real_name
        
        print(f"Équipe 1: {team1}")
        print(f"Équipe 2: {team2}")
        
        return {
            "team1": team1,
            "team2": team2
        }
    
    def _clean_pokemon_name(self, name):
        """Nettoie le nom d'un Pokémon et vérifie s'il existe dans la liste."""
        # Vérifier si le nom complet existe
        if name.lower() in self.pokemon_names:
            return name
        
        # Si le nom contient un tiret, essayer avec le nom de base
        if '-' in name:
            base_name = name.split('-')[0].strip()
            if base_name.lower() in self.pokemon_names:
                return base_name
        
        print(f"Nom de Pokémon non reconnu: {name}")
        return None
    
    def _get_real_pokemon_name(self, display_name, team_dict):
        """
        Trouve le vrai nom d'un Pokémon dans l'équipe.
        Utilise une correspondance souple basée sur la contenance.
        """
        # Vérification exacte d'abord
        if display_name in team_dict:
            return team_dict[display_name]
        
        # Sinon, vérifier si un nom d'affichage contient le nom affiché
        for team_display_name, real_name in team_dict.items():
            if display_name in team_display_name or team_display_name in display_name:
                return real_name
        
        return None
    
    def _analyze_battle(self, content, teams, filename):
        """Analyse le combat pour déterminer les victoires et les défaites."""
        lines = content.split('\n')
        
        # Variables pour suivre l'état du combat
        current_turn = 0
        active_pokemon = {"p1a": None, "p2a": None}
        pokemon_hp = {}  # Format: {pokemon: [current_hp, max_hp]}
        damage_trackers = {}  # Format: {(attacker, defender): [total_damage, last_turn]}
        ko_trackers = set()  # Ensemble des Pokémon qui ont été KO
        full_healed = set()  # Ensemble des Pokémon qui ont été complètement soignés
        
        # Variables pour garder en mémoire l'état entre les lignes
        last_attacker = None
        last_move = None
        switched_out = set()  # Ensemble des Pokémon qui ont été switchés
        
        # Liste des effets de statut à ignorer pour les dégâts
        status_effects = ["psn", "brn", "Salt Cure", "Leftovers", "hurt by", "restored", "heal"]
        # Liste des effets qui indiquent un KO indirect
        ko_indirect_effects = ["psn", "brn", "confusion", "Salt Cure", "sandstorm", "hail", "recoil"]
        # Liste des mouvements de soin
        healing_moves = ["Recover", "Synthesis", "Roost", "Moonlight", "Morning Sun", "Slack Off", "Soft-Boiled"]
        
        # Maximal turns to keep damage record
        max_damage_turns = 5
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Détecter le tour actuel
            turn_match = re.search(r'\|turn\|(\d+)', line)
            if turn_match:
                current_turn = int(turn_match.group(1))
                
                # Réinitialiser certains trackers à chaque nouveau tour
                switched_out = set()
                
                # Nettoyer les vieux trackers de dégâts (plus de max_damage_turns tours)
                keys_to_remove = []
                for key, (damage, turn) in damage_trackers.items():
                    if current_turn - turn > max_damage_turns:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del damage_trackers[key]
                
                continue
            
            # Détecter les changements de Pokémon - permet de suivre qui est actif
            switch_match = re.search(r'\|(switch|drag|replace)\|(p\da): (.*?)\|', line)
            if switch_match:
                pos = switch_match.group(2)
                pokemon = switch_match.group(3).split(',')[0].strip()
                active_pokemon[pos] = pokemon
                
                # Réinitialiser les trackers de dégâts pour ce Pokémon
                keys_to_remove = []
                for key in damage_trackers:
                    if pokemon in key[1]:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del damage_trackers[key]
                
                continue
            
            # Détecter les retraits de Pokémon (switch out)
            withdraw_match = re.search(r'withdrew (.*?)!', line)
            if withdraw_match:
                withdrawn_pokemon = withdraw_match.group(1)
                switched_out.add(withdrawn_pokemon)  # Marquer comme retiré
                # Trouver quelle position avait ce Pokémon
                for pos, poke in active_pokemon.items():
                    if poke == withdrawn_pokemon:
                        active_pokemon[pos] = None  # Marquer comme retiré
                continue
            
            # Détection de "sent out" (après un withdraw)
            sent_out_match = re.search(r'sent out (.*?)!', line)
            if sent_out_match:
                pokemon = sent_out_match.group(1)
                # Trouver la position libre
                for pos in active_pokemon:
                    if active_pokemon[pos] is None:
                        active_pokemon[pos] = pokemon
                        break
                continue
            
            # Détecter les attaques de soin
            heal_move_match = re.search(r'used (.*?)!', line)
            if heal_move_match:
                move = heal_move_match.group(1)
                if move in healing_moves:
                    # Trouver le Pokémon qui a utilisé l'attaque
                    for j in range(max(0, i-3), i):
                        prev_line = lines[j].strip()
                        pokemon_match = re.search(r'(.*?) used', prev_line)
                        if pokemon_match:
                            pokemon = pokemon_match.group(1)
                            if "The opposing " in pokemon:
                                pokemon = pokemon[13:]
                            
                            # Marquer comme complètement soigné
                            full_healed.add(pokemon)
                            
                            # Réinitialiser les trackers de dégâts pour ce Pokémon
                            keys_to_remove = []
                            for key in damage_trackers:
                                if pokemon in key[1]:
                                    keys_to_remove.append(key)
                            
                            for key in keys_to_remove:
                                del damage_trackers[key]
                            
                            break
                    continue
            
            # Détecter les soins significatifs (>40%)
            heal_match = re.search(r'\|-heal\|(p\da): (.*?)\|', line)
            if heal_match:
                pokemon = heal_match.group(2).split(',')[0].strip()
                
                # Si ce n'est pas un soin de Leftovers ou autre effet mineur
                if not "from" in line or not any(effect in line for effect in ["Leftovers", "Black Sludge", "ability"]):
                    # Réinitialiser les trackers de dégâts pour ce Pokémon
                    keys_to_remove = []
                    for key in damage_trackers:
                        if pokemon in key[1]:
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        del damage_trackers[key]
                
                continue
            
            # Détecter les attaques
            move_match = re.search(r'\|move\|(p\da): (.*?)\|(.*?)\|', line)
            if move_match:
                attacker_pos = move_match.group(1)
                attacker = move_match.group(2).split(',')[0].strip()
                move = move_match.group(3)
                
                # Mettre à jour le dernier attaquant
                last_attacker = attacker
                last_move = move
                
                continue
            
            # Détecter quand une attaque est utilisée (format texte)
            used_match = re.search(r'(.*?) used (.+?)!', line)
            if used_match:
                attacker_text = used_match.group(1)
                move = used_match.group(2)
                
                # Nettoyer le nom de l'attaquant
                if "The opposing " in attacker_text:
                    attacker = attacker_text[13:]
                else:
                    attacker = attacker_text
                
                # Mettre à jour le dernier attaquant
                last_attacker = attacker
                last_move = move
                
                continue
            
            # Détecter les dégâts avec HP
            # Format: |-damage|p1a: Bisharp|265/334
            damage_match = re.search(r'\|-damage\|(p\da): (.*?)\|(\d+)/(\d+)(?:\s|$)', line)
            if damage_match:
                pos = damage_match.group(1)
                defender = damage_match.group(2).split(',')[0].strip()
                current_hp = int(damage_match.group(3))
                max_hp = int(damage_match.group(4))
                
                # Vérifier si ce sont des dégâts indirects
                is_indirect = False
                if "from" in line:
                    is_indirect = True
                else:
                    for j in range(max(0, i-3), i):
                        prev_line = lines[j].strip()
                        if any(effect in prev_line for effect in status_effects):
                            is_indirect = True
                            break
                
                if not is_indirect and last_attacker and last_attacker != defender:
                    # Récupérer les HP précédents
                    prev_hp, max_hp_prev = pokemon_hp.get(defender, [max_hp, max_hp])
                    
                    # Calculer les dégâts en pourcentage
                    if prev_hp > current_hp:  # S'assurer qu'il y a eu des dégâts
                        damage_percent = ((prev_hp - current_hp) / max_hp) * 100
                        
                        # Ajouter les dégâts au traceur
                        key = (last_attacker, defender)
                        if key not in damage_trackers:
                            damage_trackers[key] = [0, current_turn]
                        
                        damage_trackers[key][0] += damage_percent
                        damage_trackers[key][1] = current_turn  # Mettre à jour le dernier tour
                        
                        print(f"Tour {current_turn}: {last_attacker} a infligé {damage_percent:.1f}% à {defender}, total: {damage_trackers[key][0]:.1f}%")
                
                # Mettre à jour les HP actuels
                pokemon_hp[defender] = [current_hp, max_hp]
                
                continue
            
            # Détecter les KO complets (0 HP)
            # Exemple: |-damage|p1a: Bisharp|0 fnt
            ko_damage_match = re.search(r'\|-damage\|(p\da): (.*?)\|0 fnt', line)
            if ko_damage_match:
                pos = ko_damage_match.group(1)
                defender = ko_damage_match.group(2).split(',')[0].strip()
                
                # Vérifier si ce sont des dégâts indirects
                if "from" in line and any(effect in line for effect in ko_indirect_effects):
                    print(f"Tour {current_turn}: {defender} KO par effet indirect (damage 0 fnt)")
                    continue
                
                # Récupérer les HP précédents
                prev_hp, max_hp = pokemon_hp.get(defender, [0, 100])
                
                # Calculer les dégâts en pourcentage (tout le reste des HP)
                if prev_hp > 0 and last_attacker and last_attacker != defender:
                    damage_percent = (prev_hp / max_hp) * 100
                    
                    # Ajouter les dégâts au traceur
                    key = (last_attacker, defender)
                    if key not in damage_trackers:
                        damage_trackers[key] = [0, current_turn]
                    
                    damage_trackers[key][0] += damage_percent
                    damage_trackers[key][1] = current_turn  # Mettre à jour le dernier tour
                    
                    print(f"Tour {current_turn}: {last_attacker} a infligé {damage_percent:.1f}% à {defender} (KO), total: {damage_trackers[key][0]:.1f}%")
                
                continue
            
            # Détecter les dégâts pourcentage direct dans le texte
            # Format: (The opposing Ninetales lost 62.3% of its health!)
            percent_damage_match = re.search(r'\((.*?) lost (\d+\.\d+)% of its health!\)', line)
            if percent_damage_match:
                defender_text = percent_damage_match.group(1)
                damage_percent = float(percent_damage_match.group(2))
                
                # Nettoyer le nom du défenseur
                if "The opposing " in defender_text:
                    defender = defender_text[13:]
                else:
                    defender = defender_text
                
                # Vérifier si ce sont des dégâts indirects
                is_indirect = False
                for j in range(max(0, i-3), i):
                    prev_line = lines[j].strip()
                    if any(effect in prev_line for effect in status_effects):
                        is_indirect = True
                        break
                
                if not is_indirect and last_attacker and last_attacker != defender:
                    # Ajouter les dégâts au traceur
                    key = (last_attacker, defender)
                    if key not in damage_trackers:
                        damage_trackers[key] = [0, current_turn]
                    
                    damage_trackers[key][0] += damage_percent
                    damage_trackers[key][1] = current_turn  # Mettre à jour le dernier tour
                    
                    print(f"Tour {current_turn}: {last_attacker} a infligé {damage_percent:.1f}% à {defender}, total: {damage_trackers[key][0]:.1f}%")
                
                continue
            
            # Détecter les KO
            faint_match = re.search(r'\|faint\|(p\da): (.*)', line)
            if faint_match:
                pos = faint_match.group(1)
                defender = faint_match.group(2).split(',')[0].strip()
                
                # Éviter de traiter le même KO plusieurs fois
                if defender in ko_trackers:
                    continue
                
                ko_trackers.add(defender)
                
                # Vérifier si le Pokémon a été switch out (ce n'est pas un vrai KO)
                if defender in switched_out:
                    print(f"Tour {current_turn}: {defender} a été switch out, pas un vrai KO")
                    continue
                
                # Vérifier si c'est un KO dû à des effets indirects
                is_indirect_ko = False
                for j in range(max(0, i-5), i):
                    prev_line = lines[j].strip()
                    # Chercher les effets de dégâts indirects spécifiques
                    if any(effect in prev_line and defender in prev_line for effect in ko_indirect_effects):
                        is_indirect_ko = True
                        print(f"Tour {current_turn}: {defender} KO par effet indirect détecté dans: {prev_line}")
                        break
                
                if is_indirect_ko:
                    print(f"Tour {current_turn}: {defender} KO par effet indirect, pas comptabilisé")
                    continue
                
                # Trouver l'attaquant qui a infligé le plus de dégâts récemment
                best_attacker = None
                highest_damage = 0
                most_recent_turn = 0
                
                for (atk, def_), (damage, turn) in damage_trackers.items():
                    if defender in def_ or def_ in defender:  # Correspondance souple
                        # Privilégier les dégâts récents
                        if damage > highest_damage or (damage == highest_damage and turn > most_recent_turn):
                            highest_damage = damage
                            best_attacker = atk
                            most_recent_turn = turn
                
                print(f"Tour {current_turn}: {defender} mis KO, meilleur attaquant: {best_attacker} avec {highest_damage:.1f}% de dégâts")
                
                # Vérifier si les dégâts dépassent le seuil
                if highest_damage >= self.damage_threshold and best_attacker:
                    # Déterminer les équipes
                    attacker_team = "team1" if pos.startswith("p2") else "team2"
                    defender_team = "team2" if pos.startswith("p2") else "team1"
                    
                    # Trouver les vrais noms des Pokémon
                    attacker_real = self._get_real_pokemon_name(best_attacker, teams[attacker_team])
                    defender_real = self._get_real_pokemon_name(defender, teams[defender_team])
                    
                    # Si les deux noms sont reconnus, enregistrer la victoire
                    if attacker_real and defender_real:
                        # Enregistrer la victoire
                        matchup = (attacker_real, defender_real)
                        self.matchups[matchup]["total"] += 1
                        self.matchups[matchup]["wins"] += 1
                        
                        # Enregistrer également la défaite dans l'autre sens
                        reverse_matchup = (defender_real, attacker_real)
                        self.matchups[reverse_matchup]["total"] += 1
                        self.matchups[reverse_matchup]["losses"] += 1
                        
                        print(f"[{filename}] Victoire de {attacker_real} contre {defender_real} avec {highest_damage:.1f}% de dégâts")
                
                # Réinitialiser les dégâts pour ce défenseur
                keys_to_remove = []
                for key in damage_trackers:
                    if defender in key[1] or key[1] in defender:  # Correspondance souple
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del damage_trackers[key]
                
                # Réinitialiser les HP
                if defender in pokemon_hp:
                    del pokemon_hp[defender]
                
                continue
    
    def _save_results(self):
        """Sauvegarde les résultats dans un fichier CSV."""
        with open(self.output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Pokemon1", "Pokemon2", "Wins", "Losses", "Total", "Win Rate"])
            
            for matchup, data in sorted(self.matchups.items(), key=lambda x: x[1]["total"], reverse=True):
                pokemon1, pokemon2 = matchup
                win_rate = data["wins"] / data["total"] if data["total"] > 0 else 0
                writer.writerow([
                    pokemon1,
                    pokemon2,
                    data["wins"],
                    data["losses"],
                    data["total"],
                    f"{win_rate:.2%}"
                ])

def main():
    analyzer = BattleAnalyzer()
    analyzer.analyze_battles()

if __name__ == "__main__":
    main() 