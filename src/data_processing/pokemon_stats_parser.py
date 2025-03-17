import os
import re
import csv
import glob

def extract_tables(file_content):
    """Extrait les tableaux de statistiques des Pokémon du contenu du fichier."""
    # Plusieurs patterns pour capturer différents formats de tableaux
    patterns = [
        # Format standard avec + ---- + et lignes complètes
        r'\+ ---- \+ ------------------ \+ -+? \+ ------- \+(?:\s*\+ ------- \+)?\n((?:\| \d+ +\| [^|]+ \| \d+ +\| +\d+\.\d+% +\|(?:\s*\d+\.\d+% +\|)?\n)+)',
        
        # Format avec Code: suivi d'un tableau
        r'Code:\s*\n((?:\| \d+ +\| [^|]+ \| \d+ +\| +\d+\.\d+% +\|(?:\s*\d+\.\d+% +\|)?\n)+)',
        
        # Format avec + ---- + mais sans les lignes complètes
        r'\+ ---- \+ ------------------ \+ ---- \+ ------- \+ ------- \+\n((?:\| \d+ +\| [^|]+ \| +\d+ +\| +\d+\.\d+% +\| +\d+\.\d+% +\|\n)+)'
    ]
    
    all_tables = []
    for pattern in patterns:
        tables = re.findall(pattern, file_content)
        all_tables.extend(tables)
    
    return all_tables

def parse_table(table_text, season):
    """Parse un tableau et retourne les données sous forme de liste de dictionnaires."""
    rows = []
    
    # Analyse chaque ligne du tableau
    for line in table_text.strip().split('\n'):
        # Ignorer les lignes qui ne contiennent pas de données
        if not line.strip() or '|' not in line or line.startswith('+'):
            continue
            
        # Extraction des colonnes
        columns = [col.strip() for col in line.split('|')]
        if len(columns) < 5:  # Ignorer les lignes mal formatées
            continue
        
        # Nettoyage des colonnes vides au début et à la fin
        columns = [col for col in columns if col]
        
        if len(columns) < 4:  # Pas assez de colonnes après nettoyage
            continue
            
        try:
            rank = columns[0].strip()
            pokemon = columns[1].strip()
            usage = columns[2].strip()
            
            # Vérifier si la colonne usage est un nombre
            try:
                int(usage)
            except ValueError:
                # Si ce n'est pas un nombre, essayons de réorganiser
                if len(columns) >= 5:
                    rank = columns[0].strip()
                    pokemon = columns[1].strip()
                    usage = columns[2].strip()
                    usage_percent = columns[3].strip()
                    win_percent = columns[4].strip() if len(columns) > 4 else None
                else:
                    continue
            else:
                # Si c'est un nombre, continuons normalement
                usage_percent = columns[3].strip()
                win_percent = columns[4].strip() if len(columns) > 4 else None
            
            # Vérifier que les pourcentages sont bien formatés
            if not usage_percent.endswith('%'):
                continue
                
            if win_percent and not win_percent.endswith('%'):
                win_percent = None
            
            # Création d'un dictionnaire pour cette ligne
            row = {
                  'Rank': rank,
                  'Pokemon': pokemon,
                  'Usage': usage,
                  'Usage_Percent': usage_percent,
                  'Win_Percent': win_percent if win_percent else ''  # Toujours inclure la clé, même vide
              }
                  
            rows.append(row)
            
        except (IndexError, ValueError) as e:
            # Ignorer les lignes mal formatées
            continue
    
    # Ajouter la saison à chaque ligne
    for row in rows:
        row['Season'] = season
    
    return rows

def extract_season_number(file_content, filename):
    """Extrait le numéro de saison du contenu du fichier ou du nom de fichier."""
    # Essayer d'extraire du contenu du fichier
    season_patterns = [
        r'Smogon Tour (\d+)',
        r'Tour (\d+)',
        r'ST(\d+)',
        r'S(\d+)'
    ]
    
    for pattern in season_patterns:
        match = re.search(pattern, file_content)
        if match:
            return match.group(1)
    
    # Si pas trouvé dans le contenu, essayer d'extraire du nom de fichier
    match = re.search(r'S(\d+)', os.path.basename(filename))
    if match:
        return match.group(1)
    
    # Si toujours pas trouvé, utiliser le nom du fichier sans extension
    return os.path.splitext(os.path.basename(filename))[0]

def extract_tables_alternative(file_content):
    """Méthode alternative pour extraire les tableaux."""
    # Rechercher des lignes qui ressemblent à des entrées de tableau
    lines = file_content.split('\n')
    table_lines = []
    in_table = False
    current_table = []
    
    for line in lines:
        # Détecter le début d'un tableau
        if '|' in line and ('Rank' in line or 'Pokemon' in line):
            in_table = True
            current_table = [line]
        # Continuer à collecter les lignes du tableau
        elif in_table and '|' in line:
            current_table.append(line)
        # Détecter la fin d'un tableau
        elif in_table and not '|' in line:
            in_table = False
            if len(current_table) > 2:  # Au moins une ligne d'en-tête et une ligne de données
                table_lines.append('\n'.join(current_table))
            current_table = []
    
    # Ajouter le dernier tableau s'il existe
    if in_table and len(current_table) > 2:
        table_lines.append('\n'.join(current_table))
    
    # Si aucun tableau trouvé, essayer une approche plus simple
    if not table_lines:
        # Recherche des lignes qui ressemblent à des entrées de tableau
        pattern = r'(\| *\d+ *\| *[^|]+ *\| *\d+ *\| *\d+\.\d+% *\|(?:[^|]*\|)?)'
        matches = re.findall(pattern, file_content)
        if matches:
            table_lines.append('\n'.join(matches))
    
    return table_lines

def process_file(file_path):
    """Traite un fichier et extrait les données des tableaux."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        # Extraire le numéro de saison
        season = extract_season_number(content, file_path)
        print(f"Traitement du fichier {os.path.basename(file_path)} (Saison {season})...")
        
        # Extraire les tableaux
        tables = extract_tables(content)
        
        if not tables:
            print(f"Aucun tableau trouvé dans {os.path.basename(file_path)}")
            
            # Essayer une méthode alternative
            tables = extract_tables_alternative(content)
            if tables:
                print(f"  - {len(tables)} tableaux trouvés avec la méthode alternative")
        
        all_rows = []
        for i, table in enumerate(tables):
            rows = parse_table(table, season)
            all_rows.extend(rows)
            print(f"  - Tableau {i+1}: {len(rows)} entrées extraites")
        
        return all_rows
    
    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")
        return []

def process_files(data_dir):
    """Traite tous les fichiers dans le répertoire de données."""
    # Rechercher les fichiers de saison
    file_pattern = os.path.join(data_dir, "season_usage", "S*.txt")
    files = glob.glob(file_pattern)
    
    # Si aucun fichier trouvé, essayer sans le sous-dossier
    if not files:
        file_pattern = os.path.join(data_dir, "S*.txt")
        files = glob.glob(file_pattern)
    
    all_data = []
    
    for file_path in sorted(files):
        # Ignorer S17 car pas de winrate
        if "S17" in file_path:
            print(f"Ignoré {os.path.basename(file_path)} (pas de données de taux de victoire)")
            continue
            
        rows = process_file(file_path)
        all_data.extend(rows)
    
    return all_data

def save_to_csv(data, output_file):
    """Sauvegarde les données dans un fichier CSV."""
    if not data:
        print("Aucune donnée à sauvegarder.")
        return
    
    # S'assurer que tous les dictionnaires ont les mêmes clés
    all_keys = set()
    for row in data:
        all_keys.update(row.keys())
    
    # Ajouter les clés manquantes avec des valeurs vides
    for row in data:
        for key in all_keys:
            if key not in row:
                row[key] = ''
    
    # Détermination des en-têtes
    fieldnames = sorted(all_keys)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Données sauvegardées dans {output_file}")

def main():
    # Répertoire contenant les fichiers de données
    data_dir = "data"
    
    # Fichier de sortie CSV
    output_file = "data/pokemon_usage_stats.csv"
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Traitement des fichiers
    data = process_files(data_dir)
    
    # Sauvegarde des données en CSV
    save_to_csv(data, output_file)
    
    # Afficher un résumé des saisons extraites
    if data:
        seasons = set(row['Season'] for row in data)
        print(f"Traitement terminé. {len(data)} entrées extraites pour les saisons: {', '.join(sorted(seasons))}")
    else:
        print("Aucune donnée extraite.")

if __name__ == "__main__":
    main() 