import os
import time
import random
import requests
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime

class ReplayDownloader:
    """
    Classe pour télécharger automatiquement les replays de combat depuis Pokémon Showdown.
    """
    
    def __init__(self, max_replays=5000, output_dir="data/battle_history"):
        """
        Initialise le téléchargeur de replays.
        
        Args:
            max_replays: Nombre maximum de replays à télécharger
            output_dir: Répertoire de sortie pour les fichiers HTML
        """
        self.base_url = "https://replay.pokemonshowdown.com/"
        self.max_replays = max_replays
        self.output_dir = output_dir
        self.downloaded_urls = set()  # IDs complets des replays déjà téléchargés
        self.processed_urls = set()   # URLs déjà traitées dans cette session
        self.consent_accepted = False
        self.consecutive_empty_attempts = 0  # Compteur de tentatives sans nouveaux replays
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Charger les URLs déjà téléchargées et compter les fichiers existants
        self.downloaded_count = self._load_downloaded_urls()
        
        # Configurer le navigateur
        self._setup_browser()
    
    def _setup_browser(self):
        """Configure le navigateur Chrome pour le téléchargement."""
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")  # Activer le mode headless
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def _load_downloaded_urls(self):
        """
        Charge la liste des URLs déjà téléchargées.
        
        Returns:
            int: Nombre de fichiers HTML dans le répertoire de sortie
        """
        file_count = 0
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".html"):
                file_count += 1
                # Extraire l'ID complet du replay à partir du nom de fichier
                parts = filename.split("-")
                if len(parts) >= 1:
                    # Stocker l'ID complet (format-battleid)
                    self.downloaded_urls.add(parts[0])
        
        print(f"Chargé {len(self.downloaded_urls)} IDs uniques de replays déjà téléchargés")
        print(f"Nombre total de fichiers dans le répertoire: {file_count}")
        return file_count
    
    def _accept_cookie_consent(self):
        """Accepte la fenêtre de consentement des cookies si elle est présente."""
        if self.consent_accepted:
            return True
            
        try:
            # Vérifier si la fenêtre de consentement est présente
            consent_button = self.driver.find_element(By.CSS_SELECTOR, ".fc-button.fc-cta-consent")
            if consent_button:
                print("Fenêtre de consentement détectée, acceptation...")
                consent_button.click()
                time.sleep(2)
                self.consent_accepted = True
                return True
        except NoSuchElementException:
            try:
                # Méthode alternative
                consent_button = self.driver.find_element(By.CSS_SELECTOR, "button[aria-label='Consent']")
                if consent_button:
                    print("Fenêtre de consentement détectée (méthode 2), acceptation...")
                    consent_button.click()
                    time.sleep(2)
                    self.consent_accepted = True
                    return True
            except NoSuchElementException:
                # JavaScript en dernier recours
                try:
                    self.driver.execute_script("""
                        var buttons = document.querySelectorAll('button');
                        for (var i = 0; i < buttons.length; i++) {
                            if (buttons[i].textContent.includes('Consent')) {
                                buttons[i].click();
                                return true;
                            }
                        }
                        return false;
                    """)
                    print("Fenêtre de consentement acceptée via JavaScript")
                    time.sleep(2)
                    self.consent_accepted = True
                    return True
                except Exception:
                    pass
                
            # Si nous arrivons ici, pas de fenêtre de consentement
            self.consent_accepted = True
            return False
        except Exception as e:
            print(f"Erreur lors de la gestion du consentement: {e}")
            return False
    
    def download_replays(self):
        """Télécharge les replays de combat depuis la page principale."""
        try:
            attempts = 0
            
            # Continuer tant qu'on n'a pas atteint le nombre max de replays et qu'on n'a pas eu 10 tentatives vides consécutives
            while self.downloaded_count < self.max_replays and self.consecutive_empty_attempts < 10:
                attempts += 1
                print(f"\nTentative {attempts} - Replays téléchargés: {self.downloaded_count}/{self.max_replays}")
                print(f"Tentatives consécutives sans nouveaux replays: {self.consecutive_empty_attempts}/10")
                
                # Charger la page principale
                print(f"Chargement de la page principale: {self.base_url}")
                self.driver.get(self.base_url)
                
                # Attendre le chargement et gérer le consentement
                time.sleep(5)
                self._accept_cookie_consent()
                
                # Faire défiler pour charger tout le contenu
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(2)
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                
                # Extraire les liens de replay
                replay_links = self._extract_replay_links()
                if not replay_links:
                    print("Aucun replay trouvé, tentative suivante...")
                    self.consecutive_empty_attempts += 1
                    continue
                
                # Afficher les détails des replays trouvés
                self._debug_replay_links(replay_links)
                
                # Filtrer les liens déjà traités
                new_links = [link for link in replay_links if link not in self.processed_urls]
                if not new_links:
                    print("Tous les replays ont déjà été traités, rafraîchissement...")
                    self.consecutive_empty_attempts += 1
                    continue
                
                print(f"Nouveaux replays à traiter: {len(new_links)}")
                
                # Télécharger les nouveaux replays
                downloads_in_this_attempt = 0
                for replay_link in new_links:
                    # Vérifier si on a atteint la limite
                    if self.downloaded_count >= self.max_replays:
                        break
                    
                    # Marquer comme traité pour éviter les doublons
                    self.processed_urls.add(replay_link)
                    
                    # Vérifier si déjà téléchargé (par ID complet)
                    if replay_link in self.downloaded_urls:
                        print(f"Replay déjà téléchargé: {replay_link}")
                        continue
                    
                    # Télécharger le replay
                    success = self._download_replay(replay_link)
                    if success:
                        downloads_in_this_attempt += 1
                        # Ajouter à la liste des téléchargés
                        self.downloaded_urls.add(replay_link)
                    
                    # Pause entre les téléchargements
                    time.sleep(random.uniform(1.0, 2.0))
                
                print(f"{downloads_in_this_attempt} nouveaux replays téléchargés dans cette tentative")
                
                # Si des téléchargements ont été effectués, réinitialiser le compteur de tentatives vides
                if downloads_in_this_attempt > 0:
                    self.consecutive_empty_attempts = 0
                else:
                    self.consecutive_empty_attempts += 1
                    print(f"Aucun nouveau replay téléchargé, tentatives consécutives sans succès: {self.consecutive_empty_attempts}/10")
            
            # Afficher la raison de l'arrêt
            if self.downloaded_count >= self.max_replays:
                print(f"\nTéléchargement terminé: nombre maximum de replays atteint ({self.max_replays}).")
            elif self.consecutive_empty_attempts >= 10:
                print(f"\nTéléchargement terminé: 10 tentatives consécutives sans nouveaux replays.")
            
            print(f"Total: {self.downloaded_count} replays téléchargés en {attempts} tentatives.")
        
        except Exception as e:
            print(f"Erreur lors du téléchargement des replays: {e}")
        finally:
            self.driver.quit()
    
    def _extract_replay_links(self):
        """Extrait les liens de replay de la page actuelle."""
        replay_links = set()
        
        # Cibler la section "Recent replays"
        try:
            recent_section = self.driver.find_element(By.XPATH, "//section[.//h1[text()='Recent replays']]")
            if recent_section:
                print("Section 'Recent replays' trouvée!")
                
                # Extraire tous les liens de cette section
                replay_links_elements = recent_section.find_elements(By.CSS_SELECTOR, "a.blocklink")
                for link in replay_links_elements:
                    href = link.get_attribute("href")
                    if href and self.base_url in href:
                        replay_id = href.replace(self.base_url, "")
                        replay_links.add(replay_id)
                
                print(f"Trouvé {len(replay_links)} replays dans la section 'Recent replays'")
        except NoSuchElementException:
            print("Section 'Recent replays' non trouvée, essai avec JavaScript...")
            
            # Méthode JavaScript alternative
            try:
                js_links = self.driver.execute_script("""
                    var recentSection = Array.from(document.querySelectorAll('section')).find(
                        section => section.textContent.includes('Recent replays')
                    );
                    
                    var links = [];
                    
                    if (recentSection) {
                        var blockLinks = recentSection.querySelectorAll('a.blocklink');
                        for (var i = 0; i < blockLinks.length; i++) {
                            var href = blockLinks[i].getAttribute('href');
                            if (href && href.includes('-')) {
                                links.push(href);
                            }
                        }
                    }
                    
                    return links;
                """)
                
                for link in js_links:
                    if link.startswith('/'):
                        link = link[1:]
                    replay_links.add(link)
                
                print(f"Trouvé {len(replay_links)} replays via JavaScript")
            except Exception as js_err:
                print(f"Erreur lors de l'extraction JavaScript: {js_err}")
        
        return replay_links
    
    def _debug_replay_links(self, replay_links):
        """Affiche des informations détaillées sur les liens de replay trouvés."""
        print("\n=== Détails des replays trouvés ===")
        for i, link in enumerate(sorted(replay_links)[:10]):
            parts = link.split("-")
            format_id = parts[0] if parts else "inconnu"
            battle_id = parts[1] if len(parts) > 1 else "inconnu"
            print(f"{i+1}. Format: {format_id}, ID: {battle_id}, Lien complet: {link}")
        
        if len(replay_links) > 10:
            print(f"... et {len(replay_links) - 10} autres replays")
        print("=====================================\n")
    
    def _download_replay(self, replay_link):
        """
        Télécharge un replay spécifique.
        
        Args:
            replay_link: Lien relatif vers le replay
        
        Returns:
            bool: True si le téléchargement a réussi, False sinon
        """
        try:
            # Construire l'URL complète
            replay_url = self.base_url + replay_link
            log_url = replay_url + ".log"
            
            print(f"Téléchargement du replay: {replay_url}")
            
            # Ouvrir la page du replay
            self.driver.get(replay_url)
            
            # Gérer la fenêtre de consentement
            self._accept_cookie_consent()
            
            # Attendre que la page soit chargée
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "h1"))
            )
            
            # Extraire le log
            log_content = None
            try:
                log_script = self.driver.find_element(By.CSS_SELECTOR, "script.log")
                log_content = log_script.get_attribute("innerHTML")
            except NoSuchElementException:
                # Fallback: télécharger le log directement
                response = requests.get(log_url)
                response.raise_for_status()
                log_content = response.text
            
            if not log_content:
                raise Exception("Impossible de récupérer le contenu du log")
            
            # Extraire les informations pour le nom de fichier
            title_element = self.driver.find_element(By.TAG_NAME, "h1")
            format_text = ""
            player1 = ""
            player2 = ""
            
            if title_element:
                title_text = title_element.text.strip()
                
                # Extraire le format
                format_match = re.search(r'\[(.*?)\]', title_text)
                if format_match:
                    format_text = format_match.group(1)
                
                # Extraire les joueurs
                players_match = re.search(r'(.*?) vs\. (.*?)$', title_text)
                if players_match:
                    player1 = players_match.group(1).strip()
                    player2 = players_match.group(2).strip()
            
            # Nettoyer les noms pour le nom de fichier
            format_id = replay_link.split("-")[0]
            today = datetime.now().strftime("%Y-%m-%d")
            
            format_text = "".join(c if c.isalnum() else "" for c in format_text)
            player1 = "".join(c if c.isalnum() else "" for c in player1)
            player2 = "".join(c if c.isalnum() else "" for c in player2)
            
            filename = f"{format_id}-{today}-{format_text}{player1}-{player2}.html"
            file_path = os.path.join(self.output_dir, filename)
            
            # Créer un fichier HTML avec le log du combat
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Replay: {player1} vs. {player2}</title>
    <style>
        body {{ font-family: monospace; white-space: pre-wrap; }}
    </style>
</head>
<body>
{log_content}
</body>
</html>
"""
            
            # Enregistrer le contenu HTML
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            # Incrémenter le compteur
            self.downloaded_count += 1
            
            print(f"Replay téléchargé avec succès: {filename}")
            
            return True
            
        except Exception as e:
            print(f"Erreur lors du téléchargement du replay {replay_link}: {e}")
            return False

def main():
    downloader = ReplayDownloader(max_replays=5000)
    downloader.download_replays()

if __name__ == "__main__":
    main() 