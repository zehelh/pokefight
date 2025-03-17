import os
import sys
import subprocess
import threading
import time

def start_api():
    """Démarre l'API Flask sur le port 5001"""
    # Utiliser le répertoire de travail du projet
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Utiliser subprocess au lieu de os.system pour un meilleur contrôle
    api_process = subprocess.Popen([sys.executable, "-m", "src.api.counter_api"], 
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  universal_newlines=True)  # Utiliser universal_newlines pour gérer l'encodage
    
    # Lire et afficher la sortie de l'API en temps réel
    def monitor_output():
        try:
            for line in api_process.stdout:
                print(f"API: {line.strip()}")
        except Exception as e:
            print(f"Erreur lors de la surveillance de la sortie de l'API: {e}")
    
    # Démarrer un thread pour surveiller la sortie
    monitor_thread = threading.Thread(target=monitor_output)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    return api_process

def wait_for_api(max_attempts=15):
    """Attend que l'API soit prête"""
    print("Vérification que l'API est prête...")
    
    for attempt in range(max_attempts):
        try:
            import requests
            response = requests.get("http://localhost:5001/api/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ API prête après {attempt+1} tentatives")
                return True
        except Exception as e:
            print(f"Tentative {attempt+1}/{max_attempts}: {e}")
        
        print(f"Attente de l'API (tentative {attempt+1}/{max_attempts})...")
        time.sleep(2)  # Attendre plus longtemps entre les tentatives
    
    print("⚠️ L'API ne semble pas répondre, mais on continue quand même...")
    return False

def start_web():
    """Démarre l'interface web sur le port 5000"""
    # Utiliser le répertoire de travail du projet
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Importer et démarrer l'application web
    from src.web.app import app
    
    # Désactiver le rechargement automatique pour éviter le double lancement de l'API
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

if __name__ == "__main__":
    # Démarrer l'API
    print("Démarrage de l'API...")
    api_process = start_api()
    
    # Attendre que l'API soit prête
    wait_for_api()
    
    # Démarrer l'interface web
    print("Démarrage de l'interface web...")
    try:
        start_web()
    finally:
        # S'assurer que le processus API est terminé lorsque l'application web s'arrête
        print("Arrêt de l'API...")
        api_process.terminate()
        api_process.wait() 