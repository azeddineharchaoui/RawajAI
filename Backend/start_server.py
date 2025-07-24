#!/usr/bin/env python3
"""
Script de démarrage du serveur RawaJAI Backend
"""
import subprocess
import sys
import os

def install_requirements():
    """Installe les dépendances si nécessaire"""
    try:
        print(" Installation des dépendances...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(" Dépendances installées avec succès!")
    except subprocess.CalledProcessError as e:
        print(f" Erreur lors de l'installation des dépendances: {e}")
        return False
    return True

def start_server():
    """Démarre le serveur FastAPI"""
    try:
        print(" Démarrage du serveur RawaJAI Backend...")
        print(" API sera disponible sur: http://localhost:5000")
        print(" Documentation: http://localhost:5000/docs")
        print(" Mode reload activé pour le développement")
        print("\n" + "="*50)
        
        # Démarrer le serveur
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", "5000", 
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n Serveur arrêté par l'utilisateur")
    except Exception as e:
        print(f" Erreur lors du démarrage du serveur: {e}")

if __name__ == "__main__":
    print("🔧 RawaJAI Backend Setup")
    print("="*30)
    
    # Vérifier si on est dans le bon répertoire
    if not os.path.exists("main.py"):
        print(" Fichier main.py non trouvé. Assurez-vous d'être dans le répertoire Backend/")
        sys.exit(1)
    
    # Installer les dépendances
    if install_requirements():
        # Démarrer le serveur
        start_server()
    else:
        print(" Impossible de démarrer le serveur à cause des erreurs de dépendances")
        sys.exit(1)
