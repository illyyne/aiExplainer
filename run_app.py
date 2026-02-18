#!/usr/bin/env python3
"""
Quick launcher for the Alien Classifier system
Runs Streamlit on the correct port with auto-browser opening
"""

import subprocess
import sys
import webbrowser
from pathlib import Path
from threading import Timer

def check_requirements():
    """Check if requirements are installed"""
    try:
        import torch
        import streamlit
        import cv2
        print("Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"Dépendance manquante: {e}")
        print("\nInstallez avec: pip install -r requirements.txt")
        return False

def check_model():
    """Check if model is trained"""
    model_path = Path('models/alien_classifier_best.pth')
    if model_path.exists():
        print("Modèle entrainé trouvé")
        return True
    else:
        print("✗ Modèle non trouvé")
        return False

def check_dataset():
    """Check if dataset exists"""
    images_dir = Path('images/aliens')
    csv_path = Path('images/class/classification.csv')
    
    if not images_dir.exists():
        print("Répertoire images introuvable: images/aliens/")
        return False
    
    if not csv_path.exists():
        print("Fichier CSV des labels introuvable: images/class/classification.csv")
        return False
    
    num_images = len(list(images_dir.glob('*.*')))
    print(f"Dataset trouvé ({num_images} images)")
    return True

def open_browser(port=8502):
    """Open browser after a short delay"""
    def _open():
        url = f'http://localhost:{port}'
        print(f"\nOuverture du navigateur: {url}")
        webbrowser.open(url)
    
    # Wait 3 seconds for Streamlit to start
    Timer(3.0, _open).start()

def train_model():
    """Train the model"""
    print("\n" + "="*60)
    print("🎓 Entraînement du modèle...")
    print("="*60 + "\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'train_model.py'],
            check=True
        )
        print("\n✓ Modèle entraîné avec succès!")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Erreur lors de l'entraînement")
        return False
    except FileNotFoundError:
        print("\n✗ train_model.py introuvable")
        return False

def run_streamlit(port=8502):
    """Run Streamlit app on specified port"""
    print("\n" + "="*60)
    print("Lancement de l'interface Streamlit...")
    print("="*60)
    print(f"\nURL: http://localhost:{port}")
    print("Le navigateur va s'ouvrir automatiquement dans 3 secondes...")
    print("\nPour arrêter: Ctrl+C dans ce terminal")
    print("="*60 + "\n")
    
    # Open browser automatically
    open_browser(port)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'streamlit_app.py',
            '--server.port', str(port),
            '--server.headless', 'true',
            '--browser.serverAddress', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\n\nArrêt de l'application")
    except FileNotFoundError:
        print("\nstreamlit_app.py introuvable")
        print("Assurez-vous d'être dans le bon répertoire")
        sys.exit(1)

def main():
    print("\n" + "="*60)
    print("🛸 Alien Classifier - Ericsson Research")
    print("="*60)
    print("\n📋 Vérification du système...")
    print()
    
    # Check dataset
    if not check_dataset():
        print("\nDataset manquant.")
        sys.exit(1)
    
    # Check dependencies
    if not check_requirements():
        print("\nDépendances manquantes.")
        print("Exécutez: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check and train model if needed
    if not check_model():
        print("\n" + "="*60)
        print("Le modèle n'est pas encore entraîné")
        print("="*60)
        print("\n🎯 Options:")
        print("  1. Entraîner maintenant (recommandé)")
        print("  2. Quitter et entraîner manuellement")
        print()
        
        try:
            choice = input("Votre choix (1/2): ").strip()
            
            if choice == '1':
                if not train_model():
                    print("\nImpossible de continuer sans modèle")
                    sys.exit(1)
            else:
                print("\nPour entraîner manuellement:")
                print("python train_model.py")
                print("\n👋 Au revoir!")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\n\n👋 Annulé")
            sys.exit(0)
    
    print("\nTous les composants sont prêts!")
    
    # Run Streamlit with auto-open browser
    run_streamlit()

if __name__ == '__main__':
    main()
