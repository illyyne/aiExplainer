#!/usr/bin/env python3
"""
Quick launcher for the Alien Classifier system
Runs Streamlit on the correct port
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import torch
        import streamlit
        import cv2
        print("All dependencies installed")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall with: pip install -r requirements.txt")
        return False

def check_model():
    """Check if model is trained"""
    model_path = Path('models/alien_classifier_best.pth')
    if model_path.exists():
        print("Model found")
        return True
    else:
        print("Model not found")
        print("\nTrain model first with: python train_model.py")
        return False

def check_dataset():
    """Check if dataset exists"""
    images_dir = Path('images/aliens')
    csv_path = Path('images/class/classification.csv')
    
    if not images_dir.exists():
        print("Images directory not found: images/aliens/")
        return False
    
    if not csv_path.exists():
        print("CSV file not found: images/class/classification.csv")
        return False
    
    num_images = len(list(images_dir.glob('*.*')))
    print(f"✓ Dataset found ({num_images} images)")
    return True

def run_streamlit():
    """Run Streamlit app on port 8502"""
    print("\n" + "="*60)
    print("Lancement de l'interface Streamlit...")
    print("="*60)
    print("\nURL: http://localhost:8502")
    print("\nPour arrêter: Ctrl+C")
    print("="*60 + "\n")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'streamlit_app.py',
            '--server.port', '8502',
            '--server.headless', 'true'
        ])
    except KeyboardInterrupt:
        print("\n\n Arrêt de l'application")

def main():
    print("\n🛸 Alien Classifier - Launcher")
    print("="*60)
    
    # Check everything
    if not check_dataset():
        sys.exit(1)
    
    if not check_requirements():
        sys.exit(1)
    
    if not check_model():
        print("\n❓ Voulez-vous entraîner le modèle maintenant? (o/n): ", end='')
        response = input().lower()
        if response == 'o':
            print("\n🎓 Entraînement du modèle...")
            subprocess.run([sys.executable, 'train_model.py'])
        else:
            print("\n✋ Veuillez entraîner le modèle avec: python train_model.py")
            sys.exit(1)
    
    # Run Streamlit
    run_streamlit()

if __name__ == '__main__':
    main()
