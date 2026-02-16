#!/bin/bash

echo "========================================"
echo " Alien Classifier - Launcher Linux/Mac"
echo "========================================"
echo ""

# Check if venv exists
if [ ! -d "ecole" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv ecole
    echo ""
fi

# Activate venv
source ecole/bin/activate

# Install requirements if needed
if [ ! -d "ecole/lib/python*/site-packages/torch" ]; then
    echo "Installation des dépendances..."
    pip install -r requirements.txt
    echo ""
fi

# Run the app
python run_app.py
