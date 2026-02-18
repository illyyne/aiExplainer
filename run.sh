#!/bin/bash
echo ""
echo "========================================"
echo " 🛸 Alien Classifier - Ericsson Research"
echo "========================================"
echo ""

# Check if venv exists
if [ ! -d "ecole" ]; then
    echo "[INFO] Création de l'environnement virtuel 'ecole'..."
    python3 -m venv ecole
    echo "[OK] Environnement virtuel créé"
    echo ""
fi

# Activate venv
echo "[INFO] Activation de l'environnement virtuel..."
source ecole/bin/activate

# Check if requirements are installed
if [ ! -f "ecole/installed.flag" ]; then
    echo "[INFO] Installation des dépendances..."
    pip install -r requirements.txt -q
    touch ecole/installed.flag
    echo "[OK] Dépendances installées"
    echo ""
fi

# Run the launcher (which will auto-open browser and handle training)
python run_app.py