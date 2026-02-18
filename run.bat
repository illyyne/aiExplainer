@echo off
title Alien Classifier - Ericsson Research

echo.
echo ========================================
echo  Alien Classifier - Ericsson Research
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python non installe
    echo Telechargez sur python.org
    pause
    exit /b 1
)

REM Create venv if needed
if not exist "ecole\Scripts\activate.bat" (
    echo [INFO] Creation environnement virtuel 'ecole'...
    python -m venv ecole
    echo [OK] Environnement virtuel cree
    echo.
)

REM Activate venv
echo [INFO] Activation de l'environnement virtuel...
call ecole\Scripts\activate.bat

REM Install dependencies if needed
if not exist "ecole\installed.flag" (
    echo [INFO] Installation des dependances...
    pip install -r requirements.txt -q
    echo. > ecole\installed.flag
    echo [OK] Dependances installees
    echo.
)

REM Run the launcher (which will auto-open browser and handle training)
python run_app.py

pause
