@echo off
echo ========================================
echo  Alien Classifier - Launcher Windows
echo ========================================
echo.

REM Check if venv exists
if not exist "ecole\" (
    echo Creation de l'environnement virtuel...
    python -m venv ecole
    echo.
)

REM Activate venv
call ecole\Scripts\activate.bat

REM Install requirements if needed
if not exist "ecole\Lib\site-packages\torch\" (
    echo Installation des dependances...
    pip install -r requirements.txt
    echo.
)

REM Run the app
python run_app.py

pause
