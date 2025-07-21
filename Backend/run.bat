@echo off
cls
echo ================================
echo    RAWAJAI BACKEND LAUNCHER
echo ================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou pas dans le PATH
    pause
    exit /b 1
)

echo ✅ Python détecté
echo.

REM Installer FastAPI et Uvicorn si nécessaire
echo 📦 Installation des dépendances...
pip install fastapi uvicorn --quiet

echo ✅ Dépendances installées
echo.

REM Démarrer le serveur
echo 🚀 Démarrage du serveur...
echo.
python simple_server.py

pause
