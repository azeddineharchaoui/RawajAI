@echo off
cls
echo ==========================================
echo     FORCE RESTART RAWAJAI SERVER
echo ==========================================
echo.

echo  Arrêt FORCÉ de tous les processus sur port 5000...
echo.

REM Tuer tous les processus Python
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM python3.exe >nul 2>&1

REM Tuer spécifiquement le port 5000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5000') do (
    echo 🔪 Killing process %%a
    taskkill /F /PID %%a >nul 2>&1
)

echo ✅ Port 5000 complètement libéré
echo.

echo  Attente quelque secondes...
timeout /t 5 /nobreak >nul

echo 📦 Installation/Mise à jour FastAPI...
pip install --upgrade fastapi uvicorn --quiet
echo ✅ FastAPI installé
echo.

echo 🚀 Démarrage du nouveau serveur FastAPI...
echo.
python fastapi_server.py

pause
