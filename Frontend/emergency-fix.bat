@echo off
cls
echo ==========================================
echo    🚨 CORRECTION URGENTE EXPO
echo ==========================================
echo.

echo 🛑 ARRÊT TOTAL...
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM expo.exe >nul 2>&1
taskkill /F /IM chrome.exe >nul 2>&1

echo 🧹 NETTOYAGE COMPLET...
rmdir /s /q .expo >nul 2>&1
rmdir /s /q node_modules >nul 2>&1
rmdir /s /q .next >nul 2>&1
del package-lock.json >nul 2>&1

echo 📦 RÉINSTALLATION PROPRE...
npm cache clean --force
npm install

echo 🔧 CORRECTION CONFIGURATION...
copy expo-config-fixed.json app.json

echo 🚀 REDÉMARRAGE SUR PORT 8082...
echo.
echo ⚠️  IMPORTANT:
echo 1. Fermez TOUS les onglets du navigateur
echo 2. Ouvrez en MODE INCOGNITO
echo 3. Allez sur http://localhost:8082
echo 4. Ou essayez Firefox/Edge
echo.

start "Expo Clean" cmd /c "npx expo start --web --port 8082 --clear --reset-cache"

echo ✅ Serveur démarré en mode propre
echo 🌐 Ouvrez http://localhost:8082 en INCOGNITO
pause
