@echo off
cls
echo ========================================
echo    🔧 CORRECTION ERREURS EXPO
echo ========================================
echo.

echo 🛑 Arrêt de tous les serveurs...
taskkill /F /IM node.exe >nul 2>&1
taskkill /F /IM expo.exe >nul 2>&1
netstat -ano | findstr :8081 >nul && (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8081') do (
        taskkill /F /PID %%a >nul 2>&1
    )
)

echo ✅ Serveurs arrêtés
echo.

echo 🧹 Nettoyage cache Expo...
rmdir /s /q .expo >nul 2>&1
rmdir /s /q node_modules\.cache >nul 2>&1
rmdir /s /q .next >nul 2>&1
del package-lock.json >nul 2>&1
del yarn.lock >nul 2>&1

echo ✅ Cache nettoyé
echo.

echo 📦 Réinstallation dépendances...
npm install --force

echo ✅ Dépendances installées
echo.

echo 🚀 Redémarrage Expo sur port 8082...
echo ⏳ Patientez...
timeout /t 3 /nobreak >nul

start "Expo Server" cmd /c "npx expo start --web --port 8082 --clear"

echo.
echo ✅ Expo redémarré sur http://localhost:8082
echo 🌐 Ouvrez votre navigateur en mode incognito
echo 📱 Ou utilisez un autre navigateur
echo.
pause
