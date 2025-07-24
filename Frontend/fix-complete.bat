@echo off
echo 🔧 CORRECTION COMPLETE DES ERREURS EXPO
echo =====================================

echo  Nettoyage des caches...
rmdir /s /q node_modules 2>nul
rmdir /s /q .expo 2>nul
del package-lock.json 2>nul
del yarn.lock 2>nul

echo  Installation des dépendances avec le transformer...
npm install metro-react-native-babel-transformer@^0.77.0
npm install

echo  Nettoyage cache Expo...
npx expo install --fix

echo  Démarrage du serveur...
npx expo start --web --port 8082 --clear

echo  Correction terminée !
echo  Ouvrez: http://localhost:8082
pause
