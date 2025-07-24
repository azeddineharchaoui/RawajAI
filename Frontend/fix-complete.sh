#!/bin/bash
echo "====================================="

echo "📁 Nettoyage des caches..."
rm -rf node_modules
rm -rf .expo
rm -f package-lock.json
rm -f yarn.lock

echo " Installation des dépendances avec le transformer..."
npm install metro-react-native-babel-transformer@^0.77.0
npm install

echo " Nettoyage cache Expo..."
npx expo install --fix

echo " Démarrage du serveur..."
npx expo start --web --port 8082 --clear

echo " Correction terminée !"
echo " Ouvrez: http://localhost:8082"
