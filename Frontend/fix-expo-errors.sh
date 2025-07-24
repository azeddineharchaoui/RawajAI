#!/bin/bash
echo "========================================"
echo "    🔧 CORRECTION ERREURS EXPO"
echo "========================================"
echo

echo "🛑 Arrêt de tous les serveurs..."
pkill -f expo 2>/dev/null || true
pkill -f node 2>/dev/null || true
lsof -ti:8081 | xargs kill -9 2>/dev/null || true

echo "✅ Serveurs arrêtés"
echo

echo "🧹 Nettoyage cache Expo..."
rm -rf .expo
rm -rf node_modules/.cache
rm -rf .next
rm -f package-lock.json
rm -f yarn.lock

echo "✅ Cache nettoyé"
echo

echo "📦 Réinstallation dépendances..."
npm install --force

echo "✅ Dépendances installées"
echo

echo "🚀 Redémarrage Expo sur port 8082..."
echo "⏳ Patientez..."
sleep 3

npx expo start --web --port 8082 --clear &

echo
echo "✅ Expo redémarré sur http://localhost:8082"
echo "🌐 Ouvrez votre navigateur en mode incognito"
echo "📱 Ou utilisez un autre navigateur"
