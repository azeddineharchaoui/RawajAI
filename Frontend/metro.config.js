const { getDefaultConfig } = require("expo/metro-config")

const config = getDefaultConfig(__dirname)

// Configuration pour corriger les erreurs MIME et serveur
config.server = {
  ...config.server,
  port: 8082,
  enhanceMiddleware: (middleware, metroServer) => {
    return (req, res, next) => {
      // Corriger les types MIME pour les bundles JavaScript
      if (req.url && req.url.includes(".bundle")) {
        res.setHeader("Content-Type", "application/javascript; charset=utf-8")
      }

      // Corriger les types MIME pour les assets
      if (req.url && req.url.includes(".js")) {
        res.setHeader("Content-Type", "application/javascript; charset=utf-8")
      }

      if (req.url && req.url.includes(".json")) {
        res.setHeader("Content-Type", "application/json; charset=utf-8")
      }

      // Headers CORS pour éviter les erreurs
      res.setHeader("Access-Control-Allow-Origin", "*")
      res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
      res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization")

      return middleware(req, res, next)
    }
  },
}

// Résoudre les conflits de modules
config.resolver = {
  ...config.resolver,
  alias: {
    "@": __dirname,
  },
  // Éviter les doublons qui causent 'chrome already declared'
  dedupe: ["react", "react-native", "expo-router", "expo"],
  // Résoudre les extensions dans l'ordre
  sourceExts: ["js", "jsx", "ts", "tsx", "json"],
  assetExts: ["png", "jpg", "jpeg", "gif", "svg", "ttf", "otf", "woff", "woff2"],
}

// Configuration transformer SIMPLIFIÉE (sans babel-transformer manquant)
config.transformer = {
  ...config.transformer,
  minifierConfig: {
    // Désactiver la minification en dev pour éviter les erreurs
    keep_fnames: true,
    mangle: {
      keep_fnames: true,
    },
  },
  // Éviter les conflits de transformation
  unstable_allowRequireContext: true,
}

// Configuration watchman pour éviter les erreurs de cache
config.watchFolders = [__dirname]

module.exports = config
