"use client"
import { Text, StyleSheet, Platform } from "react-native"
import { WebSafeView } from "./WebSafeView"
import { WebSafeTouchableOpacity } from "./WebSafeTouchableOpacity"

interface ErrorFallbackProps {
  error?: Error
  resetError?: () => void
}

export default function ErrorFallback({ error, resetError }: ErrorFallbackProps) {
  const handleReload = () => {
    if (resetError) {
      resetError()
    } else {
      // Recharger la page sur web
      if (Platform.OS === "web") {
        window.location.reload()
      }
    }
  }

  const handleClearCache = () => {
    if (Platform.OS === "web") {
      // Nettoyer le cache du navigateur
      if ("caches" in window) {
        caches.keys().then((names) => {
          names.forEach((name) => {
            caches.delete(name)
          })
        })
      }

      // Nettoyer le localStorage
      localStorage.clear()
      sessionStorage.clear()

      // Recharger
      window.location.reload()
    }
  }

  return (
    <WebSafeView style={styles.container}>
      <Text style={styles.title}>🔧 Erreur détectée</Text>

      <WebSafeView style={styles.errorContainer}>
        <Text style={styles.errorTitle}>Détails de l'erreur :</Text>
        <Text style={styles.errorMessage}>{error?.message || "Erreur inconnue du serveur Expo"}</Text>
      </WebSafeView>

      <WebSafeView style={styles.solutionsContainer}>
        <Text style={styles.solutionsTitle}>🛠️ Solutions :</Text>

        <WebSafeTouchableOpacity style={styles.solutionButton} onPress={handleReload}>
          <Text style={styles.solutionText}>🔄 Recharger l'application</Text>
        </WebSafeTouchableOpacity>

        <WebSafeTouchableOpacity style={styles.solutionButton} onPress={handleClearCache}>
          <Text style={styles.solutionText}>🧹 Nettoyer le cache</Text>
        </WebSafeTouchableOpacity>

        <WebSafeView style={styles.instructionsContainer}>
          <Text style={styles.instructionsTitle}>📋 Instructions manuelles :</Text>
          <Text style={styles.instruction}>1. Fermer tous les onglets</Text>
          <Text style={styles.instruction}>2. Ouvrir en mode incognito</Text>
          <Text style={styles.instruction}>3. Aller sur http://localhost:8082</Text>
          <Text style={styles.instruction}>4. Ou essayer un autre navigateur</Text>
        </WebSafeView>
      </WebSafeView>

      <WebSafeView style={styles.debugContainer}>
        <Text style={styles.debugTitle}>🐛 Debug Info :</Text>
        <Text style={styles.debugText}>Platform: {Platform.OS}</Text>
        <Text style={styles.debugText}>
          User Agent: {Platform.OS === "web" ? navigator.userAgent.substring(0, 50) + "..." : "N/A"}
        </Text>
        <Text style={styles.debugText}>Timestamp: {new Date().toISOString()}</Text>
      </WebSafeView>
    </WebSafeView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: "#f5f5f5",
    justifyContent: "center",
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    textAlign: "center",
    marginBottom: 20,
    color: "#e74c3c",
  },
  errorContainer: {
    backgroundColor: "#fff",
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
    borderLeftWidth: 4,
    borderLeftColor: "#e74c3c",
  },
  errorTitle: {
    fontSize: 16,
    fontWeight: "bold",
    marginBottom: 10,
    color: "#2c3e50",
  },
  errorMessage: {
    fontSize: 14,
    color: "#7f8c8d",
    fontFamily: Platform.OS === "ios" ? "Courier" : "monospace",
  },
  solutionsContainer: {
    backgroundColor: "#fff",
    padding: 15,
    borderRadius: 10,
    marginBottom: 20,
  },
  solutionsTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 15,
    color: "#27ae60",
  },
  solutionButton: {
    backgroundColor: "#3498db",
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
    alignItems: "center",
  },
  solutionText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
  instructionsContainer: {
    marginTop: 15,
    padding: 10,
    backgroundColor: "#ecf0f1",
    borderRadius: 8,
  },
  instructionsTitle: {
    fontSize: 16,
    fontWeight: "bold",
    marginBottom: 10,
    color: "#2c3e50",
  },
  instruction: {
    fontSize: 14,
    marginBottom: 5,
    color: "#34495e",
  },
  debugContainer: {
    backgroundColor: "#2c3e50",
    padding: 15,
    borderRadius: 10,
  },
  debugTitle: {
    fontSize: 16,
    fontWeight: "bold",
    marginBottom: 10,
    color: "#ecf0f1",
  },
  debugText: {
    fontSize: 12,
    color: "#bdc3c7",
    fontFamily: Platform.OS === "ios" ? "Courier" : "monospace",
    marginBottom: 3,
  },
})
