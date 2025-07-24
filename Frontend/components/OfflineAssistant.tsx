"use client"

import { useState, useRef, useEffect } from "react"
import { Text, TextInput, ScrollView, StyleSheet, KeyboardAvoidingView, Platform, Alert } from "react-native"
import { WebSafeView } from "./WebSafeView"
import { WebSafeTouchableOpacity } from "./WebSafeTouchableOpacity"
import * as Speech from "expo-speech"

interface Message {
  id: string
  text: string
  isUser: boolean
  timestamp: Date
}

// Base de connaissances locale - RÉPONSES INSTANTANÉES
const KNOWLEDGE_BASE = {
  greetings: {
    keywords: ["bonjour", "salut", "hello", "hi", "bonsoir"],
    responses: [
      "🎉 Bonjour ! Je suis RawaJAI, votre assistant supply chain. Comment puis-je vous aider ?",
      "👋 Salut ! Prêt à optimiser votre supply chain ensemble ?",
      "✨ Hello ! Je suis là pour vous aider avec vos questions logistiques !",
    ],
  },
  help: {
    keywords: ["aide", "help", "que peux-tu faire", "capacités"],
    responses: [
      `🔧 **Mes capacités :**

• 📊 **Prévisions** - Conseils sur la demande
• 📦 **Inventaire** - Optimisation des stocks  
• 🚚 **Logistique** - Gestion des flux
• 📈 **Analytics** - Analyse de performance
• 💡 **Conseils** - Stratégies d'amélioration

Que voulez-vous explorer ?`,
    ],
  },
  forecast: {
    keywords: ["prévision", "forecast", "demande", "tendance", "prédiction"],
    responses: [
      "📊 **Prévisions de demande :**\n\n• Analysez vos données historiques\n• Identifiez les tendances saisonnières\n• Utilisez des moyennes mobiles\n• Considérez les événements externes\n\nVoulez-vous des conseils spécifiques ?",
      "📈 Pour de bonnes prévisions :\n\n1. **Collectez** 12-24 mois de données\n2. **Nettoyez** les anomalies\n3. **Segmentez** par produit/région\n4. **Validez** avec les équipes terrain",
    ],
  },
  inventory: {
    keywords: ["inventaire", "inventory", "stock", "stockage", "entrepôt"],
    responses: [
      "📦 **Optimisation d'inventaire :**\n\n• **Stock de sécurité** = √(délai × demande moyenne)\n• **Point de commande** = demande × délai + stock sécurité\n• **Rotation** = Coût des ventes / Stock moyen\n\nQuel aspect vous intéresse ?",
      "🎯 **Règles d'or des stocks :**\n\n• 80/20 : 20% des produits = 80% de la valeur\n• Classement ABC pour prioriser\n• Révision mensuelle des seuils\n• Automatisation des commandes",
    ],
  },
  logistics: {
    keywords: ["logistique", "transport", "livraison", "distribution", "supply chain"],
    responses: [
      "🚚 **Optimisation logistique :**\n\n• **Consolidation** des envois\n• **Planification** des tournées\n• **Tracking** en temps réel\n• **Partenariats** transporteurs\n\nSur quoi voulez-vous vous concentrer ?",
      "⚡ **KPIs logistiques clés :**\n\n• Taux de service : >95%\n• Coût transport/CA : <5%\n• Délai moyen livraison\n• Taux de retour : <2%",
    ],
  },
  analytics: {
    keywords: ["analyse", "analytics", "performance", "kpi", "métrique"],
    responses: [
      "📈 **Analytics Supply Chain :**\n\n• **Taux de service** client\n• **Rotation** des stocks\n• **Coûts** logistiques\n• **Délais** de livraison\n• **Qualité** des prévisions\n\nQuel KPI vous préoccupe ?",
      "🎯 **Tableau de bord essentiel :**\n\n• Disponibilité produits : 98%+\n• Précision prévisions : 85%+\n• Coût stockage/CA : 2-4%\n• Délai cycle commande : <48h",
    ],
  },
  thanks: {
    keywords: ["merci", "thank", "parfait", "super", "génial"],
    responses: [
      "😊 De rien ! Ravi de vous aider avec votre supply chain !",
      "🎉 Parfait ! N'hésitez pas pour d'autres questions !",
      "💪 C'est un plaisir ! Votre supply chain va être au top !",
    ],
  },
}

export default function OfflineAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "🚀 Bonjour ! Je suis RawaJAI, votre assistant supply chain OFFLINE. Tapez 'aide' pour voir mes capacités !",
      isUser: false,
      timestamp: new Date(),
    },
  ])
  const [inputText, setInputText] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [speechEnabled, setSpeechEnabled] = useState(true)
  const scrollViewRef = useRef<ScrollView>(null)

  // Fonction pour trouver une réponse
  const findResponse = (query: string): string => {
    const lowerQuery = query.toLowerCase().trim()

    // Chercher dans chaque catégorie
    for (const [category, data] of Object.entries(KNOWLEDGE_BASE)) {
      for (const keyword of data.keywords) {
        if (lowerQuery.includes(keyword)) {
          const responses = data.responses
          return responses[Math.floor(Math.random() * responses.length)]
        }
      }
    }

    // Réponse par défaut
    return `🤔 Question intéressante : "${query}"\n\n💡 **Suggestions :**\n• Tapez "aide" pour mes capacités\n• Demandez des "prévisions"\n• Parlez d'"inventaire" ou de "logistique"\n• Explorez les "analytics"\n\nComment puis-je vous aider concrètement ?`
  }

  // Fonction pour parler (Text-to-Speech)
  const speakText = async (text: string) => {
    if (!speechEnabled || Platform.OS === "web") return

    try {
      // Nettoyer le texte des emojis et markdown
      const cleanText = text
        .replace(/[🎉👋✨🔧📊📦🚚📈💡😊🎯⚡💪🤔🚀]/gu, "")
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/•/g, "")
        .replace(/\n+/g, ". ")
        .trim()

      await Speech.speak(cleanText, {
        language: "fr-FR",
        pitch: 1.0,
        rate: 0.9,
      })
    } catch (error) {
      console.warn("Speech error:", error)
    }
  }

  const handleSend = async () => {
    if (!inputText.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText.trim(),
      isUser: true,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputText("")
    setIsLoading(true)

    // Simulation d'un petit délai pour l'effet "réflexion"
    setTimeout(async () => {
      const response = findResponse(inputText.trim())

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response,
        isUser: false,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, aiMessage])
      setIsLoading(false)

      // Lire la réponse à voix haute
      await speakText(response)
    }, 500)
  }

  const handleVoiceRecord = () => {
    if (isRecording) {
      setIsRecording(false)
      Alert.alert("🎤 Enregistrement", "Fonction vocale en développement. Utilisez le clavier pour l'instant !")
    } else {
      setIsRecording(true)
      // Simuler l'arrêt après 3 secondes
      setTimeout(() => setIsRecording(false), 3000)
    }
  }

  const toggleSpeech = () => {
    setSpeechEnabled(!speechEnabled)
    Alert.alert("🔊 Text-to-Speech", speechEnabled ? "Audio désactivé" : "Audio activé")
  }

  useEffect(() => {
    scrollViewRef.current?.scrollToEnd({ animated: true })
  }, [messages])

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      keyboardVerticalOffset={Platform.OS === "ios" ? 90 : 0}
    >
      <WebSafeView style={styles.header}>
        <Text style={styles.headerTitle}>🚀 Assistant IA RawaJAI</Text>
        <Text style={styles.headerSubtitle}>Mode OFFLINE - Réponses instantanées</Text>

        <WebSafeView style={styles.statusRow}>
          <WebSafeView style={[styles.statusBadge, styles.onlineBadge]}>
            <Text style={styles.statusText}>✅ OFFLINE</Text>
          </WebSafeView>

          <WebSafeTouchableOpacity
            style={[styles.audioButton, speechEnabled && styles.audioButtonActive]}
            onPress={toggleSpeech}
          >
            <Text style={styles.audioButtonText}>{speechEnabled ? "🔊" : "🔇"}</Text>
          </WebSafeTouchableOpacity>
        </WebSafeView>
      </WebSafeView>

      <ScrollView ref={scrollViewRef} style={styles.messagesContainer} showsVerticalScrollIndicator={false}>
        {messages.map((message) => (
          <WebSafeView
            key={message.id}
            style={[styles.messageContainer, message.isUser ? styles.userMessage : styles.aiMessage]}
          >
            <Text style={[styles.messageText, message.isUser ? styles.userMessageText : styles.aiMessageText]}>
              {message.text}
            </Text>
            <Text style={styles.timestamp}>
              {message.timestamp.toLocaleTimeString("fr-FR", { timeStyle: "short" })}
            </Text>
          </WebSafeView>
        ))}

        {isLoading && (
          <WebSafeView style={[styles.messageContainer, styles.aiMessage]}>
            <Text style={[styles.messageText, styles.aiMessageText]}>🤖 Réflexion en cours...</Text>
          </WebSafeView>
        )}
      </ScrollView>

      <WebSafeView style={styles.inputContainer}>
        <WebSafeView style={styles.inputRow}>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Tapez votre question..."
            placeholderTextColor="#999"
            multiline
            maxLength={500}
            onSubmitEditing={handleSend}
            editable={!isLoading}
          />

          <WebSafeTouchableOpacity
            style={[styles.voiceButton, isRecording && styles.voiceButtonActive]}
            onPress={handleVoiceRecord}
            disabled={isLoading}
          >
            <Text style={styles.voiceButtonText}>{isRecording ? "⏹️" : "🎤"}</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity
            style={[styles.sendButton, (!inputText.trim() || isLoading) && styles.sendButtonDisabled]}
            onPress={handleSend}
            disabled={!inputText.trim() || isLoading}
          >
            <Text style={styles.sendButtonText}>➤</Text>
          </WebSafeTouchableOpacity>
        </WebSafeView>

        <WebSafeView style={styles.quickActions}>
          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("aide")}>
            <Text style={styles.quickButtonText}>💡 Aide</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("prévisions")}>
            <Text style={styles.quickButtonText}>📊 Prévisions</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("inventaire")}>
            <Text style={styles.quickButtonText}>📦 Stock</Text>
          </WebSafeTouchableOpacity>
        </WebSafeView>
      </WebSafeView>
    </KeyboardAvoidingView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#f5f7fa",
  },
  header: {
    backgroundColor: "#fff",
    paddingTop: Platform.OS === "ios" ? 50 : 20,
    paddingBottom: 15,
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    borderBottomColor: "#e1e8ed",
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#1a202c",
    textAlign: "center",
  },
  headerSubtitle: {
    fontSize: 14,
    color: "#718096",
    textAlign: "center",
    marginTop: 4,
  },
  statusRow: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    marginTop: 10,
    gap: 10,
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  onlineBadge: {
    backgroundColor: "#10B981",
  },
  statusText: {
    color: "white",
    fontSize: 12,
    fontWeight: "600",
  },
  audioButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "#f1f5f9",
    justifyContent: "center",
    alignItems: "center",
  },
  audioButtonActive: {
    backgroundColor: "#3B82F6",
  },
  audioButtonText: {
    fontSize: 16,
  },
  messagesContainer: {
    flex: 1,
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  messageContainer: {
    marginVertical: 4,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 18,
    maxWidth: "80%",
  },
  userMessage: {
    alignSelf: "flex-end",
    backgroundColor: "#007AFF",
  },
  aiMessage: {
    alignSelf: "flex-start",
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: "#e1e8ed",
  },
  messageText: {
    fontSize: 16,
    lineHeight: 22,
  },
  userMessageText: {
    color: "#fff",
  },
  aiMessageText: {
    color: "#1a202c",
  },
  timestamp: {
    fontSize: 12,
    color: "#718096",
    marginTop: 4,
    alignSelf: "flex-end",
  },
  inputContainer: {
    backgroundColor: "#fff",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderTopWidth: 1,
    borderTopColor: "#e1e8ed",
  },
  inputRow: {
    flexDirection: "row",
    alignItems: "flex-end",
    gap: 8,
    marginBottom: 8,
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: "#e1e8ed",
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    maxHeight: 100,
    backgroundColor: "#f8f9fa",
  },
  voiceButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "#f8f9fa",
    borderWidth: 1,
    borderColor: "#e1e8ed",
    justifyContent: "center",
    alignItems: "center",
  },
  voiceButtonActive: {
    backgroundColor: "#ff4757",
    borderColor: "#ff4757",
  },
  voiceButtonText: {
    fontSize: 18,
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "#007AFF",
    justifyContent: "center",
    alignItems: "center",
  },
  sendButtonDisabled: {
    backgroundColor: "#cbd5e0",
  },
  sendButtonText: {
    color: "#fff",
    fontSize: 18,
    fontWeight: "bold",
  },
  quickActions: {
    flexDirection: "row",
    gap: 8,
    justifyContent: "center",
  },
  quickButton: {
    backgroundColor: "#f1f5f9",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  quickButtonText: {
    fontSize: 12,
    color: "#475569",
    fontWeight: "500",
  },
})
