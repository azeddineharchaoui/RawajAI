"use client"

import { useState, useRef, useEffect } from "react"
import { Text, TextInput, ScrollView, StyleSheet, KeyboardAvoidingView, Platform, Alert } from "react-native"
import { WebSafeView } from "./WebSafeView"
import { WebSafeTouchableOpacity } from "./WebSafeTouchableOpacity"
import { FreeAIService } from "../services/freeAIService"
import { FixedSpeechService } from "../services/fixedSpeechService"
import { OptimizedTTSService } from "../services/optimizedTTSService"

interface Message {
  id: string
  text: string
  isUser: boolean
  timestamp: Date
  language: string
  hasAudio?: boolean
  isTyping?: boolean
}

interface Document {
  id: string
  name: string
  content: string
  type: string
  uploadDate: Date
}

// Détection de langue ultra-précise
const detectLanguageUltra = (text: string): "fr" | "en" | "ar" | "darija" => {
  const lowerText = text.toLowerCase()

  // Darija - Patterns très spécifiques
  const darijaPatterns = [
    /\b(wa7ed|jouj|tlata|arb3a|khamsa|stta|sb3a|tmnya|ts3od|3ashra)\b/,
    /\b(dyali|dyalek|dyalo|dyalha|dyal|li|lli)\b/,
    /\b(hna|hnak|bach|wach|ila|daba|ghir|bghit|bghiti|bghina)\b/,
    /\b(katdir|katdiri|kandiro|kandir|ndir|tdiri)\b/,
    /\b(kifach|kifash|ash|ashno|3lash|mnin|fin)\b/,
    /\b(3andi|3andek|3ando|3andha|3andna|3andhum)\b/,
    /\b(f|mn|3la|m3a|b7al|b7ala|zay|kay)\b/,
    /salam.*kifach|ahlan.*ash|marhaba.*wach/,
    /[0-9].*\b(dyali|dyal|bach|wach|3la|mn)\b/,
  ]

  if (darijaPatterns.some((pattern) => pattern.test(lowerText))) {
    return "darija"
  }

  // Arabe classique
  if (/[\u0600-\u06FF]/.test(text) && !/[a-zA-Z0-9]/.test(text)) {
    return "ar"
  }

  // Anglais
  const englishIndicators = [
    /\b(the|and|you|are|have|with|for|this|that|from|they|know|want|been|good|much|some|time|very|when|come|here|just|like|long|make|many|over|such|take|than|them|well|were)\b/g,
    /\b(hello|help|how|what|where|why|who|which|would|could|should|will|can|may|might)\b/g,
    /\b(forecast|inventory|supply|chain|logistics|transport|stock|warehouse|demand)\b/g,
  ]

  const englishMatches = englishIndicators.reduce((count, pattern) => {
    const matches = lowerText.match(pattern)
    return count + (matches ? matches.length : 0)
  }, 0)

  if (englishMatches >= 3) {
    return "en"
  }

  return "fr"
}

// Système RAG simple mais efficace
class SimpleRAGSystem {
  private documents: Document[] = []

  addDocument(doc: Document) {
    this.documents.push(doc)
    console.log(`📄 Document ajouté: ${doc.name}`)
  }

  findRelevantInfo(query: string): { found: boolean; content: string; sources: string[] } {
    const lowerQuery = query.toLowerCase()
    const queryWords = lowerQuery.split(/\s+/).filter((word) => word.length > 2)

    let relevantContent = ""
    const sources: string[] = []
    let found = false

    for (const doc of this.documents) {
      const docContent = doc.content.toLowerCase()
      const docLines = doc.content.split("\n")

      // Recherche par mots-clés
      const matchingLines = docLines.filter((line) => {
        const lineLower = line.toLowerCase()
        return queryWords.some((word) => lineLower.includes(word))
      })

      if (matchingLines.length > 0) {
        found = true
        sources.push(doc.name)
        relevantContent += `\n\n**${doc.name}:**\n${matchingLines.slice(0, 3).join("\n")}`
      }
    }

    return { found, content: relevantContent, sources }
  }

  getDocumentCount(): number {
    return this.documents.length
  }
}

export default function FixedUltimateAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "🚀 Salut ! Je suis RawaJAI, ton assistant supply chain CORRIGÉ.\n\n✨ **Fonctionnalités FIXES :**\n• 🎤 **Speech RÉEL** - Reconnaissance vocale corrigée\n• 🔊 **Audio optimisé** - Plus de répétitions\n• 🤖 **IA stable** - Réponses fiables\n• 🌍 **Détection auto** - Langue détectée précisément\n• 📱 **Mobile/Web** - Compatible partout\n\nDis 'aide' pour commencer !",
      isUser: false,
      timestamp: new Date(),
      language: "fr",
    },
  ])

  const [inputText, setInputText] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [currentLanguage, setCurrentLanguage] = useState<"fr" | "en" | "ar" | "darija">("fr")
  const [speechEnabled, setSpeechEnabled] = useState(true)
  const [documents, setDocuments] = useState<Document[]>([])

  // Services CORRIGÉS
  const aiService = useRef(new FreeAIService())
  const speechService = useRef(new FixedSpeechService())
  const ttsService = useRef(new OptimizedTTSService())
  const ragSystem = useRef(new SimpleRAGSystem())
  const scrollViewRef = useRef<ScrollView>(null)

  // Vérifier si la question concerne la supply chain
  const isSupplyChainRelated = (query: string): boolean => {
    const supplyChainTerms = [
      "stock",
      "inventaire",
      "prévision",
      "logistique",
      "transport",
      "livraison",
      "fournisseur",
      "client",
      "entrepôt",
      "commande",
      "coût",
      "prix",
      "demande",
      "optimisation",
      "performance",
      "efficacité",
      "kpi",
      "analyse",
      "données",
      "supply chain",
      "chaîne",
      "approvisionnement",
      "distribution",
      "procurement",
      "inventory",
      "forecast",
      "logistics",
      "supplier",
      "customer",
      "warehouse",
      "order",
      "cost",
      "optimization",
      "analytics",
      "transportation",
      "delivery",
      "مخزون",
      "تنبؤ",
      "لوجستيات",
      "مورد",
      "عميل",
      "مستودع",
      "طلب",
      "تكلفة",
      "تحسين",
      "أداء",
      "كفاءة",
      "تحليل",
      "بيانات",
      "توريد",
      "سلسلة",
      "tawaqo3",
      "ta7sin",
      "ta7lil",
      "talab",
    ]

    const lowerQuery = query.toLowerCase()
    return supplyChainTerms.some((term) => lowerQuery.includes(term.toLowerCase()))
  }

  // Génération de réponse intelligente
  const generateIntelligentResponse = async (
    query: string,
    language: "fr" | "en" | "ar" | "darija",
  ): Promise<string> => {
    const lowerQuery = query.toLowerCase()

    // Vérifier si c'est hors contexte
    if (!isSupplyChainRelated(query)) {
      const redirectResponses = {
        fr: "🎯 Je suis spécialisé en supply chain ! Pose-moi des questions sur :\n• 📊 Prévisions et analyses\n• 📦 Gestion des stocks\n• 🚚 Logistique et transport\n• 📈 Optimisation des processus\n\nComment puis-je t'aider ?",
        en: "🎯 I specialize in supply chain! Ask me about:\n• 📊 Forecasting and analysis\n• 📦 Inventory management\n• 🚚 Logistics and transport\n• 📈 Process optimization\n\nHow can I help?",
        ar: "🎯 أنا متخصص في سلسلة التوريد! اسألني عن:\n• 📊 التنبؤ والتحليل\n• 📦 إدارة المخزون\n• 🚚 اللوجستيات والنقل\n• 📈 تحسين العمليات\n\nكيف يمكنني مساعدتك؟",
        darija:
          "🎯 Ana mutakhassis f supply chain! Soulni 3la:\n• 📊 Tawaqo3 w ta7lil\n• 📦 Tadbir stock\n• 🚚 Logistique w transport\n• 📈 Ta7sin 3amaliyat\n\nKifach n9der n3awnek?",
      }
      return redirectResponses[language]
    }

    // Recherche RAG
    const ragResult = ragSystem.current.findRelevantInfo(query)

    // Salutations
    if (/bonjour|salut|hello|hi|salam|ahlan|marhaba/.test(lowerQuery)) {
      const greetings = {
        fr: "🌟 Salut ! Je suis RawaJAI, ton expert supply chain. Comment puis-je optimiser ta logistique aujourd'hui ?",
        en: "🌟 Hey! I'm RawaJAI, your supply chain expert. How can I optimize your logistics today?",
        ar: "🌟 أهلاً! أنا راوا جاي، خبيرك في سلسلة التوريد. كيف يمكنني تحسين لوجستياتك اليوم؟",
        darija: "🌟 Ahlan! Ana RawaJAI, expert dyalek f supply chain. Kifach n9der n7assan logistique dyalek lyoum?",
      }
      return greetings[language] + (ragResult.found ? ragResult.content : "")
    }

    // Aide
    if (/aide|help|mosa3ada|3awn/.test(lowerQuery)) {
      const helpResponses = {
        fr: `🧠 **Mes capacités :**

• 📊 **Prévisions** - Méthodes et optimisation
• 📦 **Stock** - Calculs EOQ, stock de sécurité
• 🚚 **Logistique** - Transport, distribution
• 📈 **Analytics** - KPIs, tableaux de bord
• 🎤 **Vocal** - Parle-moi directement
• 📄 **Documents** - J'analyse tes fichiers

**Pose-moi tes questions supply chain !**`,

        en: `🧠 **My capabilities:**

• 📊 **Forecasting** - Methods and optimization
• 📦 **Inventory** - EOQ calculations, safety stock
• 🚚 **Logistics** - Transport, distribution
• 📈 **Analytics** - KPIs, dashboards
• 🎤 **Voice** - Speak to me directly
• 📄 **Documents** - I analyze your files

**Ask me your supply chain questions!**`,

        ar: `🧠 **قدراتي:**

• 📊 **التنبؤ** - الطرق والتحسين
• 📦 **المخزون** - حسابات EOQ، مخزون الأمان
• 🚚 **اللوجستيات** - النقل، التوزيع
• 📈 **التحليلات** - مؤشرات، لوحات معلومات
• 🎤 **الصوت** - تحدث معي مباشرة
• 📄 **المستندات** - أحلل ملفاتك

**اسألني أسئلة سلسلة التوريد!**`,

        darija: `🧠 **Qudrati:**

• 📊 **Tawaqo3** - Turuq w ta7sin
• 📦 **Stock** - 7isabat EOQ, stock sécurité
• 🚚 **Logistique** - Transport, tawzi3
• 📈 **Analytics** - KPIs, dashboards
• 🎤 **Sout** - Hder m3aya direct
• 📄 **Watha2eq** - Kan7allel files dyalek

**Soulni as2ila supply chain!**`,
      }
      return helpResponses[language]
    }

    // Réponses spécialisées
    if (/stock|inventaire|inventory/.test(lowerQuery)) {
      const stockResponses = {
        fr: `📦 **Optimisation Stock :**

🔢 **Formules clés :**
• EOQ = √(2 × Demande × Coût commande / Coût stockage)
• Stock sécurité = Z × √(Délai × Variance)
• Point commande = Demande × Délai + Stock sécurité

💡 **Stratégies :**
• Classification ABC
• Révision continue
• Just-in-time
• Analyse coûts cachés

🎯 **Objectif :** -20% stock, +15% service`,

        en: `📦 **Inventory Optimization:**

🔢 **Key formulas:**
• EOQ = √(2 × Demand × Order cost / Holding cost)
• Safety stock = Z × √(Lead time × Variance)
• Reorder point = Demand × Lead time + Safety stock

💡 **Strategies:**
• ABC classification
• Continuous review
• Just-in-time
• Hidden cost analysis

🎯 **Goal:** -20% inventory, +15% service`,

        darija: `📦 **Ta7sin Stock:**

🔢 **Formulas muhimmin:**
• EOQ = √(2 × Talab × Coût commande / Coût stockage)
• Stock sécurité = Z × √(Délai × Variance)
• Point commande = Talab × Délai + Stock sécurité

💡 **Strategies:**
• Taqsim ABC
• Muraja3a mustamirra
• Just-in-time
• Ta7lil coûts makhfiya

🎯 **Hadaf:** -20% stock, +15% service`,
      }
      return stockResponses[language as keyof typeof stockResponses] || stockResponses.fr
    }

    if (/prévision|forecast|tawaqo3/.test(lowerQuery)) {
      const forecastResponses = {
        fr: `📊 **Prévisions Intelligentes :**

🎯 **Méthodes :**
• Moyenne mobile pondérée
• Lissage exponentiel
• Régression linéaire
• Machine Learning

📈 **Facteurs :**
• Saisonnalité
• Événements promotionnels
• Tendances marché
• Données externes

💡 **Amélioration :**
• Mesurer MAPE, MAD
• Combiner méthodes
• Révision mensuelle

🎯 **Précision :** 85-95%`,

        en: `📊 **Smart Forecasting:**

🎯 **Methods:**
• Weighted moving average
• Exponential smoothing
• Linear regression
• Machine Learning

📈 **Factors:**
• Seasonality
• Promotional events
• Market trends
• External data

💡 **Improvement:**
• Measure MAPE, MAD
• Combine methods
• Monthly review

🎯 **Accuracy:** 85-95%`,

        darija: `📊 **Tawaqo3 Dkiya:**

🎯 **Turuq:**
• Moyenne mobile muwazana
• Lissage exponentiel
• Regression khattiya
• Machine Learning

📈 **3awamil:**
• Mawsimiya
• A7dath tarwijiya
• Trends suq
• Data kharijiya

💡 **Ta7sin:**
• Qiyass MAPE, MAD
• Khallat turuq
• Muraja3a shahriya

🎯 **Diqa:** 85-95%`,
      }
      return forecastResponses[language as keyof typeof forecastResponses] || forecastResponses.fr
    }

    // Réponse par défaut
    const defaultResponses = {
      fr: `🤔 **Question supply chain intéressante !**

Je peux t'aider avec :
• 📊 Prévisions et analyses
• 📦 Optimisation de stock
• 🚚 Logistique et transport
• 📈 KPIs et performance

💡 **Précise ta question** pour une réponse détaillée !`,

      en: `🤔 **Interesting supply chain question!**

I can help you with:
• 📊 Forecasting and analysis
• 📦 Inventory optimization
• 🚚 Logistics and transport
• 📈 KPIs and performance

💡 **Be more specific** for a detailed answer!`,

      darija: `🤔 **Su2al supply chain muhimm!**

N9der n3awnek f:
• 📊 Tawaqo3 w ta7lil
• 📦 Ta7sin stock
• 🚚 Logistique w transport
• 📈 KPIs w performance

💡 **Wad7 su2alek** bach n3tik jawab mfasal!`,
    }

    return (
      defaultResponses[language as keyof typeof defaultResponses] ||
      defaultResponses.fr + (ragResult.found ? ragResult.content : "")
    )
  }

  // Speech-to-Text CORRIGÉ
  const handleVoiceRecord = async () => {
    if (isRecording) {
      setIsRecording(false)
      speechService.current.stopListening()
      return
    }

    try {
      setIsRecording(true)

      // Attendre que le service soit prêt
      let attempts = 0
      while (!speechService.current.isReady() && attempts < 10) {
        await new Promise((resolve) => setTimeout(resolve, 100))
        attempts++
      }

      if (!speechService.current.isReady()) {
        throw new Error("Service de reconnaissance vocale non initialisé")
      }

      if (speechService.current.isSupported()) {
        console.log("🎤 Utilisation Web Speech API...")

        try {
          const transcript = await speechService.current.startListening(currentLanguage)
          setInputText(transcript)
          setIsRecording(false)

          Alert.alert("🎤 Transcription réussie", `J'ai entendu : "${transcript}"\n\nAppuie sur Envoyer !`, [
            { text: "OK" },
          ])
        } catch (speechError) {
          console.log("🎤 Web Speech API échoué, essai MediaRecorder...")

          // Fallback MediaRecorder
          const audioBlob = await speechService.current.startRecordingWithMediaRecorder()
          const transcript = await speechService.current.transcribeAudio(audioBlob)

          setInputText(transcript)
          setIsRecording(false)

          Alert.alert("🎤 Transcription réussie", `J'ai transcrit : "${transcript}"\n\nAppuie sur Envoyer !`, [
            { text: "OK" },
          ])
        }
      } else {
        // Utiliser MediaRecorder directement
        console.log("🎤 Web Speech API non supporté, utilisation MediaRecorder...")
        const audioBlob = await speechService.current.startRecordingWithMediaRecorder()
        const transcript = await speechService.current.transcribeAudio(audioBlob)

        setInputText(transcript)
        setIsRecording(false)

        Alert.alert("🎤 Transcription réussie", `J'ai transcrit : "${transcript}"\n\nAppuie sur Envoyer !`, [
          { text: "OK" },
        ])
      }
    } catch (error) {
      setIsRecording(false)
      console.error("🎤 Erreur enregistrement:", error)

      let errorMessage = "Erreur d'enregistrement"
      if (error instanceof Error) {
        errorMessage = error.message
      }

      Alert.alert("❌ Erreur vocale", errorMessage, [{ text: "OK" }])
    }
  }

  // Effet de frappe
  const addTypingEffect = (text: string, messageId: string) => {
    let i = 0
    const speed = 20

    const typeWriter = () => {
      if (i < text.length) {
        setMessages((prev) =>
          prev.map((msg) => (msg.id === messageId ? { ...msg, text: text.substring(0, i + 1) } : msg)),
        )
        i++
        setTimeout(typeWriter, speed)
      } else {
        // Lecture vocale après la frappe
        if (speechEnabled) {
          setTimeout(() => {
            ttsService.current.speak(text, currentLanguage)
          }, 500)
        }
      }
    }

    typeWriter()
  }

  const handleSend = async () => {
    if (!inputText.trim() || isLoading) return

    // Détection automatique de la langue
    const detectedLang = detectLanguageUltra(inputText.trim())
    setCurrentLanguage(detectedLang)

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText.trim(),
      isUser: true,
      timestamp: new Date(),
      language: detectedLang,
    }

    setMessages((prev) => [...prev, userMessage])
    const currentInput = inputText.trim()
    setInputText("")
    setIsLoading(true)

    // Message de typing
    const typingMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: "🧠 Analyse en cours...",
      isUser: false,
      timestamp: new Date(),
      language: detectedLang,
      isTyping: true,
    }

    setMessages((prev) => [...prev, typingMessage])

    try {
      // Génération de réponse
      const response = await generateIntelligentResponse(currentInput, detectedLang)

      // Supprimer le message de typing
      setMessages((prev) => prev.filter((msg) => !msg.isTyping))

      const aiMessage: Message = {
        id: (Date.now() + 2).toString(),
        text: "",
        isUser: false,
        timestamp: new Date(),
        language: detectedLang,
        hasAudio: speechEnabled,
      }

      setMessages((prev) => [...prev, aiMessage])

      // Effet de frappe
      addTypingEffect(response, aiMessage.id)
    } catch (error) {
      console.error("Erreur génération:", error)

      setMessages((prev) => prev.filter((msg) => !msg.isTyping))

      const errorMessage: Message = {
        id: (Date.now() + 3).toString(),
        text: "❌ Désolé, j'ai eu un problème. Peux-tu reformuler ?",
        isUser: false,
        timestamp: new Date(),
        language: detectedLang,
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const toggleSpeech = () => {
    if (speechEnabled) {
      ttsService.current.stop()
    }
    setSpeechEnabled(!speechEnabled)

    const message = speechEnabled ? "🔇 Audio désactivé" : "🔊 Audio activé"
    Alert.alert("🔊 Text-to-Speech", message)
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
        <Text style={styles.headerTitle}>🧠 RawaJAI CORRIGÉ</Text>
        <Text style={styles.headerSubtitle}>
          Speech RÉEL • {ragSystem.current.getDocumentCount()} docs • {currentLanguage.toUpperCase()} • Audio optimisé
        </Text>

        <WebSafeView style={styles.controlsRow}>
          <WebSafeView style={styles.statusIndicator}>
            <Text style={styles.statusText}>🌍 {currentLanguage.toUpperCase()}</Text>
          </WebSafeView>

          <WebSafeTouchableOpacity
            style={[styles.audioButton, speechEnabled && styles.audioButtonActive]}
            onPress={toggleSpeech}
          >
            <Text style={styles.audioButtonText}>{speechEnabled ? "🔊" : "🔇"}</Text>
          </WebSafeTouchableOpacity>

          <WebSafeView style={styles.speechIndicator}>
            <Text style={styles.speechIndicatorText}>{speechService.current.isSupported() ? "🎤✅" : "🎤❌"}</Text>
          </WebSafeView>
        </WebSafeView>
      </WebSafeView>

      <ScrollView ref={scrollViewRef} style={styles.messagesContainer} showsVerticalScrollIndicator={false}>
        {messages.map((message) => (
          <WebSafeView
            key={message.id}
            style={[
              styles.messageContainer,
              message.isUser ? styles.userMessage : styles.aiMessage,
              message.isTyping && styles.typingMessage,
            ]}
          >
            <Text style={[styles.messageText, message.isUser ? styles.userMessageText : styles.aiMessageText]}>
              {message.text}
            </Text>
            {!message.isTyping && (
              <WebSafeView style={styles.messageFooter}>
                <Text style={styles.timestamp}>
                  {message.timestamp.toLocaleTimeString("fr-FR", { timeStyle: "short" })}
                </Text>
                <WebSafeView style={styles.messageIcons}>
                  {message.hasAudio && <Text style={styles.audioIcon}>🔊</Text>}
                  <Text style={styles.languageTag}>{message.language.toUpperCase()}</Text>
                </WebSafeView>
              </WebSafeView>
            )}
          </WebSafeView>
        ))}
      </ScrollView>

      <WebSafeView style={styles.inputContainer}>
        <WebSafeView style={styles.inputRow}>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Tapez ou parlez (🎤) - Détection automatique..."
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

          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("optimiser stock")}>
            <Text style={styles.quickButtonText}>📦 Stock</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("prévisions")}>
            <Text style={styles.quickButtonText}>📊 Prévisions</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("logistique")}>
            <Text style={styles.quickButtonText}>🚚 Logistique</Text>
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
    fontSize: 12,
    color: "#718096",
    textAlign: "center",
    marginTop: 4,
  },
  controlsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 12,
  },
  statusIndicator: {
    backgroundColor: "#f1f5f9",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  statusText: {
    fontSize: 12,
    color: "#475569",
    fontWeight: "600",
  },
  audioButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "#f1f5f9",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  audioButtonActive: {
    backgroundColor: "#10B981",
    borderColor: "#10B981",
  },
  audioButtonText: {
    fontSize: 16,
  },
  speechIndicator: {
    backgroundColor: "#f1f5f9",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  speechIndicatorText: {
    fontSize: 12,
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
    maxWidth: "85%",
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
  typingMessage: {
    backgroundColor: "#f8f9fa",
    borderColor: "#3B82F6",
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
  messageFooter: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 6,
  },
  timestamp: {
    fontSize: 12,
    color: "#718096",
  },
  messageIcons: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  audioIcon: {
    fontSize: 12,
  },
  languageTag: {
    fontSize: 10,
    color: "#9CA3AF",
    backgroundColor: "#F3F4F6",
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 8,
    fontWeight: "600",
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
    gap: 6,
    justifyContent: "center",
    flexWrap: "wrap",
  },
  quickButton: {
    backgroundColor: "#f1f5f9",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  quickButtonText: {
    fontSize: 11,
    color: "#475569",
    fontWeight: "500",
  },
})
