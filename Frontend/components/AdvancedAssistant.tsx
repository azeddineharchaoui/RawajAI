"use client"

import { useState, useRef, useEffect } from "react"
import { Text, TextInput, ScrollView, StyleSheet, KeyboardAvoidingView, Platform, Alert } from "react-native"
import { WebSafeView } from "./WebSafeView"
import { WebSafeTouchableOpacity } from "./WebSafeTouchableOpacity"
import * as Speech from "expo-speech"
import * as DocumentPicker from "expo-document-picker"
import { Audio } from "expo-av"

interface Message {
  id: string
  text: string
  isUser: boolean
  timestamp: Date
  language: string
  audioUrl?: string
}

interface Document {
  id: string
  name: string
  content: string
  type: string
  uploadDate: Date
}

// Système de modération - Mots interdits
const INAPPROPRIATE_WORDS = {
  fr: ["merde", "putain", "connard", "salope", "bordel"],
  ar: ["كلب", "حمار", "غبي"],
  darija: ["7mar", "kelb", "wa7ed"],
  en: ["fuck", "shit", "damn", "bitch", "asshole"],
}

// Base de connaissances RAG avec documents
class RAGSystem {
  private documents: Document[] = []
  private vectorStore: Map<string, number[]> = new Map()

  // Ajouter un document au système RAG
  addDocument(doc: Document) {
    this.documents.push(doc)
    // Simulation de vectorisation (dans un vrai système, utiliser des embeddings)
    const vector = this.createSimpleVector(doc.content)
    this.vectorStore.set(doc.id, vector)
  }

  // Créer un vecteur simple basé sur les mots-clés
  private createSimpleVector(text: string): number[] {
    const keywords = ["supply", "chain", "logistique", "stock", "prévision", "inventaire", "transport"]
    return keywords.map((keyword) => (text.toLowerCase().includes(keyword) ? 1 : 0))
  }

  // Rechercher des documents pertinents
  searchRelevantDocs(query: string, limit = 3): Document[] {
    const queryVector = this.createSimpleVector(query)

    const scores = this.documents.map((doc) => {
      const docVector = this.vectorStore.get(doc.id) || []
      const similarity = this.cosineSimilarity(queryVector, docVector)
      return { doc, similarity }
    })

    return scores
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit)
      .map((item) => item.doc)
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * (b[i] || 0), 0)
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0))
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0))
    return magnitudeA && magnitudeB ? dotProduct / (magnitudeA * magnitudeB) : 0
  }

  getAllDocuments(): Document[] {
    return this.documents
  }
}

// Système de réponses multilingues avec personnalité humaine
const MULTILINGUAL_RESPONSES = {
  greetings: {
    fr: [
      "🌟 Salut ! Je suis RawaJAI, ton assistant supply chain. Ravi de te rencontrer ! Comment ça va aujourd'hui ?",
      "👋 Hey ! C'est parti pour optimiser ta supply chain ensemble ! Tu as des défis intéressants pour moi ?",
      "✨ Bonjour ! Je suis là pour t'accompagner dans tes projets logistiques. Qu'est-ce qui t'amène ?",
    ],
    en: [
      "🌟 Hey there! I'm RawaJAI, your supply chain buddy. Great to meet you! How's your day going?",
      "👋 Hi! Ready to tackle some supply chain challenges together? What's on your mind?",
      "✨ Hello! I'm here to help you optimize your logistics. What brings you here today?",
    ],
    ar: [
      "🌟 أهلاً وسهلاً! أنا راوا جاي، مساعدك في سلسلة التوريد. كيف حالك اليوم؟",
      "👋 مرحباً! مستعد لحل تحديات سلسلة التوريد معاً؟ بماذا تفكر؟",
      "✨ السلام عليكم! أنا هنا لمساعدتك في تحسين اللوجستيات. ما الذي يجلبك هنا؟",
    ],
    darija: [
      "🌟 Ahlan! Ana RawaJAI, m3ak f supply chain. Kifach nta lyoum?",
      "👋 Salam! Wach bghiti n7ello chi mushkil f logistique m3a b3d?",
      "✨ Marhaba! Ana hna bach n3awnek f ta7sin transport w stock. Ash katdir?",
    ],
  },

  help: {
    fr: [
      `🧠 **Mes super-pouvoirs :**

• 📊 **Prévisions intelligentes** - J'analyse tes données comme un pro
• 📦 **Optimisation stock** - Fini les ruptures et surstocks !  
• 🚚 **Logistique fluide** - Transport optimisé, clients contents
• 📈 **Analytics poussés** - KPIs qui comptent vraiment
• 🤖 **IA conversationnelle** - Je comprends tes nuances
• 📄 **RAG Documents** - J'apprends de tes fichiers

**Alors, on commence par quoi ?** 🚀`,
    ],
    en: [
      `🧠 **My superpowers:**

• 📊 **Smart forecasting** - I analyze your data like a pro
• 📦 **Stock optimization** - No more stockouts or overstock!  
• 🚚 **Smooth logistics** - Optimized transport, happy customers
• 📈 **Advanced analytics** - KPIs that really matter
• 🤖 **Conversational AI** - I understand your nuances
• 📄 **RAG Documents** - I learn from your files

**So, what should we start with?** 🚀`,
    ],
    ar: [
      `🧠 **قدراتي الخارقة:**

• 📊 **التنبؤ الذكي** - أحلل بياناتك كالمحترفين
• 📦 **تحسين المخزون** - لا مزيد من النفاد أو الفائض!  
• 🚚 **لوجستيات سلسة** - نقل محسن، عملاء سعداء
• 📈 **تحليلات متقدمة** - مؤشرات مهمة حقاً
• 🤖 **ذكاء محادثة** - أفهم تفاصيلك
• 📄 **مستندات RAG** - أتعلم من ملفاتك

**إذن، بماذا نبدأ؟** 🚀`,
    ],
    darija: [
      `🧠 **Qudrati dyali:**

• 📊 **Tawaqo3 dkiya** - Kan7allel data dyalek b7al pro
• 📦 **Ta7sin stock** - Makainch nfad wla ziyada!  
• 🚚 **Logistique sa3ba** - Transport m7assan, clients far7anin
• 📈 **Analytics qwiya** - KPIs li muhimmin b9a9
• 🤖 **AI katkalem** - Kanfhem tafasil dyalek
• 📄 **Watha2eq RAG** - Kant3allem mn files dyalek

**Iwa, nbd2o b ash?** 🚀`,
    ],
  },

  moderation: {
    fr: [
      "😊 Hey, restons professionnels s'il te plaît ! Je suis là pour t'aider avec ta supply chain de manière constructive. Reformule ta question ?",
      "🤝 J'apprécie qu'on garde un ton respectueux dans nos échanges. Comment puis-je t'aider autrement ?",
      "✨ Gardons une ambiance positive ! Pose-moi plutôt une question sur la logistique ou l'optimisation.",
    ],
    en: [
      "😊 Hey, let's keep it professional please! I'm here to help with your supply chain constructively. Can you rephrase?",
      "🤝 I appreciate keeping a respectful tone in our exchanges. How else can I help you?",
      "✨ Let's maintain a positive vibe! Ask me about logistics or optimization instead.",
    ],
    ar: [
      "😊 دعنا نحافظ على الاحترافية من فضلك! أنا هنا لمساعدتك في سلسلة التوريد بطريقة بناءة. هل يمكنك إعادة صياغة سؤالك؟",
      "🤝 أقدر الحفاظ على نبرة محترمة في تبادلنا. كيف يمكنني مساعدتك بطريقة أخرى؟",
      "✨ دعنا نحافظ على جو إيجابي! اسألني عن اللوجستيات أو التحسين بدلاً من ذلك.",
    ],
    darija: [
      "😊 Khallina nkuno professionels 3afak! Ana hna bach n3awnek f supply chain b tariqa constructive. Wach t9der t3awed sual?",
      "🤝 Kan9adder nkhalliu ton respectueux f hadchi. Kifach n9der n3awnek b tariqa khra?",
      "✨ Khallina nkhallio jaw positif! Soulni 3la logistique wla optimization.",
    ],
  },
}

export default function AdvancedAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "🚀 Salut ! Je suis RawaJAI, ton assistant supply chain avec IA avancée. Je parle français, anglais, arabe et darija ! Dis-moi 'aide' pour découvrir mes capacités.",
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

  const scrollViewRef = useRef<ScrollView>(null)
  const ragSystem = useRef(new RAGSystem())
  const [recording, setRecording] = useState<Audio.Recording | null>(null)

  // Détection de langue automatique
  const detectLanguage = (text: string): "fr" | "en" | "ar" | "darija" => {
    const lowerText = text.toLowerCase()

    // Darija (mélange arabe-français avec chiffres)
    if (/[0-9]/.test(text) && /wa7ed|b7al|dyali|hna|bach|wach|kifach|3la|mn/.test(lowerText)) {
      return "darija"
    }

    // Arabe
    if (/[\u0600-\u06FF]/.test(text)) {
      return "ar"
    }

    // Anglais
    if (
      /\b(the|and|you|are|have|with|for|this|that|from|they|know|want|been|good|much|some|time|very|when|come|here|just|like|long|make|many|over|such|take|than|them|well|were)\b/.test(
        lowerText,
      )
    ) {
      return "en"
    }

    // Français par défaut
    return "fr"
  }

  // Modération de contenu
  const moderateContent = (text: string, language: string): boolean => {
    const inappropriateWords = INAPPROPRIATE_WORDS[language as keyof typeof INAPPROPRIATE_WORDS] || []
    return inappropriateWords.some((word) => text.toLowerCase().includes(word.toLowerCase()))
  }

  // Génération de réponse avec RAG
  const generateResponse = (query: string, language: "fr" | "en" | "ar" | "darija"): string => {
    const lowerQuery = query.toLowerCase()

    // Vérification de modération
    if (moderateContent(query, language)) {
      const moderationResponses = MULTILINGUAL_RESPONSES.moderation[language]
      return moderationResponses[Math.floor(Math.random() * moderationResponses.length)]
    }

    // Recherche dans les documents RAG
    const relevantDocs = ragSystem.current.searchRelevantDocs(query)
    let contextInfo = ""

    if (relevantDocs.length > 0) {
      contextInfo = `\n\n📄 **Basé sur tes documents :**\n${relevantDocs
        .map((doc) => `• ${doc.name}: ${doc.content.substring(0, 100)}...`)
        .join("\n")}`
    }

    // Salutations
    if (/bonjour|salut|hello|hi|salam|ahlan|marhaba/.test(lowerQuery)) {
      const greetings = MULTILINGUAL_RESPONSES.greetings[language]
      return greetings[Math.floor(Math.random() * greetings.length)] + contextInfo
    }

    // Aide
    if (/aide|help|mosa3ada|3awn/.test(lowerQuery)) {
      return MULTILINGUAL_RESPONSES.help[language][0] + contextInfo
    }

    // Réponses spécialisées par langue
    const responses = {
      fr: generateFrenchResponse(lowerQuery, contextInfo),
      en: generateEnglishResponse(lowerQuery, contextInfo),
      ar: generateArabicResponse(lowerQuery, contextInfo),
      darija: generateDarijaResponse(lowerQuery, contextInfo),
    }

    return responses[language]
  }

  const generateFrenchResponse = (query: string, context: string): string => {
    if (/prévision|forecast|demande/.test(query)) {
      return `📊 **Prévisions intelligentes :**

Pour des prévisions précises, je recommande :
• **Analyse historique** sur 12-24 mois minimum
• **Segmentation ABC** de tes produits  
• **Facteurs externes** (saisonnalité, promotions)
• **Machine Learning** pour les patterns complexes

💡 **Astuce pro :** Combine plusieurs méthodes (moyenne mobile + régression + IA) pour plus de robustesse !${context}`
    }

    if (/stock|inventaire|inventory/.test(query)) {
      return `📦 **Optimisation stock avancée :**

Mes algorithmes calculent :
• **Stock de sécurité** = √(délai × variance demande)
• **Point de commande** optimal avec incertitude
• **Coûts cachés** (obsolescence, opportunité)
• **Rotation ABC** par catégorie

🎯 **Résultat :** -20% de stock, +15% de service client !${context}`
    }

    return `🤔 Question intéressante ! Basé sur mon analyse :

Je peux t'aider avec des stratégies personnalisées pour ton contexte. Mes spécialités :
• Supply chain resiliente 
• Optimisation multi-critères
• IA prédictive avancée
• Automatisation intelligente

💬 **Précise ta question** pour une réponse sur-mesure !${context}`
  }

  const generateEnglishResponse = (query: string, context: string): string => {
    if (/forecast|prediction|demand/.test(query)) {
      return `📊 **Smart Forecasting:**

For accurate predictions, I recommend:
• **Historical analysis** over 12-24 months minimum
• **ABC segmentation** of your products  
• **External factors** (seasonality, promotions)
• **Machine Learning** for complex patterns

💡 **Pro tip:** Combine multiple methods (moving average + regression + AI) for robustness!${context}`
    }

    return `🤔 Interesting question! Based on my analysis:

I can help with personalized strategies for your context. My specialties:
• Resilient supply chain 
• Multi-criteria optimization
• Advanced predictive AI
• Intelligent automation

💬 **Be more specific** for a tailored answer!${context}`
  }

  const generateArabicResponse = (query: string, context: string): string => {
    if (/تنبؤ|توقع|طلب/.test(query)) {
      return `📊 **التنبؤ الذكي:**

للتنبؤات الدقيقة، أنصح بـ:
• **تحليل تاريخي** لمدة 12-24 شهر كحد أدنى
• **تصنيف ABC** لمنتجاتك  
• **عوامل خارجية** (الموسمية، العروض)
• **تعلم الآلة** للأنماط المعقدة

💡 **نصيحة محترف:** ادمج عدة طرق للحصول على نتائج قوية!${context}`
    }

    return `🤔 سؤال مثير للاهتمام! بناءً على تحليلي:

يمكنني المساعدة بإستراتيجيات مخصصة لسياقك. تخصصاتي:
• سلسلة توريد مرنة 
• تحسين متعدد المعايير
• ذكاء اصطناعي تنبؤي متقدم
• أتمتة ذكية

💬 **حدد سؤالك أكثر** للحصول على إجابة مفصلة!${context}`
  }

  const generateDarijaResponse = (query: string, context: string): string => {
    if (/tawaqo3|stock|khadma/.test(query)) {
      return `📊 **Tawaqo3 dkiya:**

Bach tkoun tawaqo3at dyalek m9ada:
• **T7lil tarikh** 12-24 shahar 3la l9all
• **Taqsim ABC** dyal products dyalek  
• **3awamil kharijiya** (mawasim, promotions)
• **Machine Learning** lil patterns m3aqada

💡 **Nasi7a pro:** Khallat barcha turuq bach tkoun qawiya!${context}`
    }

    return `🤔 Su2al muhimm! 7asab ta7lili:

N9der n3awnek b strategies makhsusa l 7altek. Takhasusati:
• Supply chain qawiya 
• Ta7sin multi-criteria
• AI tanabo2i mutaqadim
• Automation dkiya

💬 **Wad7 su2alek aktar** bach n3tik jawab mfasal!${context}`
  }

  // Gestion de l'audio - Speech to Text
  const startRecording = async () => {
    try {
      const permission = await Audio.requestPermissionsAsync()
      if (permission.status !== "granted") {
        Alert.alert("Permission requise", "Autorisation microphone nécessaire")
        return
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      })

      const { recording } = await Audio.Recording.createAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY)

      setRecording(recording)
      setIsRecording(true)
    } catch (err) {
      console.error("Erreur enregistrement:", err)
      Alert.alert("Erreur", "Impossible de démarrer l'enregistrement")
    }
  }

  const stopRecording = async () => {
    if (!recording) return

    setIsRecording(false)
    await recording.stopAndUnloadAsync()

    const uri = recording.getURI()
    setRecording(null)

    // Simulation de Speech-to-Text (dans un vrai projet, utiliser un service comme Google Speech API)
    Alert.alert("🎤 Audio reçu", "Fonction Speech-to-Text en développement. Tapez votre question pour l'instant.", [
      { text: "OK", onPress: () => setInputText("Bonjour, comment optimiser mon stock ?") },
    ])
  }

  // Text to Speech multilingue
  const speakText = async (text: string, language: string) => {
    if (!speechEnabled || Platform.OS === "web") return

    try {
      const cleanText = text
        .replace(/[🎉👋✨🔧📊📦🚚📈💡😊🎯⚡💪🤔🚀🌟🤝]/gu, "")
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/•/g, "")
        .replace(/\n+/g, ". ")
        .trim()

      const voiceSettings = {
        fr: { language: "fr-FR", pitch: 1.0, rate: 0.9 },
        en: { language: "en-US", pitch: 1.0, rate: 0.9 },
        ar: { language: "ar-SA", pitch: 1.0, rate: 0.8 },
        darija: { language: "ar-MA", pitch: 1.0, rate: 0.8 },
      }

      const settings = voiceSettings[language as keyof typeof voiceSettings] || voiceSettings.fr

      await Speech.speak(cleanText, settings)
    } catch (error) {
      console.warn("Speech error:", error)
    }
  }

  // Import de documents
  const importDocument = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ["text/plain", "application/pdf", "text/csv"],
        copyToCacheDirectory: true,
      })

      if (!result.canceled && result.assets[0]) {
        const file = result.assets[0]

        // Simulation de lecture de fichier (dans un vrai projet, utiliser FileSystem)
        const mockContent = `Contenu simulé du fichier ${file.name}. 
        
Données supply chain importantes:
- Stock actuel: 1500 unités
- Demande moyenne: 200/jour  
- Délai livraison: 5 jours
- Coût stockage: 2€/unité/mois
- Taux service: 95%

Recommandations:
- Optimiser point de commande
- Réduire stock de sécurité
- Améliorer prévisions`

        const newDoc: Document = {
          id: Date.now().toString(),
          name: file.name,
          content: mockContent,
          type: file.mimeType || "text/plain",
          uploadDate: new Date(),
        }

        ragSystem.current.addDocument(newDoc)
        setDocuments((prev) => [...prev, newDoc])

        Alert.alert(
          "📄 Document importé !",
          `${file.name} ajouté à ma base de connaissances. Je peux maintenant répondre en me basant sur ce document !`,
        )
      }
    } catch (error) {
      console.error("Erreur import:", error)
      Alert.alert("Erreur", "Impossible d'importer le document")
    }
  }

  const handleSend = async () => {
    if (!inputText.trim() || isLoading) return

    const detectedLang = detectLanguage(inputText.trim())
    setCurrentLanguage(detectedLang)

    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputText.trim(),
      isUser: true,
      timestamp: new Date(),
      language: detectedLang,
    }

    setMessages((prev) => [...prev, userMessage])
    setInputText("")
    setIsLoading(true)

    // Simulation délai réflexion
    setTimeout(async () => {
      const response = generateResponse(inputText.trim(), detectedLang)

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response,
        isUser: false,
        timestamp: new Date(),
        language: detectedLang,
      }

      setMessages((prev) => [...prev, aiMessage])
      setIsLoading(false)

      // Lecture audio
      await speakText(response, detectedLang)
    }, 800)
  }

  const handleVoiceRecord = async () => {
    if (isRecording) {
      await stopRecording()
    } else {
      await startRecording()
    }
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
        <Text style={styles.headerTitle}>🧠 RawaJAI Advanced AI</Text>
        <Text style={styles.headerSubtitle}>Assistant multilingue avec RAG • {documents.length} documents</Text>

        <WebSafeView style={styles.controlsRow}>
          <WebSafeView style={styles.languageSelector}>
            {(["fr", "en", "ar", "darija"] as const).map((lang) => (
              <WebSafeTouchableOpacity
                key={lang}
                style={[styles.langButton, currentLanguage === lang && styles.langButtonActive]}
                onPress={() => setCurrentLanguage(lang)}
              >
                <Text style={[styles.langButtonText, currentLanguage === lang && styles.langButtonTextActive]}>
                  {lang.toUpperCase()}
                </Text>
              </WebSafeTouchableOpacity>
            ))}
          </WebSafeView>

          <WebSafeTouchableOpacity
            style={[styles.audioButton, speechEnabled && styles.audioButtonActive]}
            onPress={() => setSpeechEnabled(!speechEnabled)}
          >
            <Text style={styles.audioButtonText}>{speechEnabled ? "🔊" : "🔇"}</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity style={styles.docButton} onPress={importDocument}>
            <Text style={styles.docButtonText}>📄</Text>
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
            <WebSafeView style={styles.messageFooter}>
              <Text style={styles.timestamp}>
                {message.timestamp.toLocaleTimeString("fr-FR", { timeStyle: "short" })}
              </Text>
              <Text style={styles.languageTag}>{message.language.toUpperCase()}</Text>
            </WebSafeView>
          </WebSafeView>
        ))}

        {isLoading && (
          <WebSafeView style={[styles.messageContainer, styles.aiMessage]}>
            <Text style={[styles.messageText, styles.aiMessageText]}>🧠 Analyse en cours avec IA + RAG...</Text>
          </WebSafeView>
        )}
      </ScrollView>

      <WebSafeView style={styles.inputContainer}>
        <WebSafeView style={styles.inputRow}>
          <TextInput
            style={styles.textInput}
            value={inputText}
            onChangeText={setInputText}
            placeholder={
              currentLanguage === "fr"
                ? "Tapez votre question..."
                : currentLanguage === "en"
                  ? "Type your question..."
                  : currentLanguage === "ar"
                    ? "اكتب سؤالك..."
                    : "Kteb su2alek..."
            }
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
          <WebSafeTouchableOpacity
            style={styles.quickButton}
            onPress={() =>
              setInputText(
                currentLanguage === "fr"
                  ? "aide"
                  : currentLanguage === "en"
                    ? "help"
                    : currentLanguage === "ar"
                      ? "مساعدة"
                      : "3awn",
              )
            }
          >
            <Text style={styles.quickButtonText}>
              💡{" "}
              {currentLanguage === "fr"
                ? "Aide"
                : currentLanguage === "en"
                  ? "Help"
                  : currentLanguage === "ar"
                    ? "مساعدة"
                    : "3awn"}
            </Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity
            style={styles.quickButton}
            onPress={() =>
              setInputText(
                currentLanguage === "fr"
                  ? "prévisions"
                  : currentLanguage === "en"
                    ? "forecasting"
                    : currentLanguage === "ar"
                      ? "تنبؤات"
                      : "tawaqo3",
              )
            }
          >
            <Text style={styles.quickButtonText}>
              📊{" "}
              {currentLanguage === "fr"
                ? "Prévisions"
                : currentLanguage === "en"
                  ? "Forecast"
                  : currentLanguage === "ar"
                    ? "تنبؤ"
                    : "Tawaqo3"}
            </Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity
            style={styles.quickButton}
            onPress={() =>
              setInputText(
                currentLanguage === "fr"
                  ? "optimiser stock"
                  : currentLanguage === "en"
                    ? "optimize inventory"
                    : currentLanguage === "ar"
                      ? "تحسين المخزون"
                      : "7assan stock",
              )
            }
          >
            <Text style={styles.quickButtonText}>
              📦{" "}
              {currentLanguage === "fr"
                ? "Stock"
                : currentLanguage === "en"
                  ? "Inventory"
                  : currentLanguage === "ar"
                    ? "مخزون"
                    : "Stock"}
            </Text>
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
  controlsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 12,
  },
  languageSelector: {
    flexDirection: "row",
    gap: 4,
  },
  langButton: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    backgroundColor: "#f1f5f9",
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  langButtonActive: {
    backgroundColor: "#3B82F6",
    borderColor: "#3B82F6",
  },
  langButtonText: {
    fontSize: 10,
    color: "#64748b",
    fontWeight: "600",
  },
  langButtonTextActive: {
    color: "white",
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
  docButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: "#f1f5f9",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 1,
    borderColor: "#e2e8f0",
  },
  docButtonText: {
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
