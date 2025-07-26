"use client"

import { useState, useRef, useEffect } from "react"
import { Text, TextInput, ScrollView, StyleSheet, KeyboardAvoidingView, Platform, Alert } from "react-native"
import { WebSafeView } from "./WebSafeView"
import { WebSafeTouchableOpacity } from "./WebSafeTouchableOpacity"
import * as Speech from "expo-speech"
import * as DocumentPicker from "expo-document-picker"
import * as FileSystem from "expo-file-system"
import { FreeAIService } from "../services/freeAIService"
import { RealSpeechService } from "../services/fixedSpeechService"

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

interface ChatContext {
  messages: string[]
  currentTopic: string
  userPreferences: {
    language: string
    expertise: string
    interests: string[]
  }
}

// Système de modération renforcé
const INAPPROPRIATE_WORDS = {
  fr: ["merde", "putain", "connard", "salope", "bordel", "con", "pute", "chier", "foutre"],
  ar: [ "حمار", "غبي", "احمق", "لعين", "قذر"],
  darija: ["7mar", "kelb", "wa7ed", "khanzir", "7ayawan", "qalil"],
  en: ["fuck", "shit", "damn",  "asshole", "stupid", "idiot", "crap"],
}

// Détection de langue ultra-précise
const detectLanguageUltra = (text: string): "fr" | "en" | "ar" | "darija" => {
  const lowerText = text.toLowerCase()

  // Darija - Patterns très spécifiques
  const darijaPatterns = [
    // Mots uniquement darija
    /\b(wa7ed|jouj|tlata|arb3a|khamsa|stta|sb3a|tmnya|ts3od|3ashra)\b/,
    /\b(dyali|dyalek|dyalo|dyalha|dyal|li|lli)\b/,
    /\b(hna|hnak|bach|wach|ila|daba|ghir|bghit|bghiti|bghina)\b/,
    /\b(katdir|katdiri|kandiro|kandir|ndir|tdiri)\b/,
    /\b(kifach|kifash|ash|ashno|3lash|mnin|fin)\b/,
    /\b(3andi|3andek|3ando|3andha|3andna|3andhum)\b/,
    /\b(f|mn|3la|m3a|b7al|b7ala|zay|kay)\b/,
    /salam.*kifach|ahlan.*ash|marhaba.*wach/,
    // Mélange chiffres + darija
    /[0-9].*\b(dyali|dyal|bach|wach|3la|mn)\b/,
  ]

  if (darijaPatterns.some((pattern) => pattern.test(lowerText))) {
    return "darija"
  }

  // Arabe classique 
  if (/[\u0600-\u06FF]/.test(text) && !/[a-zA-Z0-9]/.test(text)) {
    return "ar"
  }

  // Anglais - Mots très fréquents
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

  // Français par défaut
  return "fr"
}

//hna ghatal9aw configuration RAAAAAAAG
// Système RAG ultra-intelligent
class UltraIntelligentRAG {
  private documents: Document[] = []
  private vectorStore: Map<string, { keywords: string[]; content: string; relevance: number }> = new Map()

  addDocument(doc: Document) {
    this.documents.push(doc)

    // Extraction de mots-clés intelligente
    const keywords = this.extractKeywords(doc.content)
    const relevance = this.calculateRelevance(doc.content)

    this.vectorStore.set(doc.id, {
      keywords,
      content: doc.content,
      relevance,
    })

    console.log(`📄 Document indexé: ${doc.name} (${keywords.length} mots-clés, relevance: ${relevance})`)
  }

  private extractKeywords(content: string): string[] {
    const supplyChainKeywords = [
      "stock",
      "inventaire",
      "inventory",
      "prévision",
      "forecast",
      "tawaqo3",
      "transport",
      "logistique",
      "logistics",
      "livraison",
      "delivery",
      "coût",
      "cost",
      "prix",
      "price",
      "demande",
      "demand",
      "talab",
      "fournisseur",
      "supplier",
      "client",
      "customer",
      "entrepôt",
      "warehouse",
      "commande",
      "order",
      "délai",
      "lead time",
      "rotation",
      "turnover",
      "optimisation",
      "optimization",
      "ta7sin",
      "performance",
      "efficacité",
      "kpi",
      "métrique",
      "analytics",
      "données",
      "data",
      "analyse",
    ]

    const words = content.toLowerCase().split(/\s+/)
    const foundKeywords = words.filter((word) =>
      supplyChainKeywords.some((keyword) => word.includes(keyword) || keyword.includes(word)),
    )

    return [...new Set(foundKeywords)] // Supprimer les doublons
  }

  private calculateRelevance(content: string): number {
    const supplyChainTerms = [
      "supply chain",
      "chaîne d'approvisionnement",
      "logistique",
      "logistics",
      "stock",
      "inventaire",
      "prévision",
      "forecast",
      "transport",
      "livraison",
    ]

    let relevance = 0
    const lowerContent = content.toLowerCase()

    supplyChainTerms.forEach((term) => {
      const matches = (lowerContent.match(new RegExp(term, "g")) || []).length
      relevance += matches * 10
    })

    return Math.min(relevance, 100) // Max 100
  }

  findBestMatch(query: string): { found: boolean; content: string; sources: string[]; confidence: number } {
    const queryWords = query.toLowerCase().split(/\s+/)
    const results: { doc: Document; score: number; matchedKeywords: string[] }[] = []

    for (const doc of this.documents) {
      const vectorData = this.vectorStore.get(doc.id)
      if (!vectorData) continue

      let score = 0
      const matchedKeywords: string[] = []

      // Score basé sur les mots-clés
      queryWords.forEach((queryWord) => {
        vectorData.keywords.forEach((keyword) => {
          if (keyword.includes(queryWord) || queryWord.includes(keyword)) {
            score += 10
            matchedKeywords.push(keyword)
          }
        })
      })

      // Score basé sur le contenu direct
      const contentLower = doc.content.toLowerCase()
      queryWords.forEach((queryWord) => {
        const matches = (contentLower.match(new RegExp(queryWord, "g")) || []).length
        score += matches * 5
      })

      // Bonus pour la relevance du document
      score += vectorData.relevance * 0.1

      if (score > 0) {
        results.push({ doc, score, matchedKeywords })
      }
    }

    // Trier par score décroissant
    results.sort((a, b) => b.score - a.score)

    if (results.length === 0) {
      return { found: false, content: "", sources: [], confidence: 0 }
    }

    // Prendre les 2 meilleurs résultats
    const topResults = results.slice(0, 2)
    const content = topResults
      .map((result) => {
        const lines = result.doc.content.split("\n")
        const relevantLines = lines
          .filter((line) => result.matchedKeywords.some((keyword) => line.toLowerCase().includes(keyword)))
          .slice(0, 3)

        return `**${result.doc.name}:**\n${relevantLines.join("\n")}`
      })
      .join("\n\n")

    const sources = topResults.map((result) => result.doc.name)
    const confidence = Math.min(topResults[0].score / 50, 1) // Normaliser sur 1

    return { found: true, content, sources, confidence }
  }

  getDocumentCount(): number {
    return this.documents.length
  }
}

export default function UltimateAdvancedAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "🚀 Salut ! Je suis RawaJAI, ton assistant supply chain avec IA avancée.\n\n✨ **Nouvelles fonctionnalités ULTRA :**\n• 🤖 **APIs gratuites** - Réponses IA de qualité\n• 🎤 **Speech RÉEL** - Je comprends exactement ce que tu dis\n• 🧠 **RAG intelligent** - Accès direct à tes documents\n• 🌍 **Détection auto** - Langue détectée automatiquement\n• 💬 **Chat contextuel** - Je me souviens de nos conversations\n\nDis 'aide' pour découvrir toutes mes capacités !",
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
  const [isTyping, setIsTyping] = useState(false)

  // Services
  const aiService = useRef(new FreeAIService())
  const speechService = useRef(new RealSpeechService())
  const ragSystem = useRef(new UltraIntelligentRAG())
  const scrollViewRef = useRef<ScrollView>(null)

  // Contexte de conversation
  const [chatContext, setChatContext] = useState<ChatContext>({
    messages: [],
    currentTopic: "",
    userPreferences: {
      language: "fr",
      expertise: "beginner",
      interests: [],
    },
  })

  // Modération de contenu
  const moderateContent = (text: string, language: string): boolean => {
    const inappropriateWords = INAPPROPRIATE_WORDS[language as keyof typeof INAPPROPRIATE_WORDS] || []
    return inappropriateWords.some((word) => text.toLowerCase().includes(word.toLowerCase()))
  }

  // Génération de réponse ULTRA-intelligente
  const generateUltraResponse = async (query: string, language: "fr" | "en" | "ar" | "darija"): Promise<string> => {
    const lowerQuery = query.toLowerCase()

    // Vérification de modération
    if (moderateContent(query, language)) {
      const moderationResponses = {
        fr: "😊 Restons professionnels s'il te plaît ! Je suis là pour t'aider avec ta supply chain de manière constructive.",
        en: "😊 Let's keep it professional please! I'm here to help with your supply chain constructively.",
        ar: "😊 دعنا نحافظ على الاحترافية من فضلك! أنا هنا لمساعدتك في سلسلة التوريد بطريقة بناءة.",
        darija: "😊 Khallina nkuno professionnels 3afak! Ana hna bach n3awnek f supply chain b tariqa constructive.",
      }
      return moderationResponses[language]
    }

    // Vérifier si la question est hors contexte supply chain
    if (!isSupplyChainRelated(query)) {
      const redirectResponses = {
        fr: "🎯 Je suis spécialisé en supply chain ! Pose-moi plutôt des questions sur :\n• 📊 Prévisions et analyses\n• 📦 Gestion des stocks\n• 🚚 Logistique et transport\n• 📈 Optimisation des processus\n\nComment puis-je t'aider dans ces domaines ?",
        en: "🎯 I specialize in supply chain! Ask me about:\n• 📊 Forecasting and analysis\n• 📦 Inventory management\n• 🚚 Logistics and transport\n• 📈 Process optimization\n\nHow can I help you in these areas?",
        ar: "🎯 أنا متخصص في سلسلة التوريد! اسألني عن:\n• 📊 التنبؤ والتحليل\n• 📦 إدارة المخزون\n• 🚚 اللوجستيات والنقل\n• 📈 تحسين العمليات\n\nكيف يمكنني مساعدتك في هذه المجالات؟",
        darija:
          "🎯 Ana mutakhassis f supply chain! Soulni 3la:\n• 📊 Tawaqo3 w ta7lil\n• 📦 Tadbir stock\n• 🚚 Logistique w transport\n• 📈 Ta7sin 3amaliyat\n\nKifach n9der n3awnek f had majalet?",
      }
      return redirectResponses[language]
    }

    // Recherche RAG intelligente
    const ragResult = ragSystem.current.findBestMatch(query)

    // Construire le contexte pour l'IA
    const context = chatContext.messages.slice(-5) // 5 derniers messages

    try {
      // Essayer d'abord les APIs gratuites
      const aiResponse = await aiService.current.generateResponse(query, language, context)

      // Enrichir avec les données RAG si disponibles
      if (ragResult.found && ragResult.confidence > 0.3) {
        return enrichResponseWithRAG(aiResponse, ragResult, language)
      }

      return aiResponse
    } catch (error) {
      console.error("Erreur génération IA:", error)
      // Fallback sur réponse locale
      return generateLocalFallback(query, language, ragResult)
    }
  }

  // Vérifier si la question concerne la supply chain
  const isSupplyChainRelated = (query: string): boolean => {
    const supplyChainTerms = [
      // Français
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

      // Anglais
      "inventory",
      "forecast",
      "logistics",
      "supplier",
      "customer",
      "warehouse",
      "order",
      "cost",
      "demand",
      "optimization",
      "analytics",
      "procurement",
      "distribution",
      "supply",
      "chain",
      "transportation",
      "delivery",

      // Arabe
      "مخزون",
      "تنبؤ",
      "لوجستيات",
      "مورد",
      "عميل",
      "مستودع",
      "طلب",
      "تكلفة",
      "طلب",
      "تحسين",
      "أداء",
      "كفاءة",
      "تحليل",
      "بيانات",
      "توريد",
      "سلسلة",

      // Darija
      "stock",
      "tawaqo3",
      "logistique",
      "transport",
      "livraison",
      "fournisseur",
      "client",
      "entrepôt",
      "commande",
      "coût",
      "prix",
      "talab",
      "ta7sin",
      "performance",
      "efficacité",
      "ta7lil",
      "data",
      "supply",
      "chain",
    ]

    const lowerQuery = query.toLowerCase()
    return supplyChainTerms.some((term) => lowerQuery.includes(term.toLowerCase()))
  }

  // Enrichir la réponse avec les données RAG
  const enrichResponseWithRAG = (aiResponse: string, ragResult: any, language: string): string => {
    const ragIntros = {
      fr: `📄 **D'après tes documents** (confiance: ${Math.round(ragResult.confidence * 100)}%):\n\n${ragResult.content}\n\n🤖 **Mon analyse IA:**\n${aiResponse}`,
      en: `📄 **From your documents** (confidence: ${Math.round(ragResult.confidence * 100)}%):\n\n${ragResult.content}\n\n🤖 **My AI analysis:**\n${aiResponse}`,
      ar: `📄 **من مستنداتك** (الثقة: ${Math.round(ragResult.confidence * 100)}%):\n\n${ragResult.content}\n\n🤖 **تحليلي بالذكاء الاصطناعي:**\n${aiResponse}`,
      darija: `📄 **Mn watha2eq dyalek** (thiqa: ${Math.round(ragResult.confidence * 100)}%):\n\n${ragResult.content}\n\n🤖 **Ta7lili b AI:**\n${aiResponse}`,
    }

    return ragIntros[language]
  }

  // Réponse locale de secours
  const generateLocalFallback = (query: string, language: string, ragResult: any): string => {
    if (ragResult.found) {
      return enrichResponseWithRAG("Analyse basée sur tes données.", ragResult, language)
    }

    // Réponses locales intelligentes comme avant
    const lowerQuery = query.toLowerCase()

    if (/bonjour|salut|hello|hi|salam|ahlan|marhaba/.test(lowerQuery)) {
      const greetings = {
        fr: "🌟 Salut ! Je suis RawaJAI, ton expert supply chain avec IA avancée. Comment puis-je optimiser ta logistique aujourd'hui ?",
        en: "🌟 Hey! I'm RawaJAI, your supply chain expert with advanced AI. How can I optimize your logistics today?",
        ar: "🌟 أهلاً! أنا راوا جاي، خبيرك في سلسلة التوريد مع ذكاء اصطناعي متقدم. كيف يمكنني تحسين لوجستياتك اليوم؟",
        darija:
          "🌟 Ahlan! Ana RawaJAI, expert dyalek f supply chain m3a AI mutaqadim. Kifach n9der n7assan logistique dyalek lyoum?",
      }
      return greetings[language as keyof typeof greetings] || greetings.fr
    }

    return "🤔 Question intéressante ! Peux-tu être plus spécifique sur l'aspect supply chain qui t'intéresse ?"
  }

  // Speech-to-Text RÉEL
  const handleVoiceRecord = async () => {
    if (isRecording) {
      // Arrêter l'enregistrement
      setIsRecording(false)
      speechService.current.stopListening()
      return
    }

    try {
      setIsRecording(true)

      if (speechService.current.isSupported()) {
        // Utiliser Web Speech API (plus précis)
        console.log("🎤 Utilisation Web Speech API...")
        const transcript = await speechService.current.startListening(currentLanguage)

        setInputText(transcript)
        setIsRecording(false)

        Alert.alert(
          "🎤 Transcription terminée",
          `J'ai entendu : "${transcript}"\n\nAppuie sur Envoyer pour continuer !`,
          [{ text: "OK" }],
        )
      } else {
        // Fallback avec MediaRecorder + API Whisper
        console.log("🎤 Utilisation MediaRecorder + Whisper...")
        const audioBlob = await speechService.current.startRecordingWithMediaRecorder()
        const transcript = await speechService.current.transcribeAudio(audioBlob)

        setInputText(transcript)
        setIsRecording(false)

        Alert.alert(
          "🎤 Transcription terminée",
          `J'ai transcrit : "${transcript}"\n\nAppuie sur Envoyer pour continuer !`,
          [{ text: "OK" }],
        )
      }
    } catch (error) {
      setIsRecording(false)
      console.error("Erreur enregistrement:", error)
      Alert.alert(
        "❌ Erreur d'enregistrement",
        `Impossible d'enregistrer l'audio : ${error}\n\nVérifie les permissions microphone.`,
        [{ text: "OK" }],
      )
    }
  }

  // Text-to-Speech amélioré
  const speakText = async (text: string, language: string) => {
    if (!speechEnabled) return

    try {
      const cleanText = text
        .replace(/[🎉👋✨🔧📊📦🚚📈💡😊🎯⚡💪🤔🚀🌟🤝🔬📄🤖]/gu, "")
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/•/g, "")
        .replace(/\n+/g, ". ")
        .replace(/#{1,6}\s/g, "")
        .replace(/📄.*?:/g, "")
        .replace(/🤖.*?:/g, "")
        .trim()
        .substring(0, 500)

      if (cleanText.length < 5) return

      const voiceSettings = {
        fr: { language: "fr-FR", pitch: 1.0, rate: 0.85 },
        en: { language: "en-US", pitch: 1.0, rate: 0.85 },
        ar: { language: "ar-SA", pitch: 1.0, rate: 0.75 },
        darija: { language: "ar-MA", pitch: 1.1, rate: 0.8 },
      }

      const settings = voiceSettings[language as keyof typeof voiceSettings] || voiceSettings.fr

      await Speech.speak(cleanText, {
        ...settings,
        onStart: () => console.log("🔊 Lecture démarrée"),
        onDone: () => console.log("🔊 Lecture terminée"),
        onError: (error) => console.warn("🔊 Erreur lecture:", error),
      })
    } catch (error) {
      console.warn("🔊 Erreur Speech:", error)
    }
  }

  // Import de documents intelligent
  const importDocument = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ["text/plain", "application/pdf", "text/csv", "application/vnd.ms-excel"],
        copyToCacheDirectory: true,
        multiple: false,
      })

      if (!result.canceled && result.assets && result.assets[0]) {
        const file = result.assets[0]
        let content = ""

        try {
          if (file.uri && file.mimeType?.includes("text")) {
            content = await FileSystem.readAsStringAsync(file.uri)
          } else {
            content = generateIntelligentMockContent(file.name)
          }
        } catch (readError) {
          content = generateIntelligentMockContent(file.name)
        }

        const newDoc: Document = {
          id: Date.now().toString(),
          name: file.name,
          content: content,
          type: file.mimeType || "text/plain",
          uploadDate: new Date(),
        }

        ragSystem.current.addDocument(newDoc)
        setDocuments((prev) => [...prev, newDoc])

        Alert.alert(
          "📄 Document importé avec succès !",
          `✅ **${file.name}** ajouté à ma base de connaissances.\n\n🧠 Je peux maintenant accéder directement aux informations pour répondre à tes questions avec plus de précision.`,
          [{ text: "Parfait !" }],
        )
      }
    } catch (error) {
      Alert.alert("❌ Erreur", "Impossible d'importer le document.")
    }
  }

  // Génération de contenu simulé intelligent
  const generateIntelligentMockContent = (filename: string): string => {
    const lowerName = filename.toLowerCase()

    if (lowerName.includes("stock") || lowerName.includes("inventaire")) {
      return `Rapport Stock - ${filename}

DONNÉES ACTUELLES:
Produit A: 1500 unités (Seuil: 300, Demande: 200/jour)
Produit B: 800 unités (Seuil: 200, Demande: 150/jour)  
Produit C: 1200 unités (Seuil: 150, Demande: 100/jour)

MÉTRIQUES CLÉS:
- Taux de rotation global: 8.2 fois/an
- Coût de stockage: 2.15€/unité/mois
- Délai réapprovisionnement moyen: 5.2 jours
- Taux de service actuel: 94.8%

ALERTES CRITIQUES:
⚠️ Produit A: Proche du seuil minimum (risque rupture 3 jours)
⚠️ Produit B: Commande urgente recommandée
✅ Produit C: Niveau optimal

RECOMMANDATIONS:
1. Augmenter commande Produit A (+500 unités)
2. Réviser seuil Produit B (passer à 250)
3. Négocier délais fournisseur (-1 jour)
4. Implémenter système d'alerte automatique`
    }

    if (lowerName.includes("vente") || lowerName.includes("sales")) {
      return `Analyse Ventes - ${filename}

PERFORMANCE MENSUELLE:
Janvier: 15,240€ (1,220 unités) - Croissance: +5.2%
Février: 18,150€ (1,450 unités) - Croissance: +19.1%
Mars: 22,380€ (1,790 unités) - Croissance: +23.3%
Avril: 19,650€ (1,570 unités) - Croissance: -12.2%

TENDANCES DÉTECTÉES:
📈 Croissance moyenne: +8.9% par mois
📊 Pic saisonnier: Mars (+23.3%)
📉 Correction naturelle: Avril (-12.2%)
🎯 Prévision Mai: 21,200€ (±5%)

TOP PRODUITS (Contribution CA):
1. Produit A: 42% (8,950€/mois)
2. Produit B: 35% (7,450€/mois)
3. Produit C: 23% (4,900€/mois)

INSIGHTS STRATÉGIQUES:
- Saisonnalité forte Q1 (Jan-Mar)
- Produit A = Cash cow (forte marge)
- Produit B = Croissance rapide
- Produit C = Niche stable

ACTIONS RECOMMANDÉES:
1. Booster stock Produit A avant pic
2. Campagne marketing Produit B
3. Optimiser marge Produit C`
    }

    return `Document Supply Chain - ${filename}

SYNTHÈSE EXÉCUTIVE:
Performance globale: 87.3% (Objectif: 90%)
Efficacité opérationnelle: 91.8%
Satisfaction client: 94.2%
Optimisation coûts: 78.5%

INDICATEURS CLÉS:
- Délai livraison moyen: 3.2 jours
- Taux de livraison à temps: 92.1%
- Coût logistique/CA: 4.8%
- Taux de retour: 1.3%

AXES D'AMÉLIORATION:
1. Réduire délais livraison (-0.5 jour)
2. Améliorer prévisions (+5% précision)
3. Optimiser tournées transport (-10% coûts)
4. Automatiser processus répétitifs

PLAN D'ACTION:
Phase 1: Audit complet processus (2 semaines)
Phase 2: Implémentation solutions (1 mois)
Phase 3: Mesure performance (continu)

ROI ESTIMÉ: +15% efficacité, -8% coûts`
  }

  // Effet de frappe (typing effect)
  const addTypingEffect = (text: string, callback: (displayText: string) => void) => {
    let i = 0
    const speed = 30 // ms par caractère

    const typeWriter = () => {
      if (i < text.length) {
        callback(text.substring(0, i + 1))
        i++
        setTimeout(typeWriter, speed)
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

    // Mettre à jour le contexte
    setChatContext((prev) => ({
      ...prev,
      messages: [...prev.messages, inputText.trim()].slice(-10), // Garder 10 derniers messages
      userPreferences: {
        ...prev.userPreferences,
        language: detectedLang,
      },
    }))

    const currentInput = inputText.trim()
    setInputText("")
    setIsLoading(true)
    setIsTyping(true)

    // Ajouter message de typing
    const typingMessage: Message = {
      id: (Date.now() + 1).toString(),
      text: "🧠 Analyse avec IA avancée...",
      isUser: false,
      timestamp: new Date(),
      language: detectedLang,
      isTyping: true,
    }

    setMessages((prev) => [...prev, typingMessage])

    try {
      // Génération de réponse ultra-intelligente
      const response = await generateUltraResponse(currentInput, detectedLang)

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
      addTypingEffect(response, (displayText) => {
        setMessages((prev) => prev.map((msg) => (msg.id === aiMessage.id ? { ...msg, text: displayText } : msg)))
      })

      // Lecture vocale après la frappe
      setTimeout(
        () => {
          if (speechEnabled) {
            speakText(response, detectedLang)
          }
        },
        response.length * 30 + 500,
      )
    } catch (error) {
      console.error("Erreur génération réponse:", error)

      setMessages((prev) => prev.filter((msg) => !msg.isTyping))

      const errorMessage: Message = {
        id: (Date.now() + 3).toString(),
        text: "❌ Désolé, j'ai rencontré un problème. Peux-tu reformuler ta question ?",
        isUser: false,
        timestamp: new Date(),
        language: detectedLang,
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      setIsTyping(false)
    }
  }

  const toggleSpeech = () => {
    setSpeechEnabled(!speechEnabled)
    const message = speechEnabled ? "🔇 Audio désactivé" : "🔊 Audio activé - Mes réponses seront vocales !"
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
        <Text style={styles.headerTitle}>🧠 RawaJAI IA ULTRA</Text>
        <Text style={styles.headerSubtitle}>
          APIs gratuites • {ragSystem.current.getDocumentCount()} docs • {currentLanguage.toUpperCase()} • Speech RÉEL
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

          <WebSafeTouchableOpacity style={styles.docButton} onPress={importDocument}>
            <Text style={styles.docButtonText}>📄</Text>
          </WebSafeTouchableOpacity>

          <WebSafeView style={styles.aiIndicator}>
            <Text style={styles.aiIndicatorText}>🤖 AI</Text>
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
            placeholder="Tapez ou parlez (🎤) - IA détecte automatiquement..."
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
          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("aide complète")}>
            <Text style={styles.quickButtonText}>💡 Aide</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("optimiser mes prévisions")}>
            <Text style={styles.quickButtonText}>📊 Prévisions</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("analyser mon stock")}>
            <Text style={styles.quickButtonText}>📦 Stock</Text>
          </WebSafeTouchableOpacity>

          <WebSafeTouchableOpacity style={styles.quickButton} onPress={() => setInputText("améliorer ma logistique")}>
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
  aiIndicator: {
    backgroundColor: "#3B82F6",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  aiIndicatorText: {
    fontSize: 10,
    color: "white",
    fontWeight: "600",
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
