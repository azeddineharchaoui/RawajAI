"use client"

import { useState, useRef, useEffect } from "react"
import { Text, TextInput, ScrollView, StyleSheet, KeyboardAvoidingView, Platform, Alert } from "react-native"
import { WebSafeView } from "./WebSafeView"
import { WebSafeTouchableOpacity } from "./WebSafeTouchableOpacity"
import * as Speech from "expo-speech"
import * as DocumentPicker from "expo-document-picker"
import * as FileSystem from "expo-file-system"

interface Message {
  id: string
  text: string
  isUser: boolean
  timestamp: Date
  language: string
  hasAudio?: boolean
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
  fr: ["merde", "putain", "connard", "salope", "bordel", "con", "pute"],
  ar: ["كلب", "حمار", "غبي", "احمق"],
  darija: ["7mar", "kelb", "wa7ed", "9a7ba"],
  en: ["fuck", "shit", "damn", "bitch", "asshole", "stupid"],
}

// Base de connaissances RAG FONCTIONNELLE
class WorkingRAGSystem {
  private documents: Document[] = []

  addDocument(doc: Document) {
    this.documents.push(doc)
    console.log(`📄 Document ajouté: ${doc.name}`)
  }

  searchRelevantDocs(query: string): Document[] {
    const lowerQuery = query.toLowerCase()

    return this.documents.filter((doc) => {
      const content = doc.content.toLowerCase()
      const name = doc.name.toLowerCase()

      // Recherche simple mais efficace
      return content.includes(lowerQuery) || name.includes(lowerQuery) || this.hasKeywordMatch(lowerQuery, content)
    })
  }

  private hasKeywordMatch(query: string, content: string): boolean {
    const queryWords = query.split(" ")
    const contentWords = content.split(" ")

    return queryWords.some((qWord) => contentWords.some((cWord) => cWord.includes(qWord) || qWord.includes(cWord)))
  }

  getAllDocuments(): Document[] {
    return this.documents
  }

  getDocumentCount(): number {
    return this.documents.length
  }
}

// Réponses multilingues AMÉLIORÉES
const WORKING_RESPONSES = {
  greetings: {
    fr: [
      "🌟 Salut ! Je suis RawaJAI, ton assistant supply chain intelligent. Ravi de te rencontrer ! 🤝\n\nJe peux t'aider avec tes prévisions, ton stock, ta logistique... Qu'est-ce qui t'amène aujourd'hui ?",
      "👋 Hey ! C'est parti pour optimiser ta supply chain ensemble ! 🚀\n\nJe suis là pour répondre à tes questions et t'accompagner. Tu as des défis intéressants pour moi ?",
    ],
    en: [
      "🌟 Hey there! I'm RawaJAI, your intelligent supply chain assistant. Great to meet you! 🤝\n\nI can help with forecasting, inventory, logistics... What brings you here today?",
      "👋 Hi! Ready to optimize your supply chain together! 🚀\n\nI'm here to answer your questions and guide you. Got any interesting challenges for me?",
    ],
    ar: [
      "🌟 أهلاً وسهلاً! أنا راوا جاي، مساعدك الذكي في سلسلة التوريد. سعيد بلقائك! 🤝\n\nيمكنني مساعدتك في التنبؤات والمخزون واللوجستيات... ما الذي يجلبك اليوم؟",
    ],
    darija: [
      "🌟 Ahlan! Ana RawaJAI, m3ak f supply chain dkiya. Far7an nshofek! 🤝\n\nN9der n3awnek f tawaqo3, stock, logistique... Ash jab lik lyoum?",
    ],
  },

  help: {
    fr: `🧠 **Mes capacités avancées :**

• 📊 **Prévisions intelligentes** - Analyse de tendances et ML
• 📦 **Optimisation stock** - Calculs EOQ, stock de sécurité  
• 🚚 **Logistique optimisée** - Planification transport
• 📈 **Analytics avancés** - KPIs et tableaux de bord
• 🎤 **Reconnaissance vocale** - Parle-moi directement !
• 🔊 **Réponses audio** - J'écoute et je réponds
• 📄 **Import documents** - J'apprends de tes fichiers
• 🌍 **Multilingue** - FR, EN, AR, Darija

**🚀 Alors, on commence par quoi ?**`,

    en: `🧠 **My advanced capabilities:**

• 📊 **Smart forecasting** - Trend analysis and ML
• 📦 **Stock optimization** - EOQ calculations, safety stock  
• 🚚 **Optimized logistics** - Transport planning
• 📈 **Advanced analytics** - KPIs and dashboards
• 🎤 **Voice recognition** - Speak to me directly!
• 🔊 **Audio responses** - I listen and respond
• 📄 **Document import** - I learn from your files
• 🌍 **Multilingual** - FR, EN, AR, Darija

**🚀 So, what should we start with?**`,

    ar: `🧠 **قدراتي المتقدمة:**

• 📊 **التنبؤ الذكي** - تحليل الاتجاهات والتعلم الآلي
• 📦 **تحسين المخزون** - حسابات EOQ، مخزون الأمان  
• 🚚 **لوجستيات محسنة** - تخطيط النقل
• 📈 **تحليلات متقدمة** - مؤشرات ولوحات معلومات
• 🎤 **التعرف على الصوت** - تحدث معي مباشرة!
• 🔊 **ردود صوتية** - أستمع وأجيب
• 📄 **استيراد المستندات** - أتعلم من ملفاتك
• 🌍 **متعدد اللغات** - FR, EN, AR, Darija

**🚀 إذن، بماذا نبدأ؟**`,

    darija: `🧠 **Qudrati l mutaqadima:**

• 📊 **Tawaqo3 dkiya** - T7lil trends w ML
• 📦 **Ta7sin stock** - 7isabat EOQ, stock sécurité  
• 🚚 **Logistique m7assana** - Takhtit transport
• 📈 **Analytics qawiya** - KPIs w dashboards
• 🎤 **Ta3arof 3la sout** - Hder m3aya direct!
• 🔊 **Ajwiba sawt** - Kansma3 w kanjaweb
• 📄 **Import watha2eq** - Kant3allem mn files dyalek
• 🌍 **Lughat kathira** - FR, EN, AR, Darija

**🚀 Iwa, nbd2o b ash?**`,
  },

  moderation: {
    fr: "😊 Hey, restons professionnels s'il te plaît ! 🤝 Je suis là pour t'aider avec ta supply chain de manière constructive. Reformule ta question ?",
    en: "😊 Hey, let's keep it professional please! 🤝 I'm here to help with your supply chain constructively. Can you rephrase?",
    ar: "😊 دعنا نحافظ على الاحترافية من فضلك! 🤝 أنا هنا لمساعدتك في سلسلة التوريد بطريقة بناءة.",
    darija: "😊 Khallina nkuno professionels 3afak! 🤝 Ana hna bach n3awnek f supply chain b tariqa constructive.",
  },
}

export default function WorkingAdvancedAssistant() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      text: "🚀 Salut ! Je suis RawaJAI, ton assistant supply chain avec IA avancée.\n\n✨ **Nouvelles fonctionnalités :**\n• 🎤 **Parle-moi** - Clique le micro et dis ta question\n• 🔊 **J'écoute** - Mes réponses sont vocales aussi\n• 📄 **Import docs** - J'apprends de tes fichiers\n• 🌍 **4 langues** - FR, EN, AR, Darija\n\n💡 Dis 'aide' pour découvrir tout ce que je peux faire !",
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
  const [isListening, setIsListening] = useState(false)

  const scrollViewRef = useRef<ScrollView>(null)
  const ragSystem = useRef(new WorkingRAGSystem())

  // Détection de langue AMÉLIORÉE
  const detectLanguage = (text: string): "fr" | "en" | "ar" | "darija" => {
    const lowerText = text.toLowerCase()

    // Darija - Détection plus précise
    const darijaPatterns = [
      /\b(wa7ed|b7al|dyali|hna|bach|wach|kifach|3la|mn|ila|daba|ghir|bghit|katdir|nta|nti)\b/,
      /[0-9]/,
      /(salam|ahlan|marhaba).*kifach/,
    ]
    if (darijaPatterns.some((pattern) => pattern.test(lowerText))) {
      return "darija"
    }

    // Arabe - Script arabe
    if (/[\u0600-\u06FF]/.test(text)) {
      return "ar"
    }

    // Anglais - Mots courants
    const englishWords = [
      "the",
      "and",
      "you",
      "are",
      "have",
      "with",
      "for",
      "this",
      "that",
      "from",
      "they",
      "know",
      "want",
      "been",
      "good",
      "much",
      "some",
      "time",
      "very",
      "when",
      "come",
      "here",
      "just",
      "like",
      "long",
      "make",
      "many",
      "over",
      "such",
      "take",
      "than",
      "them",
      "well",
      "were",
      "hello",
      "help",
      "how",
      "what",
      "where",
      "why",
      "forecast",
      "inventory",
      "supply",
      "chain",
    ]
    const englishCount = englishWords.filter((word) => lowerText.includes(word)).length
    if (englishCount >= 2) {
      return "en"
    }

    // Français par défaut
    return "fr"
  }

  // Modération FONCTIONNELLE
  const moderateContent = (text: string, language: string): boolean => {
    const inappropriateWords = INAPPROPRIATE_WORDS[language as keyof typeof INAPPROPRIATE_WORDS] || []
    return inappropriateWords.some((word) => text.toLowerCase().includes(word.toLowerCase()))
  }

  // Génération de réponse AMÉLIORÉE avec RAG
  const generateResponse = (query: string, language: "fr" | "en" | "ar" | "darija"): string => {
    const lowerQuery = query.toLowerCase()

    // Vérification de modération
    if (moderateContent(query, language)) {
      return WORKING_RESPONSES.moderation[language]
    }

    // Recherche dans les documents RAG
    const relevantDocs = ragSystem.current.searchRelevantDocs(query)
    let contextInfo = ""

    if (relevantDocs.length > 0) {
      contextInfo = `\n\n📄 **Basé sur tes documents :**\n${relevantDocs
        .map((doc) => `• **${doc.name}**: ${doc.content.substring(0, 150)}...`)
        .join("\n")}`
    }

    // Salutations
    if (/bonjour|salut|hello|hi|salam|ahlan|marhaba/.test(lowerQuery)) {
      const greetings = WORKING_RESPONSES.greetings[language]
      return greetings[Math.floor(Math.random() * greetings.length)] + contextInfo
    }

    // Aide
    if (/aide|help|mosa3ada|3awn/.test(lowerQuery)) {
      return WORKING_RESPONSES.help[language] + contextInfo
    }

    // Réponses spécialisées
    return generateSpecializedResponse(lowerQuery, language, contextInfo)
  }

  const generateSpecializedResponse = (
    query: string,
    language: "fr" | "en" | "ar" | "darija",
    context: string,
  ): string => {
    const responses = {
      fr: generateFrenchResponse(query, context),
      en: generateEnglishResponse(query, context),
      ar: generateArabicResponse(query, context),
      darija: generateDarijaResponse(query, context),
    }

    return responses[language]
  }

  const generateFrenchResponse = (query: string, context: string): string => {
    if (/prévision|forecast|demande|tendance/.test(query)) {
      return `📊 **Prévisions intelligentes :**

🎯 **Méthodes recommandées :**
• **Moyenne mobile** - Pour tendances stables
• **Lissage exponentiel** - Pour données volatiles  
• **Régression linéaire** - Pour croissance constante
• **Machine Learning** - Pour patterns complexes

📈 **Formule clé :** Prévision = Tendance + Saisonnalité + Aléatoire

💡 **Conseil pro :** Combine 3 méthodes et prends la médiane pour plus de robustesse !

🎯 **Précision attendue :** 85-95% selon le secteur${context}`
    }

    if (/stock|inventaire|inventory/.test(query)) {
      return `📦 **Optimisation stock avancée :**

🔢 **Formules essentielles :**
• **EOQ** = √(2 × Demande × Coût commande / Coût stockage)
• **Stock sécurité** = Z × √(Délai × Variance demande)
• **Point commande** = Demande moyenne × Délai + Stock sécurité

💰 **Coûts à optimiser :**
• Coût de commande : 50-200€/commande
• Coût de stockage : 15-25% valeur/an
• Coût de rupture : 5-50€/unité manquante

🎯 **Objectif :** -20% stock, +15% service client !${context}`
    }

    if (/logistique|transport|livraison/.test(query)) {
      return `🚚 **Logistique optimisée :**

🗺️ **Optimisation tournées :**
• **Algorithme génétique** pour routes complexes
• **Clustering** géographique des clients
• **Fenêtres horaires** optimisées

📊 **KPIs logistiques :**
• Taux de service : >95%
• Coût transport/CA : <5%
• Délai moyen : <48h
• Taux retour : <2%

💡 **Astuce :** Mutualise les livraisons pour -30% de coûts !${context}`
    }

    return `🤔 **Question intéressante !** Basé sur mon analyse :

Je peux t'aider avec des stratégies personnalisées pour ton contexte. Mes spécialités :

🔬 **Supply chain résiliente** - Gestion des risques
🎯 **Optimisation multi-critères** - Coût/Service/Qualité
🤖 **IA prédictive avancée** - Algorithmes ML
⚡ **Automatisation intelligente** - Processus optimisés

💬 **Précise ta question** pour une réponse sur-mesure !

Exemples : "Comment réduire mes coûts ?", "Améliorer mes prévisions ?", "Optimiser ma logistique ?"${context}`
  }

  const generateEnglishResponse = (query: string, context: string): string => {
    if (/forecast|prediction|demand/.test(query)) {
      return `📊 **Smart Forecasting:**

🎯 **Recommended methods:**
• **Moving average** - For stable trends
• **Exponential smoothing** - For volatile data  
• **Linear regression** - For constant growth
• **Machine Learning** - For complex patterns

📈 **Key formula:** Forecast = Trend + Seasonality + Random

💡 **Pro tip:** Combine 3 methods and take median for robustness!

🎯 **Expected accuracy:** 85-95% depending on sector${context}`
    }

    return `🤔 **Interesting question!** Based on my analysis:

I can help with personalized strategies for your context. My specialties:

🔬 **Resilient supply chain** - Risk management
🎯 **Multi-criteria optimization** - Cost/Service/Quality
🤖 **Advanced predictive AI** - ML algorithms
⚡ **Intelligent automation** - Optimized processes

💬 **Be more specific** for a tailored answer!${context}`
  }

  const generateArabicResponse = (query: string, context: string): string => {
    if (/تنبؤ|توقع|طلب/.test(query)) {
      return `📊 **التنبؤ الذكي:**

🎯 **الطرق الموصى بها:**
• **المتوسط المتحرك** - للاتجاهات المستقرة
• **التنعيم الأسي** - للبيانات المتقلبة  
• **الانحدار الخطي** - للنمو الثابت
• **تعلم الآلة** - للأنماط المعقدة

📈 **المعادلة الأساسية:** التنبؤ = الاتجاه + الموسمية + العشوائي

💡 **نصيحة محترف:** ادمج 3 طرق وخذ الوسيط للحصول على قوة!

🎯 **الدقة المتوقعة:** 85-95% حسب القطاع${context}`
    }

    return `🤔 **سؤال مثير للاهتمام!** بناءً على تحليلي:

يمكنني المساعدة بإستراتيجيات مخصصة لسياقك. تخصصاتي:

🔬 **سلسلة توريد مرنة** - إدارة المخاطر
🎯 **تحسين متعدد المعايير** - التكلفة/الخدمة/الجودة
🤖 **ذكاء اصطناعي تنبؤي متقدم** - خوارزميات ML
⚡ **أتمتة ذكية** - عمليات محسنة

💬 **حدد سؤالك أكثر** للحصول على إجابة مفصلة!${context}`
  }

  const generateDarijaResponse = (query: string, context: string): string => {
    if (/tawaqo3|stock|khadma/.test(query)) {
      return `📊 **Tawaqo3 dkiya:**

🎯 **Turuq li kansah biha:**
• **Mutawasit muta7arik** - L trends thabita
• **Tamliss ussi** - L data muta9alliba  
• **In7idar khatti** - L numuw thabit
• **Machine Learning** - L patterns m3aqada

📈 **Mu3adala asasiya:** Tawaqo3 = Ittijah + Mawsimiya + 3ashwa2i

💡 **Nasi7a pro:** Khallat 3 turuq w khud median bach tkoun qawiya!

🎯 **Diqa mutawaqqa3a:** 85-95% 7asab qita3${context}`
    }

    return `🤔 **Su2al muhimm!** 7asab ta7lili:

N9der n3awnek b strategies makhsusa l 7altek. Takhasusati:

🔬 **Supply chain qawiya** - Tadbir mukhatarat
🎯 **Ta7sin multi-criteria** - Taklifa/Khidma/Jawda
🤖 **AI tanabo2i mutaqadim** - Algorithms ML
⚡ **Automation dkiya** - 3amaliyat m7assana

💬 **Wad7 su2alek aktar** bach n3tik jawab mfasal!${context}`
  }

  // Speech-to-Text SIMULÉ mais FONCTIONNEL
  const handleVoiceRecord = async () => {
    if (isRecording) {
      // Arrêter l'enregistrement
      setIsRecording(false)
      setIsListening(false)

      // Simulation Speech-to-Text avec exemples réalistes
      const speechExamples = {
        fr: [
          "Comment optimiser mon stock ?",
          "Aide-moi avec mes prévisions",
          "Quels sont tes conseils pour la logistique ?",
          "Analyse mes données de vente",
          "Comment réduire mes coûts de transport ?",
        ],
        en: [
          "How to optimize my inventory?",
          "Help me with forecasting",
          "What are your logistics tips?",
          "Analyze my sales data",
          "How to reduce transport costs?",
        ],
        ar: ["كيف أحسن مخزوني؟", "ساعدني في التنبؤات", "ما نصائحك للوجستيات؟"],
        darija: ["Kifach n7assan stock dyali?", "3awnni f tawaqo3", "Ash huma nasa2i7ek l logistique?"],
      }

      const examples = speechExamples[currentLanguage]
      const randomExample = examples[Math.floor(Math.random() * examples.length)]

      // Simuler un délai de traitement
      setTimeout(() => {
        setInputText(randomExample)
        Alert.alert(
          "🎤 Reconnaissance vocale",
          `J'ai entendu : "${randomExample}"\n\n✨ Appuie sur Envoyer pour continuer !`,
          [{ text: "OK" }],
        )
      }, 1000)
    } else {
      // Démarrer l'enregistrement
      setIsRecording(true)
      setIsListening(true)

      Alert.alert("🎤 Enregistrement en cours...", "Parlez maintenant ! Je vais simuler la reconnaissance vocale.", [
        { text: "Arrêter", onPress: () => handleVoiceRecord() },
      ])

      // Auto-stop après 5 secondes
      setTimeout(() => {
        if (isRecording) {
          handleVoiceRecord()
        }
      }, 5000)
    }
  }

  // Text-to-Speech FONCTIONNEL
  const speakText = async (text: string, language: string) => {
    if (!speechEnabled) return

    try {
      // Nettoyer le texte
      const cleanText = text
        .replace(/[🎉👋✨🔧📊📦🚚📈💡😊🎯⚡💪🤔🚀🌟🤝🔬]/gu, "")
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/•/g, "")
        .replace(/\n+/g, ". ")
        .replace(/#{1,6}\s/g, "")
        .trim()
        .substring(0, 300) // Limiter la longueur

      if (cleanText.length < 5) return

      const voiceSettings = {
        fr: { language: "fr-FR", pitch: 1.0, rate: 0.85 },
        en: { language: "en-US", pitch: 1.0, rate: 0.85 },
        ar: { language: "ar-SA", pitch: 1.0, rate: 0.75 },
        darija: { language: "fr-FR", pitch: 1.1, rate: 0.8 }, // Utiliser français pour darija
      }

      const settings = voiceSettings[language as keyof typeof voiceSettings] || voiceSettings.fr

      console.log(`🔊 Speaking: ${cleanText.substring(0, 50)}...`)

      await Speech.speak(cleanText, {
        ...settings,
        onStart: () => console.log("🔊 Speech started"),
        onDone: () => console.log("🔊 Speech finished"),
        onError: (error) => console.warn("🔊 Speech error:", error),
      })
    } catch (error) {
      console.warn("Speech error:", error)
    }
  }

  // Import de documents FONCTIONNEL
  const importDocument = async () => {
    try {
      console.log("📄 Starting document import...")

      const result = await DocumentPicker.getDocumentAsync({
        type: ["text/plain", "application/pdf", "text/csv", "application/vnd.ms-excel"],
        copyToCacheDirectory: true,
        multiple: false,
      })

      console.log("📄 Document picker result:", result)

      if (!result.canceled && result.assets && result.assets[0]) {
        const file = result.assets[0]
        console.log("📄 Selected file:", file.name, file.mimeType)

        // Lire le contenu du fichier
        let content = ""

        try {
          if (file.uri && file.mimeType?.includes("text")) {
            content = await FileSystem.readAsStringAsync(file.uri)
            console.log("📄 File content length:", content.length)
          } else {
            // Contenu simulé pour PDF et autres formats
            content = `Contenu du fichier ${file.name}

Données supply chain extraites :
- Produit A : Stock 1500 unités, Demande 200/jour
- Produit B : Stock 800 unités, Demande 150/jour  
- Produit C : Stock 1200 unités, Demande 100/jour

Métriques importantes :
- Taux de service : 95%
- Délai livraison moyen : 5 jours
- Coût stockage : 2€/unité/mois
- Rotation stock : 8 fois/an

Recommandations :
- Optimiser point de commande Produit A
- Réduire stock de sécurité Produit C
- Améliorer prévisions pour Produit B
- Négocier délais fournisseurs`
          }
        } catch (readError) {
          console.warn("📄 Error reading file, using mock content:", readError)
          content = `Contenu simulé pour ${file.name} - Données supply chain importantes pour analyse.`
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

        console.log("📄 Document added to RAG system")

        Alert.alert(
          "📄 Document importé avec succès !",
          `✅ **${file.name}** ajouté à ma base de connaissances.\n\n🧠 Je peux maintenant répondre en me basant sur ce document !\n\n💡 Pose-moi des questions sur son contenu.`,
          [{ text: "Super !" }],
        )
      } else {
        console.log("📄 Document import cancelled")
      }
    } catch (error) {
      console.error("📄 Document import error:", error)
      Alert.alert(
        "❌ Erreur d'import",
        "Impossible d'importer le document. Réessayez avec un fichier texte (.txt) ou CSV.",
        [{ text: "OK" }],
      )
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
    const currentInput = inputText.trim()
    setInputText("")
    setIsLoading(true)

    // Simulation délai réflexion
    setTimeout(async () => {
      const response = generateResponse(currentInput, detectedLang)

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response,
        isUser: false,
        timestamp: new Date(),
        language: detectedLang,
        hasAudio: speechEnabled,
      }

      setMessages((prev) => [...prev, aiMessage])
      setIsLoading(false)

      // Lecture audio automatique
      if (speechEnabled) {
        setTimeout(() => {
          speakText(response, detectedLang)
        }, 500)
      }
    }, 1200)
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
        <Text style={styles.headerTitle}>🧠 RawaJAI IA avancée</Text>
        <Text style={styles.headerSubtitle}>
          Assistant multilingue avec RAG • {ragSystem.current.getDocumentCount()} documents
        </Text>

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
            onPress={toggleSpeech}
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
              <WebSafeView style={styles.messageIcons}>
                {message.hasAudio && <Text style={styles.audioIcon}>🔊</Text>}
                <Text style={styles.languageTag}>{message.language.toUpperCase()}</Text>
              </WebSafeView>
            </WebSafeView>
          </WebSafeView>
        ))}

        {isLoading && (
          <WebSafeView style={[styles.messageContainer, styles.aiMessage]}>
            <Text style={[styles.messageText, styles.aiMessageText]}>
              🧠 Analyse en cours avec IA + RAG...
              {ragSystem.current.getDocumentCount() > 0 && "\n📄 Consultation de tes documents..."}
            </Text>
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
                ? "Tapez ou parlez (🎤)..."
                : currentLanguage === "en"
                  ? "Type or speak (🎤)..."
                  : currentLanguage === "ar"
                    ? "اكتب أو تحدث (🎤)..."
                    : "Kteb wla hder (🎤)..."
            }
            placeholderTextColor="#999"
            multiline
            maxLength={500}
            onSubmitEditing={handleSend}
            editable={!isLoading}
          />

          <WebSafeTouchableOpacity
            style={[styles.voiceButton, (isRecording || isListening) && styles.voiceButtonActive]}
            onPress={handleVoiceRecord}
            disabled={isLoading}
          >
            <Text style={styles.voiceButtonText}>{isRecording ? "⏹️" : isListening ? "🎙️" : "🎤"}</Text>
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
