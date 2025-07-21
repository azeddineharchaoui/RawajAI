// Service Text-to-Speech optimisé sans répétitions
export class OptimizedTTSService {
  private isSpeaking = false
  private currentSpeech: any = null
  private speechQueue: string[] = []

  async speak(text: string, language: string, options: any = {}): Promise<void> {
    // Éviter les répétitions
    if (this.isSpeaking) {
      console.log("🔊 Speech déjà en cours, ajout à la queue")
      this.speechQueue.push(text)
      return
    }

    try {
      // Nettoyer le texte
      const cleanText = this.cleanTextForSpeech(text)

      if (cleanText.length < 3) {
        console.log("🔊 Texte trop court, pas de lecture")
        return
      }

      this.isSpeaking = true
      console.log("🔊 Début lecture:", cleanText.substring(0, 50) + "...")

      // Paramètres vocaux optimisés
      const speechOptions = {
        language: this.getLanguageCode(language),
        pitch: options.pitch || 1.0,
        rate: options.rate || 0.85,
        onStart: () => {
          console.log("🔊 Lecture démarrée")
        },
        onDone: () => {
          console.log("🔊 Lecture terminée")
          this.isSpeaking = false
          this.processQueue()
        },
        onError: (error: any) => {
          console.warn("🔊 Erreur lecture:", error)
          this.isSpeaking = false
          this.processQueue()
        },
      }

      // Utiliser expo-speech si disponible, sinon Web Speech API
      if (typeof require !== "undefined") {
        const Speech = require("expo-speech")
        await Speech.speak(cleanText, speechOptions)
      } else {
        // Fallback Web Speech API
        await this.webSpeak(cleanText, speechOptions)
      }
    } catch (error) {
      console.error("🔊 Erreur TTS:", error)
      this.isSpeaking = false
      this.processQueue()
    }
  }

  private async webSpeak(text: string, options: any): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!("speechSynthesis" in window)) {
        reject(new Error("Web Speech API non supportée"))
        return
      }

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.lang = options.language
      utterance.pitch = options.pitch
      utterance.rate = options.rate

      utterance.onstart = options.onStart
      utterance.onend = () => {
        options.onDone()
        resolve()
      }
      utterance.onerror = (event) => {
        options.onError(event.error)
        reject(event.error)
      }

      // Arrêter toute lecture en cours
      speechSynthesis.cancel()

      // Démarrer la nouvelle lecture
      speechSynthesis.speak(utterance)
      this.currentSpeech = utterance
    })
  }

  private cleanTextForSpeech(text: string): string {
    return (
      text
        // Supprimer les emojis
        .replace(/[🎉👋✨🔧📊📦🚚📈💡😊🎯⚡💪🤔🚀🌟🤝🔬📄🤖]/gu, "")
        // Supprimer le markdown
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/\*(.*?)\*/g, "$1")
        .replace(/`(.*?)`/g, "$1")
        // Supprimer les listes
        .replace(/•/g, "")
        .replace(/^\s*[-*+]\s+/gm, "")
        // Supprimer les titres markdown
        .replace(/#{1,6}\s/g, "")
        // Remplacer les sauts de ligne par des pauses
        .replace(/\n+/g, ". ")
        // Supprimer les références de documents
        .replace(/📄.*?:/g, "")
        .replace(/🤖.*?:/g, "")
        // Nettoyer les espaces
        .replace(/\s+/g, " ")
        .trim()
        // Limiter la longueur
        .substring(0, 400)
    )
  }

  private getLanguageCode(language: string): string {
    const languageCodes = {
      fr: "fr-FR",
      en: "en-US",
      ar: "ar-SA",
      darija: "ar-MA",
    }

    return languageCodes[language as keyof typeof languageCodes] || "fr-FR"
  }

  private processQueue() {
    if (this.speechQueue.length > 0 && !this.isSpeaking) {
      const nextText = this.speechQueue.shift()
      if (nextText) {
        // Petit délai pour éviter les conflits
        setTimeout(() => {
          this.speak(nextText, "fr")
        }, 500)
      }
    }
  }

  stop() {
    try {
      this.isSpeaking = false
      this.speechQueue = []

      // Arrêter expo-speech
      if (typeof require !== "undefined") {
        const Speech = require("expo-speech")
        Speech.stop()
      }

      // Arrêter Web Speech API
      if ("speechSynthesis" in window) {
        speechSynthesis.cancel()
      }

      console.log("🔊 Lecture arrêtée")
    } catch (error) {
      console.warn("🔊 Erreur arrêt:", error)
    }
  }

  isSpeakingNow(): boolean {
    return this.isSpeaking
  }
}
