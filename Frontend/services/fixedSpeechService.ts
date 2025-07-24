// Service Speech-to-Text CORRIGÉ et compatible mobile/web
export class FixedSpeechService {
  private recognition: any = null
  private isSupported = false
  private isInitialized = false

  constructor() {
    this.initializeSpeechRecognition()
  }

  private initializeSpeechRecognition() {
    try {
      // Vérifier si on est sur web
      if (typeof window === "undefined") {
        console.log("🎤 Pas de window - environnement serveur")
        this.isSupported = false
        return
      }

      // Vérifier le support du navigateur
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition

      if (SpeechRecognition) {
        this.recognition = new SpeechRecognition()
        this.isSupported = true
        this.configureSpeechRecognition()
        this.isInitialized = true
        console.log("🎤 Speech Recognition initialisé avec succès")
      } else {
        console.warn("🎤 Speech Recognition non supporté dans ce navigateur")
        this.isSupported = false
        this.isInitialized = true
      }
    } catch (error) {
      console.error("🎤 Erreur initialisation Speech Recognition:", error)
      this.isSupported = false
      this.isInitialized = true
    }
  }

  private configureSpeechRecognition() {
    if (!this.recognition) return

    try {
      // Configuration optimale
      this.recognition.continuous = false
      this.recognition.interimResults = true
      this.recognition.maxAlternatives = 1

      // Langue par défaut
      this.recognition.lang = "fr-FR"

      console.log("🎤 Configuration Speech Recognition terminée")
    } catch (error) {
      console.error("🎤 Erreur configuration:", error)
    }
  }

  // Méthode publique pour vérifier le support
  isSupported(): boolean {
    return this.isSupported && this.isInitialized
  }

  // Méthode pour vérifier si l'initialisation est terminée
  isReady(): boolean {
    return this.isInitialized
  }

  async startListening(language = "fr"): Promise<string> {
    // Attendre l'initialisation si nécessaire
    if (!this.isInitialized) {
      await new Promise((resolve) => setTimeout(resolve, 100))
    }

    if (!this.isSupported) {
      throw new Error("Speech Recognition non supporté")
    }

    return new Promise((resolve, reject) => {
      let finalTranscript = ""
      let interimTranscript = ""
      let hasResult = false

      // Configurer la langue
      this.setLanguage(language)

      // Timeout de sécurité
      const timeout = setTimeout(() => {
        if (!hasResult) {
          this.recognition?.stop()
          reject(new Error("Timeout - Aucune parole détectée"))
        }
      }, 10000) // 10 secondes max

      // Événements
      this.recognition.onstart = () => {
        console.log("🎤 Écoute démarrée...")
      }

      this.recognition.onresult = (event: any) => {
        hasResult = true
        interimTranscript = ""

        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript

          if (event.results[i].isFinal) {
            finalTranscript += transcript
          } else {
            interimTranscript += transcript
          }
        }

        console.log("🎤 Transcription en cours:", finalTranscript || interimTranscript)
      }

      this.recognition.onend = () => {
        clearTimeout(timeout)
        console.log("🎤 Écoute terminée")
        const result = finalTranscript.trim() || interimTranscript.trim()

        if (result && result.length > 0) {
          resolve(result)
        } else {
          reject(new Error("Aucune parole détectée"))
        }
      }

      this.recognition.onerror = (event: any) => {
        clearTimeout(timeout)
        console.error("🎤 Erreur reconnaissance:", event.error)

        let errorMessage = "Erreur de reconnaissance vocale"
        switch (event.error) {
          case "not-allowed":
            errorMessage = "Permission microphone refusée"
            break
          case "no-speech":
            errorMessage = "Aucune parole détectée"
            break
          case "network":
            errorMessage = "Erreur réseau"
            break
          default:
            errorMessage = `Erreur: ${event.error}`
        }

        reject(new Error(errorMessage))
      }

      // Démarrer l'écoute
      try {
        this.recognition.start()
      } catch (error) {
        clearTimeout(timeout)
        reject(error)
      }
    })
  }

  stopListening() {
    if (this.recognition) {
      try {
        this.recognition.stop()
        console.log("🎤 Arrêt forcé de l'écoute")
      } catch (error) {
        console.warn("🎤 Erreur arrêt:", error)
      }
    }
  }

  private setLanguage(language: string) {
    if (!this.recognition) return

    const languageCodes = {
      fr: "fr-FR",
      en: "en-US",
      ar: "ar-SA",
      darija: "ar-MA",
    }

    try {
      this.recognition.lang = languageCodes[language as keyof typeof languageCodes] || "fr-FR"
      console.log(`🎤 Langue configurée: ${this.recognition.lang}`)
    } catch (error) {
      console.warn("🎤 Erreur configuration langue:", error)
    }
  }

  // Méthode alternative avec MediaRecorder (plus compatible)
  async startRecordingWithMediaRecorder(): Promise<Blob> {
    try {
      console.log("🎤 Demande permission microphone...")
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      })

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      })
      const audioChunks: Blob[] = []

      return new Promise((resolve, reject) => {
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.push(event.data)
          }
        }

        mediaRecorder.onstop = () => {
          console.log("🎤 Enregistrement terminé, création du blob...")
          const audioBlob = new Blob(audioChunks, { type: "audio/webm" })

          // Arrêter tous les tracks
          stream.getTracks().forEach((track) => {
            track.stop()
            console.log("🎤 Track arrêté:", track.kind)
          })

          resolve(audioBlob)
        }

        mediaRecorder.onerror = (event) => {
          console.error("🎤 Erreur MediaRecorder:", event)
          stream.getTracks().forEach((track) => track.stop())
          reject(new Error("Erreur d'enregistrement"))
        }

        // Démarrer l'enregistrement
        mediaRecorder.start(1000) // Chunk toutes les secondes
        console.log("🎤 Enregistrement MediaRecorder démarré")

        // Arrêter après 10 secondes max
        setTimeout(() => {
          if (mediaRecorder.state === "recording") {
            console.log("🎤 Arrêt automatique après 10s")
            mediaRecorder.stop()
          }
        }, 10000)
      })
    } catch (error) {
      console.error("🎤 Erreur accès microphone:", error)
      throw new Error("Impossible d'accéder au microphone. Vérifiez les permissions.")
    }
  }

  // Transcription avec API gratuite (simulation améliorée)
  async transcribeAudio(audioBlob: Blob): Promise<string> {
    try {
      console.log("🎤 Transcription audio, taille:", audioBlob.size)

      // Simulation réaliste de transcription
      // Dans un vrai projet, utiliser Whisper API ou Google Speech
      await new Promise((resolve) => setTimeout(resolve, 2000)) // Simule le traitement

      const realisticTranscriptions = [
        "Comment optimiser mon stock de produits ?",
        "Aide-moi à faire des prévisions pour le mois prochain",
        "Quels sont les KPIs importants pour ma supply chain ?",
        "Comment réduire mes coûts de transport ?",
        "Analyse mes données de vente s'il te plaît",
        "Je veux améliorer mon taux de service client",
        "Peux-tu m'expliquer la méthode EOQ ?",
        "Comment calculer mon stock de sécurité ?",
        "Quelles sont les meilleures pratiques en logistique ?",
        "Comment améliorer mes prévisions de demande ?",
      ]

      const randomTranscription = realisticTranscriptions[Math.floor(Math.random() * realisticTranscriptions.length)]
      console.log("🎤 Transcription simulée:", randomTranscription)

      return randomTranscription
    } catch (error) {
      console.error("🎤 Erreur transcription:", error)
      throw new Error("Impossible de transcrire l'audio")
    }
  }

  // Vérifier les permissions microphone
  async checkMicrophonePermission(): Promise<boolean> {
    try {
      const result = await navigator.permissions.query({ name: "microphone" as PermissionName })
      console.log("🎤 Permission microphone:", result.state)
      return result.state === "granted"
    } catch (error) {
      console.warn("🎤 Impossible de vérifier les permissions:", error)
      return false
    }
  }
}
