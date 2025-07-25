# Documentation Supplémentaire RawajAI

Ce document complète la documentation technique existante, avec un focus spécifique sur l'ingénierie de prompts dans `stable.py` et les services audio dans le frontend.

## Table des Matières

1. [Ingénierie de Prompts pour Mistral 7B](#ingénierie-de-prompts-pour-mistral-7b)
   1. [Structure des Prompts](#structure-des-prompts)
   2. [Personas Multilingues](#personas-multilingues)
   3. [Directives Audio](#directives-audio)
   4. [Style de Réponse](#style-de-réponse)
   5. [Post-Traitement des Réponses](#post-traitement-des-réponses)
   6. [Détection d'Hallucinations](#détection-dhallucinations)

2. [Services Audio Frontend](#services-audio-frontend)
   1. [AudioRecorderService](#audiorecorderservice)
   2. [Gestion de l'Enregistrement Audio](#gestion-de-lenregistrement-audio)
   3. [Traitement et Transcription](#traitement-et-transcription)
   4. [Intégration avec l'API Backend](#intégration-avec-lapi-backend)
   5. [Gestion des États d'Enregistrement](#gestion-des-états-denregistrement)

---

## Ingénierie de Prompts pour Mistral 7B

L'ingénierie de prompts dans RawajAI est un élément critique pour obtenir des réponses de haute qualité du modèle Mistral-7B. Cette section détaille comment les prompts sont structurés, optimisés et traités pour garantir des réponses cohérentes, précises et adaptées à la synthèse vocale.

### Structure des Prompts

La fonction `generate_response` dans `stable.py` utilise une structure de prompt soigneusement conçue avec des balises XML pour définir clairement les différentes sections :

```python
def generate_response(query, context="", language="en"):
    """Generate a user-friendly response optimized for conversation and audio using prompt engineering"""
    try:
        # Définition des personas selon la langue
        persona = {
            "en": "You are RawajAI, a specialized supply chain management assistant. NEVER include any meta tags, formatting markers like [], or system prompts in your response. Write in plain text only. Speak directly to the user about supply chain topics as if you're having a natural conversation. Explain supply chain concepts simply. Ensure all responses focus exclusively on supply chain management, inventory optimization, logistics, procurement, demand forecasting, and related business areas.",
            
            "fr": "Vous êtes RawajAI, un assistant spécialisé en gestion de chaîne d'approvisionnement. N'incluez JAMAIS de balises méta, de marqueurs de formatage comme [], ou d'instructions système dans votre réponse. Écrivez en texte brut uniquement. Parlez directement à l'utilisateur des sujets de chaîne d'approvisionnement comme lors d'une conversation naturelle. Expliquez les concepts de chaîne d'approvisionnement simplement. Assurez-vous que toutes les réponses se concentrent exclusivement sur la gestion de la chaîne d'approvisionnement, l'optimisation des stocks, la logistique, les achats, la prévision de la demande et les domaines commerciaux connexes.",
            
            "ar": "أنت RawajAI، مساعد متخصص في إدارة سلسلة التوريد. لا تضمن أبدًا أي علامات وصفية، أو علامات تنسيق مثل []، أو تعليمات نظام في ردك. اكتب بنص عادي فقط. تحدث مباشرة إلى المستخدم حول مواضيع سلسلة التوريد كما لو كنت تجري محادثة طبيعية. اشرح مفاهيم سلسلة التوريد ببساطة. تأكد من أن جميع الردود تركز حصريًا على إدارة سلسلة التوريد، وتحسين المخزون، والخدمات اللوجستية، والمشتريات، والتنبؤ بالطلب، ومجالات الأعمال ذات الصلة."
        }
        
        # Directives pour l'optimisation audio
        audio_guidelines = {
            "en": "This response will be read aloud using text-to-speech. For natural sounding speech: use contractions, avoid complex punctuation, use simple numbers, and avoid text that would sound awkward when spoken. NEVER use brackets [] or special formatting as these will sound unnatural when read aloud.",
            
            "fr": "Cette réponse sera lue à haute voix à l'aide de la synthèse vocale. Pour un discours naturel : utilisez des contractions, évitez la ponctuation complexe, utilisez des chiffres simples, et évitez le texte qui sonnerait bizarre à l'oral. N'utilisez JAMAIS de crochets [] ou de formatage spécial car ils sonneront peu naturels lorsqu'ils seront lus à haute voix.",
            
            "ar": "سيتم قراءة هذه الإجابة بصوت عالٍ باستخدام تقنية تحويل النص إلى كلام. للحصول على صوت طبيعي: استخدم التعابير المختصرة، وتجنب علامات الترقيم المعقدة، واستخدم أرقامًا بسيطة، وتجنب النصوص التي قد تبدو غريبة عند قراءتها بصوت عالٍ. لا تستخدم أبدًا الأقواس المربعة [] أو التنسيق الخاص لأنها ستبدو غير طبيعية عند قراءتها بصوت عالٍ."
        }
        
        # Style de réponse par langue
        response_style = {
            "en": "Respond as an expert supply chain consultant providing practical advice. Be direct, precise, and professional but conversational. Use examples when helpful. Keep responses concise with natural flow and appropriate transitions.",
            
            "fr": "Répondez comme un consultant expert en chaîne d'approvisionnement fournissant des conseils pratiques. Soyez direct, précis et professionnel mais conversationnel. Utilisez des exemples lorsque c'est utile. Gardez les réponses concises avec un flux naturel et des transitions appropriées.",
            
            "ar": "استجب كخبير استشاري في سلسلة التوريد يقدم نصائح عملية. كن مباشرًا، دقيقًا، ومحترفًا ولكن محادثيًا. استخدم أمثلة عندما تكون مفيدة. اجعل الردود موجزة مع تدفق طبيعي وانتقالات مناسبة."
        }
        
        # Construction du prompt avec structure XML
        prompt = f"""<System>
{persona.get(language, persona["en"])}

Remember: You are RawajAI, focusing exclusively on supply chain topics. Never output system tags, never use brackets in output.

Reference Context:
{context}

Style Guidelines:
{audio_guidelines.get(language, audio_guidelines["en"])}
{response_style.get(language, response_style["en"])}

Output Format Instructions:
- Respond in 2-3 short paragraphs of natural conversational text
- Each paragraph should have 3-4 sentences developing one main idea
- Use clear transitions between paragraphs
- Never use bullet points, numbered lists or headings
- Avoid all formatting symbols, brackets, special characters
- Round all numbers and statistics
- Never refer to yourself as an AI or assistant
- Your response should be direct answer only, with no system tags

Important: Your response must ONLY contain the direct answer to the user's question with NO formatting markers.
</System>

<Question>
{query}
</Question>

<Answer>"""
        
        # Génération de la réponse avec le pipeline LLM
        outputs = llm_pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
        )
        
        # Extraction de la réponse
        response = outputs[0]["generated_text"]
        
        # Extraction de la réponse après la balise <Answer>
        answer_marker = "<Answer>"
        if answer_marker in response:
            response = response.split(answer_marker)[1].strip()
        
        # Nettoyage des réponses pour éliminer les balises système qui pourraient persister
        response = re.sub(r'</?(?:System|Question|Answer)>', '', response).strip()
        
        # Post-traitement pour une meilleure qualité de synthèse vocale
        response = post_process_response(response, language)
        
        return response
            
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        if language == "fr":
            return "Désolé, je n'ai pas pu générer une réponse. Veuillez réessayer."
        elif language == "ar":
            return "عذرًا، لم أتمكن من إنشاء إجابة. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, I couldn't generate a response. Please try again."
```

Cette structure de prompt comprend plusieurs éléments clés :

1. **Balises XML** : `<System>`, `<Question>` et `<Answer>` qui définissent clairement les différentes parties du prompt
2. **Persona** : Définit l'identité et le style de l'assistant selon la langue
3. **Contexte de référence** : Information extraite de la base de connaissances vectorielle pour améliorer la précision
4. **Directives de style** : Instructions pour optimiser la sortie pour la synthèse vocale
5. **Instructions de format** : Règles précises sur la structure et le style des réponses
6. **Question** : La requête de l'utilisateur

### Personas Multilingues

Les personas sont définis pour chaque langue supportée (anglais, français et arabe) et établissent l'identité et le ton de l'assistant. Chaque persona contient des instructions spécifiques pour :

1. **Éviter les métadonnées** : Instructions explicites pour ne pas inclure de balises, marqueurs de formatage ou instructions système
2. **Maintenir la concentration thématique** : Focaliser exclusivement sur la gestion de la chaîne d'approvisionnement
3. **Établir le ton conversationnel** : Parler directement à l'utilisateur de manière naturelle
4. **Simplifier les concepts complexes** : Expliquer les concepts de chaîne d'approvisionnement simplement

### Directives Audio

Les directives audio sont spécifiquement conçues pour optimiser les réponses pour la synthèse vocale :

```python
audio_guidelines = {
    "en": "This response will be read aloud using text-to-speech. For natural sounding speech: use contractions, avoid complex punctuation, use simple numbers, and avoid text that would sound awkward when spoken. NEVER use brackets [] or special formatting as these will sound unnatural when read aloud.",
    
    "fr": "Cette réponse sera lue à haute voix à l'aide de la synthèse vocale. Pour un discours naturel : utilisez des contractions, évitez la ponctuation complexe, utilisez des chiffres simples, et évitez le texte qui sonnerait bizarre à l'oral. N'utilisez JAMAIS de crochets [] ou de formatage spécial car ils sonneront peu naturels lorsqu'ils seront lus à haute voix.",
    
    "ar": "سيتم قراءة هذه الإجابة بصوت عالٍ باستخدام تقنية تحويل النص إلى كلام. للحصول على صوت طبيعي: استخدم التعابير المختصرة، وتجنب علامات الترقيم المعقدة، واستخدم أرقامًا بسيطة، وتجنب النصوص التي قد تبدو غريبة عند قراءتها بصوت عالٍ. لا تستخدم أبدًا الأقواس المربعة [] أو التنسيق الخاص لأنها ستبدو غير طبيعية عند قراءتها بصوت عالٍ."
}
```

Ces directives incluent des instructions spécifiques pour :
1. Utiliser des contractions pour un discours plus naturel
2. Éviter la ponctuation complexe qui peut causer des problèmes en TTS
3. Simplifier les nombres pour une meilleure prononciation
4. Éviter les crochets et formatages spéciaux qui sonnent mal à l'oral

### Style de Réponse

Le style de réponse établit le ton professionnel mais conversationnel que l'assistant doit adopter :

```python
response_style = {
    "en": "Respond as an expert supply chain consultant providing practical advice. Be direct, precise, and professional but conversational. Use examples when helpful. Keep responses concise with natural flow and appropriate transitions.",
    
    "fr": "Répondez comme un consultant expert en chaîne d'approvisionnement fournissant des conseils pratiques. Soyez direct, précis et professionnel mais conversationnel. Utilisez des exemples lorsque c'est utile. Gardez les réponses concises avec un flux naturel et des transitions appropriées.",
    
    "ar": "استجب كخبير استشاري في سلسلة التوريد يقدم نصائح عملية. كن مباشرًا، دقيقًا، ومحترفًا ولكن محادثيًا. استخدم أمثلة عندما تكون مفيدة. اجعل الردود موجزة مع تدفق طبيعي وانتقالات مناسبة."
}
```

Ces styles de réponse guident le modèle pour :
1. Adopter le rôle d'un consultant expert en chaîne d'approvisionnement
2. Fournir des conseils pratiques et applicables
3. Maintenir un équilibre entre professionnalisme et conversationnel
4. Utiliser des exemples pour illustrer les concepts
5. Garder les réponses concises avec des transitions naturelles

### Post-Traitement des Réponses

La fonction `post_process_response` joue un rôle crucial dans le nettoyage et l'optimisation des réponses pour une meilleure expérience utilisateur et une synthèse vocale plus naturelle :

```python
def post_process_response(text, language="en"):
    """
    Post-process the LLM response to make it more suitable for conversation and TTS
    
    This function cleans up the response to ensure it's well-formatted for speech synthesis
    and doesn't contain elements that would sound awkward when spoken.
    
    It also ensures responses are formatted in 2-3 paragraphs without numbered phrases.
    """
    
    # Skip if the text is empty or too short
    if not text or len(text.strip()) < 5:
        return text
    
    # Remove any markdown code blocks
    text = re.sub(r'```(?:[a-zA-Z]+)?\n[\s\S]*?\n```', ' ', text)
    
    # Remove JSON-like structures 
    text = re.sub(r'\{[\s\S]*?\}', ' ', text)
    
    # Remove numbered list markers (1., 2., etc.) 
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove bullet points
    text = re.sub(r'^\s*[\*\-•]\s+', '', text, flags=re.MULTILINE)
    
    # Replace symbols that don't work well in speech
    replacements = {
        '```': ' ',
        '**': ' ',
        '*': ' ',
        '#': ' ',
        '|': ', ',
        '=': ' equals ',
        '->': ' to ',
        '<-': ' from ',
        '>=': ' greater than or equal to ',
        '<=': ' less than or equal to ',
        '>': ' greater than ',
        '<': ' less than ',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix spacing issues
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure proper sentence breaks for TTS natural pauses
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Handle numbers for better speech (e.g., "100,000" -> "100 thousand")
    def number_to_words(match):
        num = match.group(0).replace(',', '')
        if len(num) >= 7:
            return f"{int(num) // 1000000} million"
        elif len(num) >= 4:
            return f"{int(num) // 1000} thousand"
        return num
    
    text = re.sub(r'\b\d{4,}(?:,\d{3})*\b', number_to_words, text)
    
    # Language-specific post-processing
    if language == "fr":
        # Fix spacing for French punctuation
        text = re.sub(r'\s+([;:!?])', r' \1', text)
    
    # Remove any system prompt leakage phrases
    system_leakage_phrases = [
        "As a supply chain assistant",
        "As an AI assistant",
        "As your AI assistant",
        "As a helpful assistant",
        "I don't have access to",
        "I don't have the ability to",
        "I'm not able to",
        "I cannot access",
        "As RawajAI",
        "As a language model",
        "I'm here to help",
        "I'm an AI",
        "I'm a supply chain",
        "I'm a helpful",
    ]
    
    for phrase in system_leakage_phrases:
        if phrase.lower() in text.lower():
            text = re.sub(re.escape(phrase), '', text, flags=re.IGNORECASE)
            
    # Remove any text inside brackets (common metadata marker)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove any XML-like tags (some models output these)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Ensure the text doesn't end with an incomplete sentence
    if not re.search(r'[.!?]$', text):
        text = text + "."
    
    # Format into 2-3 paragraphs with natural breaks
    paragraphs = re.split(r'\n\s*\n', text)
    
    # If we have too many paragraphs, consolidate them
    if len(paragraphs) > 3:
        new_paragraphs = []
        current = ""
        for i, p in enumerate(paragraphs):
            if i % 2 == 0 and i > 0:
                new_paragraphs.append(current.strip())
                current = p
            else:
                if current:
                    current += " " + p
                else:
                    current = p
        if current:
            new_paragraphs.append(current.strip())
        paragraphs = new_paragraphs
    
    # Join paragraphs with double newlines for proper spacing
    text = "\n\n".join(p for p in paragraphs if p.strip())
    
    # Final clean-up for multiple spaces and ensure good punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', text)
    
    return text
```

Ce post-traitement effectue plusieurs opérations importantes :

1. **Nettoyage du formatage** : Suppression des blocs de code, structures JSON, listes numérotées et puces
2. **Adaptation des symboles** : Remplacement des symboles problématiques pour la TTS (comme `>=` par "greater than or equal to")
3. **Correction des espaces et de la ponctuation** : Normalisation des espaces et ajustements spécifiques à la langue
4. **Traitement des nombres** : Conversion des grands nombres en format plus parlé (ex: "100,000" devient "100 thousand")
5. **Suppression des fuites de prompt système** : Élimination des phrases comme "As an AI assistant" qui révèlent la nature du modèle
6. **Suppression du texte entre crochets** : Élimination des métadonnées souvent présentes entre crochets
7. **Structuration en paragraphes** : Organisation du texte en 2-3 paragraphes pour une meilleure lisibilité et écoute
8. **Formatage final** : Nettoyage des espaces multiples et amélioration de la ponctuation pour les pauses naturelles

### Détection d'Hallucinations

Le système inclut des mécanismes pour détecter et prévenir les hallucinations ou les sorties non conformes :

1. **Vérification des balises système** : La réponse est nettoyée de toutes balises XML qui pourraient avoir fuité
2. **Élimination des phrases de fuite** : Les phrases typiques qui révèlent la nature du modèle sont supprimées
3. **Détection de contenu entre crochets** : Tout texte entre crochets (souvent des métadonnées ou des instructions) est éliminé

Ces mécanismes contribuent à une expérience utilisateur plus naturelle et cohérente, particulièrement importante lors de l'utilisation de la synthèse vocale.

---

## Services Audio Frontend

Le frontend de RawajAI inclut des services audio sophistiqués pour permettre l'interaction vocale avec l'assistant. Ces services sont principalement implémentés dans la classe `AudioRecorderService`.

### AudioRecorderService

Le service `AudioRecorderService` est une classe TypeScript qui gère l'enregistrement audio, la transcription, et l'interaction avec l'API backend pour obtenir des réponses vocales :

```typescript
class AudioRecorderService {
  private recording: Audio.Recording | null = null;
  private isRecording = false;
  private isProcessing = false;
  private recordingUri: string | null = null;
  private statusListeners: ((status: RecordingStatus) => void)[] = [];

  constructor() {
    this.initializeAudio();
  }

  // Méthodes principales (détaillées ci-dessous)...
}

// Export d'une instance singleton
export const audioRecorder = new AudioRecorderService();
```

Le service maintient plusieurs états internes :
- `recording` : L'instance d'enregistrement active
- `isRecording` : Indique si un enregistrement est en cours
- `isProcessing` : Indique si un enregistrement est en cours de traitement
- `recordingUri` : URI du fichier audio enregistré
- `statusListeners` : Liste des fonctions callback pour les mises à jour de statut

### Gestion de l'Enregistrement Audio

Le service initialise l'audio et gère le cycle d'enregistrement complet :

```typescript
/**
 * Initialize audio mode for recording
 */
private async initializeAudio(): Promise<void> {
  try {
    // Request audio permissions
    const { status } = await Audio.requestPermissionsAsync();
    if (status !== 'granted') {
      throw new Error('Audio recording permission not granted');
    }

    // Set audio mode for recording
    await Audio.setAudioModeAsync({
      allowsRecordingIOS: true,
      playsInSilentModeIOS: true,
      shouldDuckAndroid: true,
      playThroughEarpieceAndroid: false,
      staysActiveInBackground: false,
    });
  } catch (error) {
    console.error('Failed to initialize audio:', error);
    throw error;
  }
}

/**
 * Start recording audio
 */
async startRecording(): Promise<void> {
  try {
    if (this.isRecording) {
      console.log('Already recording');
      return;
    }

    // Clean up any existing recording
    await this.stopRecording();

    // Initialize audio if not done
    await this.initializeAudio();

    console.log('Starting recording...');
    
    // Create a new recording instance
    this.recording = new Audio.Recording();
    
    // Configure recording options for WAV format with high quality
    const recordingOptions = {
      isMeteringEnabled: true,
      android: {
        extension: '.wav',
        outputFormat: Audio.AndroidOutputFormat.DEFAULT,
        audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
        sampleRate: 16000, // Whisper works well with 16kHz
        numberOfChannels: 1, // Mono
        bitRate: 256000,
      },
      ios: {
        extension: '.wav',
        outputFormat: Audio.IOSOutputFormat.LINEARPCM,
        audioQuality: Audio.IOSAudioQuality.HIGH,
        sampleRate: 16000, // Whisper works well with 16kHz
        numberOfChannels: 1, // Mono
        bitRate: 256000,
        linearPCMBitDepth: 16,
        linearPCMIsBigEndian: false,
        linearPCMIsFloat: false,
      },
      web: {
        mimeType: 'audio/wav',
        bitsPerSecond: 256000,
      },
    };

    await this.recording.prepareToRecordAsync(recordingOptions);
    await this.recording.startAsync();
    
    this.isRecording = true;
    this.notifyStatusChange();

    // Set up status updates during recording
    this.recording.setOnRecordingStatusUpdate((status) => {
      if (status.isRecording) {
        this.notifyStatusChange({
          isRecording: true,
          isProcessing: false,
          duration: status.durationMillis,
        });
      }
    });

    console.log('Recording started successfully');
  } catch (error) {
    console.error('Failed to start recording:', error);
    this.isRecording = false;
    this.notifyStatusChange({
      isRecording: false,
      isProcessing: false,
      error: error instanceof Error ? error.message : 'Failed to start recording',
    });
    throw error;
  }
}

/**
 * Stop recording and return the URI
 */
async stopRecording(): Promise<string | null> {
  try {
    if (!this.recording || !this.isRecording) {
      console.log('No active recording to stop');
      return null;
    }

    console.log('Stopping recording...');
    
    await this.recording.stopAndUnloadAsync();
    const uri = this.recording.getURI();
    
    this.recordingUri = uri;
    this.isRecording = false;
    this.recording = null;
    
    this.notifyStatusChange({
      isRecording: false,
      isProcessing: false,
    });

    console.log('Recording stopped. URI:', uri);
    return uri;
  } catch (error) {
    console.error('Failed to stop recording:', error);
    this.isRecording = false;
    this.recording = null;
    this.notifyStatusChange({
      isRecording: false,
      isProcessing: false,
      error: error instanceof Error ? error.message : 'Failed to stop recording',
    });
    return null;
  }
}
```

Points clés de l'implémentation :

1. **Gestion des permissions** : Demande explicite des permissions audio
2. **Configuration audio optimisée** : Paramètres spécifiques pour Android, iOS et Web
3. **Format WAV optimisé pour Whisper** : 16kHz, mono, haute qualité
4. **Gestion robuste des erreurs** : Capture et notification des erreurs
5. **Notifications d'état en temps réel** : Mises à jour pendant l'enregistrement

### Traitement et Transcription

Une fois l'enregistrement terminé, le service traite l'audio pour la transcription et la réponse TTS :

```typescript
/**
 * Process a recorded audio file: upload, transcribe, and get TTS response
 * @param uri The URI of the recorded audio file
 * @param language The language for transcription and TTS
 * @returns Promise with the transcription and TTS result
 */
async processRecording(uri: string, language: string = 'en'): Promise<TranscriptionResult> {
  try {
    this.isProcessing = true;
    this.notifyStatusChange({
      isRecording: false,
      isProcessing: true,
    });

    console.log('Processing recording:', uri);

    // Step 1: Upload audio for transcription
    console.log('Step 1: Uploading audio for transcription...');
    
    // Create form data for the file upload
    const formData = new FormData();
    const filename = `recording_${Date.now()}.wav`;
    
    // Append the audio file with proper metadata
    formData.append('audio', {
      uri: uri,
      name: filename,
      type: 'audio/wav'
    } as any);
    
    // Append language
    formData.append('language', language);
    // Request speech generation
    formData.append('generate_speech', 'true');
    
    console.log(`Sending audio file ${filename} to backend for processing...`);
    
    // Get the base URL for the API
    const baseUrl = await getBaseUrl();
    
    // Make the upload request
    const response = await fetch(`${baseUrl}/upload_audio`, {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed with status: ${response.status}`);
    }
    
    // Parse the response
    const result = await response.json();
    
    if (!result.status || (result.status !== 'success' && !result.transcription)) {
      throw new Error(result.error || 'Failed to transcribe audio');
    }

    const transcription = result.transcription || '';
    console.log('Transcription received:', transcription);
    
    // The backend now handles both transcription and TTS response
    console.log('Backend already processed the transcription and generated a response with TTS');
    
    this.isProcessing = false;
    this.notifyStatusChange({
      isRecording: false,
      isProcessing: false,
    });

    console.log('Processing completed successfully');

    return {
      success: true,
      transcription: transcription,
      query: transcription,
      response: result.response || '',
      language: result.language_detected || language,
      speech_url: result.speech_url,
    };

  } catch (error) {
    console.error('Failed to process recording:', error);
    
    this.isProcessing = false;
    this.notifyStatusChange({
      isRecording: false,
      isProcessing: false,
      error: error instanceof Error ? error.message : 'Failed to process recording',
    });

    return {
      success: false,
      error: error instanceof Error ? error.message : 'Failed to process recording',
    };
  }
}
```

Ce processus comprend plusieurs étapes :

1. **Préparation du FormData** : Création d'un objet FormData avec le fichier audio et les paramètres nécessaires
2. **Configuration de la requête** : Définition du type de contenu et des en-têtes
3. **Envoi vers l'endpoint `/upload_audio`** : Transmission de l'audio au backend
4. **Traitement de la réponse** : Analyse et validation de la réponse du serveur
5. **Notification de l'état final** : Mise à jour du statut de traitement
6. **Retour des résultats** : Transcription, réponse et URL audio

### Intégration avec l'API Backend

Le service utilise une fonction `getBaseUrl` pour déterminer l'URL de l'API backend, avec prise en charge des tunnels Cloudflare :

```typescript
// Base URL helper function that uses the same logic as api.ts
const getBaseUrl = async (): Promise<string> => {
  // Try to get the tunnel URL first
  const tunnelUrl = await getTunnelUrl();
  if (tunnelUrl) return tunnelUrl;
  
  // For Android emulators, localhost refers to the emulator itself
  if (Platform.OS === 'android') {
    return API_URL.replace('localhost', '10.0.2.2');
  }
  
  return API_URL;
};
```

Cette fonction :
1. Tente d'abord d'obtenir une URL de tunnel Cloudflare (pour l'accès à distance)
2. Gère les spécificités des émulateurs Android (remplacement de localhost par 10.0.2.2)
3. Utilise l'URL d'API par défaut comme solution de repli

### Gestion des États d'Enregistrement

Le service implémente un système d'observateurs pour notifier les composants UI des changements d'état :

```typescript
/**
 * Add a listener for recording status updates
 * @param listener Callback function that receives status updates
 * @returns Function to remove the listener
 */
addStatusListener(listener: (status: RecordingStatus) => void): () => void {
  this.statusListeners.push(listener);
  
  // Call listener immediately with current status
  listener({
    isRecording: this.isRecording,
    isProcessing: this.isProcessing,
  });

  // Return unsubscribe function
  return () => {
    this.statusListeners = this.statusListeners.filter(l => l !== listener);
  };
}

/**
 * Notify all listeners of status changes
 */
private notifyStatusChange(status?: Partial<RecordingStatus>): void {
  const currentStatus: RecordingStatus = {
    isRecording: this.isRecording,
    isProcessing: this.isProcessing,
    ...status,
  };

  this.statusListeners.forEach(listener => {
    try {
      listener(currentStatus);
    } catch (error) {
      console.error('Error in status listener:', error);
    }
  });
}
```

Ce système :
1. Permet aux composants de s'abonner aux mises à jour d'état
2. Fournit une fonction de désabonnement pour éviter les fuites de mémoire
3. Notifie immédiatement l'état actuel lors de l'abonnement
4. Gère les erreurs dans les callbacks pour éviter les plantages
5. Fusionne les mises à jour partielles avec l'état complet

Le service expose également des méthodes d'aide pour l'UI :

```typescript
/**
 * Get current recording status
 */
getStatus(): RecordingStatus {
  return {
    isRecording: this.isRecording,
    isProcessing: this.isProcessing,
  };
}

/**
 * Check if currently recording
 */
isCurrentlyRecording(): boolean {
  return this.isRecording;
}

/**
 * Check if currently processing
 */
isCurrentlyProcessing(): boolean {
  return this.isProcessing;
}
```

Ces méthodes facilitent l'intégration avec l'interface utilisateur, permettant aux composants de réagir rapidement aux changements d'état.

---

## Intégration dans le Frontend

L'intégration du service d'enregistrement audio avec l'interface utilisateur se fait principalement dans l'écran `assistant.tsx` :

```typescript
// Dans assistant.tsx
import { audioRecorder, RecordingStatus } from '@/services/AudioRecorderService';

export default function AssistantScreen() {
  // États locaux
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  
  // Configurer l'écouteur d'état d'enregistrement
  useEffect(() => {
    const unsubscribe = audioRecorder.addStatusListener((status: RecordingStatus) => {
      setIsRecording(status.isRecording);
      setIsProcessing(status.isProcessing);
      if (status.duration !== undefined) {
        setRecordingDuration(status.duration);
      }
      if (status.error) {
        // Afficher l'erreur à l'utilisateur
        Alert.alert('Error', status.error);
      }
    });
    
    // Nettoyage lors du démontage
    return () => unsubscribe();
  }, []);
  
  // Gérer le début/fin d'enregistrement
  const handleToggleRecording = async () => {
    try {
      if (isRecording) {
        // Arrêter et traiter l'enregistrement
        const result = await audioRecorder.stopRecordingAndProcess();
        if (result.success) {
          // Ajouter la transcription et la réponse à la conversation
          addMessageToConversation({
            type: 'user',
            content: result.transcription || '',
          });
          
          addMessageToConversation({
            type: 'assistant',
            content: result.response || '',
            audioUrl: result.speech_url,
          });
        }
      } else {
        // Commencer un nouvel enregistrement
        await audioRecorder.startRecording();
      }
    } catch (error) {
      console.error('Error toggling recording:', error);
      Alert.alert('Error', 'Failed to manage recording');
    }
  };
  
  // Rendu du bouton d'enregistrement
  const renderRecordButton = () => {
    const buttonColor = isRecording ? '#ff3b30' : '#007aff';
    const iconName = isRecording ? 'stop-circle' : 'mic';
    
    return (
      <TouchableOpacity 
        style={[styles.recordButton, { backgroundColor: buttonColor }]}
        onPress={handleToggleRecording}
        disabled={isProcessing}
      >
        <Icon name={iconName} size={24} color="#ffffff" />
        {isProcessing && <ActivityIndicator color="#ffffff" style={styles.processingIndicator} />}
      </TouchableOpacity>
    );
  };
  
  // Reste du composant...
}
```

Cette intégration comporte plusieurs aspects importants :
1. Abonnement aux mises à jour d'état d'enregistrement
2. Gestion du toggle d'enregistrement (démarrer/arrêter)
3. Traitement des résultats de transcription et de réponse
4. Mise à jour de l'interface utilisateur en fonction de l'état
5. Retour visuel pour l'utilisateur (couleurs, icônes, indicateurs d'activité)

---

## Conclusion

L'ingénierie de prompts dans `stable.py` et les services audio du frontend sont des composants essentiels de RawajAI qui permettent une interaction naturelle, multilingue et vocale avec l'assistant de chaîne d'approvisionnement.

Le système utilise des techniques avancées pour :
1. **Structurer les prompts** avec une approche XML qui délimite clairement les sections
2. **Adapter le ton et le style** en fonction de la langue de l'utilisateur
3. **Optimiser les réponses pour la synthèse vocale** grâce à des directives spécifiques
4. **Éliminer les métadonnées et balises système** qui pourraient dégrader l'expérience utilisateur
5. **Enregistrer et traiter l'audio** dans un format optimal pour la reconnaissance vocale

Ces fonctionnalités, combinées avec l'architecture modulaire et l'intégration frontend-backend fluide, permettent à RawajAI d'offrir une expérience utilisateur de haute qualité pour la gestion de chaîne d'approvisionnement.
