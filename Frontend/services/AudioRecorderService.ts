import { Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system';
import { Platform } from 'react-native';
import Constants from 'expo-constants';
import { api } from '@/services/api';
import { getTunnelUrl } from '@/services/api';

// Import the same DEFAULT_API_URL from api.ts to ensure consistency
const DEFAULT_API_URL = 'https://cooked-cartridges-thoroughly-chick.trycloudflare.com';
const API_URL = Constants.expoConfig?.extra?.apiUrl || DEFAULT_API_URL;

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

export interface RecordingStatus {
  isRecording: boolean;
  isProcessing: boolean;
  duration?: number;
  error?: string;
}

export interface TranscriptionResult {
  success: boolean;
  transcription?: string;
  query?: string;
  response?: string;
  language?: string;
  speech_url?: string;
  error?: string;
  processing_info?: any;
}

/**
 * AudioRecorderService - A service for recording audio, transcribing it, and getting TTS responses
 * 
 * This service handles:
 * 1. Recording audio in WAV format
 * 2. Sending audio to backend for transcription
 * 3. Getting AI responses with TTS
 * 4. Managing recording status and notifications
 */
class AudioRecorderService {
  private recording: Audio.Recording | null = null;
  private isRecording = false;
  private isProcessing = false;
  private recordingUri: string | null = null;
  private statusListeners: ((status: RecordingStatus) => void)[] = [];

  constructor() {
    this.initializeAudio();
  }

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

  /**
   * Record, transcribe, and get TTS response in one flow
   * @param language The language for transcription and TTS response
   * @returns Promise with the complete result including transcription and TTS
   */
  async recordAndTranscribe(language: string = 'en'): Promise<TranscriptionResult> {
    try {
      // Start recording
      await this.startRecording();
      
      // Return a promise that will be resolved when the user stops recording
      return new Promise((resolve, reject) => {
        const stopAndProcess = async () => {
          try {
            // Stop recording
            const uri = await this.stopRecording();
            if (!uri) {
              throw new Error('No recording URI available');
            }

            // Process the recording
            const result = await this.processRecording(uri, language);
            resolve(result);
          } catch (error) {
            reject(error);
          }
        };

        // For now, we'll expose this method so the UI can call it
        // In a real implementation, you might want to add a timer or button press detection
        (this as any)._stopAndProcess = stopAndProcess;
      });
    } catch (error) {
      console.error('Failed to start recording and transcription flow:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start recording',
      };
    }
  }

  /**
   * Stop the current recording and process it for transcription and TTS
   * This method should be called by the UI when the user wants to stop recording
   */
  async stopRecordingAndProcess(language: string = 'en'): Promise<TranscriptionResult> {
    try {
      // Stop recording
      const uri = await this.stopRecording();
      if (!uri) {
        throw new Error('No recording URI available');
      }

      // Process the recording
      return await this.processRecording(uri, language);
    } catch (error) {
      console.error('Failed to stop recording and process:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to process recording',
      };
    }
  }

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

  /**
   * Upload an existing audio file for transcription and TTS
   * @param uri The URI of the audio file to process
   * @param language The language for transcription and TTS
   * @returns Promise with the transcription and TTS result
   */
  async transcribeAudioFile(uri: string, language: string = 'en'): Promise<TranscriptionResult> {
    return this.processRecording(uri, language);
  }

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

  /**
   * Cancel current recording without processing
   */
  async cancelRecording(): Promise<void> {
    try {
      if (this.recording && this.isRecording) {
        await this.recording.stopAndUnloadAsync();
      }
      
      this.recording = null;
      this.isRecording = false;
      this.isProcessing = false;
      this.recordingUri = null;
      
      this.notifyStatusChange();
      console.log('Recording cancelled');
    } catch (error) {
      console.error('Error cancelling recording:', error);
    }
  }

  /**
   * Clean up resources
   */
  async cleanup(): Promise<void> {
    await this.cancelRecording();
    this.statusListeners = [];
  }
}

// Export a singleton instance
export const audioRecorder = new AudioRecorderService();
// export default AudioRecorderService;
