import { Audio } from 'expo-av';
import { getTunnelUrl } from '@/services/api';
import { Platform } from 'react-native';
import Constants from 'expo-constants';

// Import the same DEFAULT_API_URL from api.ts to ensure consistency
// Import the same DEFAULT_API_URL from api.ts to ensure consistency
const DEFAULT_API_URL = 'https://mandatory-indian-thin-computers.trycloudflare.com';
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

/**
 * AudioPlayerService - A service for handling audio playback from the API
 */
class AudioPlayerService {
  private sound: Audio.Sound | null = null;
  private isPlaying = false;
  private isLoading = false;
  private currentUrl: string | null = null;

  /**
   * Play audio from a given URL
   * @param url The URL of the audio to play
   * @returns A promise that resolves when the audio starts playing
   */
  async playAudio(url: string): Promise<void> {
    return this.playFromUrl(url);
  }

  /**
   * Play audio from a given URL
   * @param url The URL of the audio to play
   * @returns A promise that resolves when the audio starts playing
   */
  async playFromUrl(url: string): Promise<void> {
    try {
      // If already playing the same audio, toggle pause/play
      if (this.sound && this.currentUrl === url) {
        const status = await this.sound.getStatusAsync();
        if (status.isLoaded) {
          if (status.isPlaying) {
            await this.sound.pauseAsync();
            this.isPlaying = false;
          } else {
            await this.sound.playAsync();
            this.isPlaying = true;
          }
          this.notifyListeners();
          return;
        }
      }

      // Clean up any existing sound
      await this.unloadSound();

      this.isLoading = true;
      this.currentUrl = url;
      // Notify listeners that we're loading
      this.notifyListeners();

      // Get the base URL for the API
      const baseUrl = await getBaseUrl();
      const fullUrl = url.startsWith('http') ? url : `${baseUrl}${url}`;

      // Set up audio mode for better playback
      await Audio.setAudioModeAsync({
        playsInSilentModeIOS: true,
        staysActiveInBackground: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      // Load and play the sound with enhanced retry and progressive loading
      console.log(`Loading audio from: ${fullUrl}`);
      
      // Implement retry logic for loading audio with progressive buffering
      let retryCount = 0;
      let sound = null;
      
      while (retryCount < 5) { // Up to 5 retry attempts
        try {
          // Handle potential CORS or network issues
          const audioType = fullUrl.endsWith('.mp3') ? 'mp3' : 'mpeg';
          
          console.log(`Attempt ${retryCount + 1} to load audio...`);
          
          // Add timestamp to URL to avoid caching issues
          const timestampedUrl = `${fullUrl}${fullUrl.includes('?') ? '&' : '?'}_t=${Date.now()}`;
          
          // Use a more robust loading configuration with explicit request headers
          const soundData = await Audio.Sound.createAsync(
            { 
              uri: timestampedUrl,
              headers: {
                'Accept': `audio/${audioType}, audio/*;q=0.8`,
                'Cache-Control': 'no-cache',
                'X-Requested-With': 'XMLHttpRequest'
              }
            },
            { 
              shouldPlay: true, 
              volume: 1.0, 
              progressUpdateIntervalMillis: 200,
              positionMillis: 0,
              rate: 1.0,
              isMuted: false
            },
            (status) => {
              this.onPlaybackStatusUpdate(status);
              
              // Enhanced monitoring for network and buffering issues
              if (status.isLoaded) {
                if (status.isBuffering) {
                  console.log(`Audio buffering: ${status.positionMillis}/${status.durationMillis} ms`);
                  
                  // If buffering takes too long, we might want to log it
                  // This could be used to trigger a retry in the future
                } else if (status.isPlaying && status.positionMillis > 0) {
                  // Successfully playing past the beginning
                  console.log(`Audio playing: ${status.positionMillis}/${status.durationMillis} ms`);
                }
                
                // Track playback progress
                if (status.positionMillis && status.durationMillis) {
                  const progress = status.positionMillis / status.durationMillis;
                  if (progress > 0.95) {
                    console.log('Almost finished playback');
                  }
                }
              }
            }
          );
          
          sound = soundData.sound;
          break;
        } catch (loadError) {
          retryCount++;
          console.warn(`Error loading audio (attempt ${retryCount}/5):`, loadError);
          
          if (retryCount >= 5) {
            throw loadError;
          }
          
          // Exponential backoff for retries
          const waitTime = Math.min(1000 * Math.pow(1.5, retryCount - 1), 5000);
          await new Promise(resolve => setTimeout(resolve, waitTime));
        }
      }

      if (!sound) {
        throw new Error('Failed to load audio after multiple attempts');
      }
      
      this.sound = sound;
      this.isPlaying = true;
      this.isLoading = false;
      this.notifyListeners();
      
      // Monitor playback to detect network issues
      this.sound.setOnPlaybackStatusUpdate(status => {
        this.onPlaybackStatusUpdate(status);
        
        // Detect stalled playback
        if (status.isLoaded && status.isPlaying && status.isBuffering) {
          console.log('Audio is buffering...');
        }
      });
    } catch (error) {
      console.error('Error playing audio:', error);
      this.isLoading = false;
      this.isPlaying = false;
      this.currentUrl = null;
      this.notifyListeners();
      throw error;
    }
  }

  // Event callbacks for audio status changes
  private playbackListeners: ((isPlaying: boolean, url: string | null) => void)[] = [];
  
  /**
   * Register a listener for playback status updates
   * @param listener Callback function that receives playback status
   * @returns Function to unregister the listener
   */
  public addPlaybackListener(listener: (isPlaying: boolean, url: string | null) => void): () => void {
    this.playbackListeners.push(listener);
    // Call the listener immediately with current state
    try {
      listener(this.isPlaying, this.currentUrl);
    } catch (error) {
      console.error('Error in initial playback listener call:', error);
    }
    
    return () => {
      this.playbackListeners = this.playbackListeners.filter(l => l !== listener);
    };
  }
  
  /**
   * Notify all listeners of playback status changes
   */
  private notifyListeners(): void {
    this.playbackListeners.forEach(listener => {
      try {
        listener(this.isPlaying, this.currentUrl);
      } catch (error) {
        console.error('Error in playback listener:', error);
      }
    });
  }
  
  /**
   * Callback for audio playback status updates
   */
  private onPlaybackStatusUpdate = (status: any) => {
    if (status.isLoaded) {
      const wasPlaying = this.isPlaying;
      this.isPlaying = status.isPlaying;
      
      // If playback ended or playback state changed, notify listeners
      if (status.didJustFinish || wasPlaying !== this.isPlaying) {
        if (status.didJustFinish) {
          this.isPlaying = false;
          console.log('Audio playback finished');
        }
        this.notifyListeners();
      }
    }
  };

  /**
   * Pause the current playback
   */
  async pause(): Promise<void> {
    if (this.sound && this.isPlaying) {
      await this.sound.pauseAsync();
      this.isPlaying = false;
    }
  }

  /**
   * Resume the current playback
   */
  async resume(): Promise<void> {
    if (this.sound && !this.isPlaying) {
      await this.sound.playAsync();
      this.isPlaying = true;
    }
  }

  /**
   * Stop playback and unload the sound
   */
  async stopPlayback(): Promise<void> {
    await this.unloadSound();
    this.isPlaying = false;
    this.currentUrl = null;
  }

  /**
   * Unload the current sound from memory
   */
  private async unloadSound(): Promise<void> {
    if (this.sound) {
      try {
        await this.sound.unloadAsync();
      } catch (e) {
        console.warn('Error unloading sound:', e);
      }
      this.sound = null;
    }
  }

  /**
   * Check if audio is currently playing
   */
  isCurrentlyPlaying(): boolean {
    return this.isPlaying;
  }

  /**
   * Check if audio is currently loading
   */
  isCurrentlyLoading(): boolean {
    return this.isLoading;
  }
}

// Export a singleton instance
export const audioPlayer = new AudioPlayerService();
