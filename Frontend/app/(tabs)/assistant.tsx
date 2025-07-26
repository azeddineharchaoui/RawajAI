import React, { useState, useRef, useEffect } from 'react';
import { StyleSheet, View, FlatList, KeyboardAvoidingView, Platform, TouchableOpacity, ActivityIndicator, ScrollView } from 'react-native';
import { Stack } from 'expo-router';
import * as Haptics from 'expo-haptics';
import {audioRecorder} from '@/services/AudioRecorderService'; // Import the audio recorder service
// Audio playback is handled by the AudioPlayerService

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Input } from '@/components/ui/Input';
// Using TouchableOpacity instead of Button
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { api } from '@/services/api';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { audioPlayer } from '@/services/AudioPlayerService';

type Message = {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  speechUrl?: string | null;
  isAudioLoading?: boolean;
  isAudioPlaying?: boolean;
  hasPlayedAudio?: boolean;
  audioPlaybackFailed?: boolean;
};




export default function AssistantScreen() {
  const colorScheme = useColorScheme();
  
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: "Hello! I'm your Supply Chain AI Assistant. How can I help you today?",
      isUser: false,
      timestamp: new Date(),
    }
  ]);
  const [loading, setLoading] = useState(false);
  
  const flatListRef = useRef<FlatList<Message>>(null);
  const [isRecording, setIsRecording] = useState(false);
  
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [currentPlayingUrl, setCurrentPlayingUrl] = useState<string | null>(null);
  
  const handleSend = async () => {
    if (!query.trim()) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      text: query,
      isUser: true,
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    const userQuery = query;
    setQuery('');
    setLoading(true);
    
    try {
      // Use TTS version of the endpoint
      const response = await api.askQuestionWithTTS(userQuery);
      
      // Store the speech URL for playback
      const speechUrl = response.speech_url || null;
      setCurrentPlayingUrl(speechUrl);
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.response || "I'm sorry, I couldn't process your request.",
        isUser: false,
        timestamp: new Date(),
        speechUrl: response.speech_url || null,
      };
      
      setMessages(prev => [...prev, botMessage]);
      
      // Auto-play the audio if available, with better error handling
      if (response.speech_url) {
        try {
          console.log(`Playing audio from URL: ${response.speech_url}`);
          setIsSpeaking(true);
          
          // Provide visual feedback that audio is being prepared
          // We'll update the message temporarily to show loading status
          setMessages(prev => prev.map(msg => 
            msg.id === botMessage.id 
              ? { ...msg, isAudioLoading: true }
              : msg
          ));
          
          // Register a one-time playback completion listener with improved monitoring
          const unsubscribe = audioPlayer.addPlaybackListener((isPlaying, url) => {
            if (url === response.speech_url) {
              // Update status based on playback state
              if (!isPlaying) {
                console.log('Audio playback completed or stopped');
                setIsSpeaking(false);
                setCurrentPlayingUrl(null);
                
                // Update message status
                setMessages(prev => prev.map(msg => 
                  msg.id === botMessage.id 
                    ? { ...msg, isAudioLoading: false, hasPlayedAudio: true }
                    : msg
                ));
                
                unsubscribe(); // Remove the listener when done
              } else {
                // Audio is actively playing
                setMessages(prev => prev.map(msg => 
                  msg.id === botMessage.id 
                    ? { ...msg, isAudioLoading: false, isAudioPlaying: true }
                    : msg
                ));
              }
            }
          });
          
          // Start playback with enhanced error handling
          await audioPlayer.playFromUrl(response.speech_url);
          
          // Add haptic feedback when audio starts playing
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        } catch (audioError) {
          console.error('Error playing audio:', audioError);
          setIsSpeaking(false);
          setCurrentPlayingUrl(null);
          
          // Update message to show audio failed
          setMessages(prev => prev.map(msg => 
            msg.id === botMessage.id 
              ? { ...msg, isAudioLoading: false, audioPlaybackFailed: true }
              : msg
          ));
          
          // Show a toast or some UI indication that audio failed
          console.log('Audio playback failed - displaying text response only');
        }
      }
    } catch (error) {
      console.error('Error asking question:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, I encountered an error. Please try again.",
        isUser: false,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
      setCurrentPlayingUrl(null);
    } finally {
      setLoading(false);
    }
  };
  
  const handleMicPress = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    
    // Toggle recording state
    setIsRecording(!isRecording);
    
    if (!isRecording) {
      // Start recording
      try {
        await audioRecorder.startRecording();
        console.log('Recording started');
      } catch (error) {
        console.error('Error starting recording:', error);
        setIsRecording(false);
      }
    } else {
      // Stop recording and process audio
      try {
        setLoading(true);
        
        // Add a message showing what we're doing
        const processingMessage: Message = {
          id: Date.now().toString(),
          text: "Processing your voice message...",
          isUser: false,
          timestamp: new Date(),
        };
        
        setMessages(prev => [...prev, processingMessage]);
        
        // Stop recording and get the audio file URI
        const uri = await audioRecorder.stopRecording();
        
        if (!uri) {
          throw new Error('No recording available');
        }
        
        // Process the recording (transcribe and get TTS response)
        const result = await audioRecorder.processRecording(uri);
        
        if (result.success && result.transcription) {
          // Add user's transcribed message
          const userMessage: Message = {
            id: Date.now().toString(),
            text: result.transcription,
            isUser: true,
            timestamp: new Date(),
          };
          
          // Add AI's response
          const botMessage: Message = {
            id: (Date.now() + 1).toString(),
            text: result.response || "I couldn't understand that. Could you try again?",
            isUser: false,
            timestamp: new Date(),
            speechUrl: result.speech_url,
          };
          
          setMessages(prev => [
            ...prev.filter(m => m.id !== processingMessage.id), // Remove processing message
            userMessage,
            botMessage,
          ]);
          
          // Auto-play the response if available
          if (result.speech_url) {
            setCurrentPlayingUrl(result.speech_url);
            try {
              await audioPlayer.playAudio(result.speech_url);
              setIsSpeaking(true);
            } catch (audioError) {
              console.log('Audio playback failed:', audioError);
            }
          }
        } else {
          throw new Error(result.error || 'Failed to process audio');
        }
      } catch (error) {
        console.error('Error processing audio:', error);
        const errorMessage: Message = {
          id: Date.now().toString(),
          text: "Sorry, I couldn't process your voice message. Please try again.",
          isUser: false,
          timestamp: new Date(),
        };
        setMessages(prev => [
          ...prev.filter(m => m.text !== "Processing your voice message..."),
          errorMessage,
        ]);
      } finally {
        setLoading(false);
        setIsRecording(false);
      }
    }
  };
  
  useEffect(() => {
    // Scroll to bottom when messages change
    if (flatListRef.current) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({ animated: true });
      }, 100);
    }
  }, [messages]);
  
  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      // Stop any playing audio when component unmounts
      audioPlayer.stopPlayback();
      setIsSpeaking(false);
      setCurrentPlayingUrl(null);
    };
  }, []);

  return (
    <ThemedView style={styles.container}>
      <Stack.Screen options={{ title: "AI Assistant", headerShown: true }} />
      
      <FlatList
        data={messages}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.messagesContainer}
        onContentSizeChange={() => {
          if (flatListRef.current && messages.length > 0) {
            flatListRef.current.scrollToEnd({ animated: true });
          }
        }}
        renderItem={({ item }) => (
          <View style={[
            styles.messageBubble,
            item.isUser ? styles.userBubble : styles.botBubble,
            { backgroundColor: item.isUser ? Colors[colorScheme ?? 'light'].tint : colorScheme === 'dark' ? '#2D3133' : '#F1F5F9' }
          ]}>
            <ThemedText style={[
              styles.messageText,
              item.isUser && { color: '#fff' }
            ]}>
              {item.text}
            </ThemedText>
            
            <View style={styles.messageFooter}>
              {!item.isUser && item.speechUrl && (
                <TouchableOpacity 
                  style={[
                    styles.audioButton,
                    currentPlayingUrl === item.speechUrl && isSpeaking && styles.audioButtonActive
                  ]}
                  onPress={async () => {
                    try {
                      // Apply haptic feedback when button is pressed
                      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                      
                      if (currentPlayingUrl === item.speechUrl && isSpeaking) {
                        // Pause current audio
                        await audioPlayer.pause();
                        setIsSpeaking(false);
                      } else if (currentPlayingUrl === item.speechUrl && !isSpeaking) {
                        // Resume current audio
                        await audioPlayer.resume();
                        setIsSpeaking(true);
                      } else {
                        // Play new audio
                        if (item.speechUrl) {
                          // Stop any currently playing audio first
                          await audioPlayer.stopPlayback();
                          
                          // Set up state for new audio
                          setCurrentPlayingUrl(item.speechUrl);
                          setIsSpeaking(true);
                          
                          // Register listener for playback completion
                          const unsubscribe = audioPlayer.addPlaybackListener((isPlaying, url) => {
                            if (!isPlaying && url === item.speechUrl) {
                              setIsSpeaking(false);
                              unsubscribe();
                            }
                          });
                          
                          // Start playback
                          try {
                            await audioPlayer.playFromUrl(item.speechUrl);
                          } catch (error) {
                            console.error('Error playing audio:', error);
                            setIsSpeaking(false);
                            setCurrentPlayingUrl(null);
                            unsubscribe();
                          }
                        }
                      }
                    } catch (error) {
                      console.error('Error handling audio playback:', error);
                      setIsSpeaking(false);
                      setCurrentPlayingUrl(null);
                    }
                  }}
                >
                  <View style={styles.audioButtonContent}>
                    <IconSymbol
                      name={currentPlayingUrl === item.speechUrl && isSpeaking ? "pause" : "play"}
                      size={16}
                      color={Colors[colorScheme ?? 'light'].text}
                    />
                    {currentPlayingUrl === item.speechUrl && isSpeaking && (
                      <ThemedText style={styles.audioPlayingText}>
                        Playing
                      </ThemedText>
                    )}
                  </View>
                </TouchableOpacity>
              )}
              <ThemedText style={styles.timestamp}>
                {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </ThemedText>
            </View>
          </View>
        )}
      />
      
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={100}
        style={styles.inputContainer}
      >
        <View style={styles.inputRow}>
          <Input
            placeholder="Ask me anything about supply chain..."
            value={query}
            onChangeText={setQuery}
            multiline
            numberOfLines={1}
            containerStyle={styles.input}
            inputStyle={styles.inputText}
          />
          
          <TouchableOpacity
            style={[
              styles.micButton,
              isRecording && styles.micButtonRecording,
              { backgroundColor: isRecording ? '#E53E3E' : Colors[colorScheme ?? 'light'].tint }
            ]}
            onPress={handleMicPress}
          >
            <IconSymbol
              size={22}
              name={isRecording ? "stop.fill" : "mic.fill"}
              color="#fff"
            />
          </TouchableOpacity>
          
          <TouchableOpacity
            style={[
              styles.sendButton,
              { backgroundColor: query.trim() ? Colors[colorScheme ?? 'light'].tint : (colorScheme === 'dark' ? '#2D3133' : '#E2E8F0') },
              query.trim() ? {} : styles.sendButtonDisabled,
            ]}
            onPress={handleSend}
            disabled={!query.trim() || loading}
          >
            {loading ? (
              <ActivityIndicator color="#fff" size="small" />
            ) : (
              <IconSymbol
                size={22}
                name="arrow.up.circle.fill"
                color={query.trim() ? "#fff" : (colorScheme === 'dark' ? '#9BA1A6' : '#A0AEC0')}
              />
            )}
          </TouchableOpacity>
        </View>
        
        <View style={styles.suggestionsContainer}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <SuggestionChip
              text="Current stock levels"
              onPress={() => setQuery("What are the current stock levels for smartphones?")}
            />
            <SuggestionChip
              text="Optimize inventory"
              onPress={() => setQuery("How can I optimize inventory for product X?")}
            />
            <SuggestionChip
              text="Demand forecast"
              onPress={() => setQuery("Show me demand forecast for next month")}
            />
            <SuggestionChip
              text="Supply chain risks"
              onPress={() => setQuery("What are the main risks in my supply chain?")}
            />
          </ScrollView>
        </View>
      </KeyboardAvoidingView>
    </ThemedView>
  );
}

interface SuggestionChipProps {
  text: string;
  onPress: () => void;
}

const SuggestionChip = ({ text, onPress }: SuggestionChipProps) => {
  const colorScheme = useColorScheme();
  // Access colors using Colors[colorScheme ?? 'light']
  
  return (
    <TouchableOpacity
      style={[
        styles.suggestionChip,
        { backgroundColor: colorScheme === 'dark' ? '#2D3133' : '#F1F5F9' }
      ]}
      onPress={() => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        onPress();
      }}
    >
      <ThemedText style={styles.suggestionText}>{text}</ThemedText>
    </TouchableOpacity>
  );
};





const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  messagesContainer: {
    padding: 16,
    paddingBottom: 80,
  },
  messageBubble: {
    padding: 16,
    borderRadius: 16,
    marginVertical: 8,
    maxWidth: '80%',
  },
  userBubble: {
    alignSelf: 'flex-end',
    borderBottomRightRadius: 4,
  },
  botBubble: {
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 4,
  },
  messageText: {
    fontSize: 16,
  },
  messageFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-end',
    marginTop: 8,
  },
  audioButton: {
    padding: 6,
    borderRadius: 16,
    backgroundColor: 'rgba(150,150,150,0.2)',
    marginRight: 10,
  },
  audioButtonActive: {
    backgroundColor: 'rgba(0,122,255,0.2)',
  },
  audioButtonContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  audioPlayingText: {
    fontSize: 10,
    marginLeft: 4,
  },
  timestamp: {
    fontSize: 12,
    opacity: 0.5,
    marginTop: 2,
  },
  inputContainer: {
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(150,150,150,0.2)',
    padding: 16,
    paddingTop: 12,
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  input: {
    flex: 1,
    marginBottom: 0,
    marginRight: 8,
  },
  inputText: {
    maxHeight: 100,
  },
  micButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  micButtonRecording: {
    transform: [{ scale: 1.1 }],
  },
  sendButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonDisabled: {
    opacity: 0.7,
  },
  suggestionsContainer: {
    marginTop: 12,
  },
  suggestionChip: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 16,
    marginRight: 8,
  },
  suggestionText: {
    fontSize: 14,
  },
});
