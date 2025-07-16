import React, { useState, useRef, useEffect } from 'react';
import { StyleSheet, View, FlatList, KeyboardAvoidingView, Platform, TouchableOpacity, ActivityIndicator, ScrollView } from 'react-native';
import { Stack } from 'expo-router';
import * as Haptics from 'expo-haptics';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { api } from '@/services/api';
import { IconSymbol } from '@/components/ui/IconSymbol';

type Message = {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
};

export default function AssistantScreen() {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];
  
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
  
  const flatListRef = useRef<FlatList>(null);
  const [isRecording, setIsRecording] = useState(false);
  
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
      const response = await api.askQuestion(userQuery);
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: response.response || "I'm sorry, I couldn't process your request.",
        isUser: false,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error asking question:', error);
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "Sorry, I encountered an error. Please try again.",
        isUser: false,
        timestamp: new Date(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };
  
  const handleMicPress = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    
    // Toggle recording state
    setIsRecording(!isRecording);
    
    if (!isRecording) {
      // Start recording logic would go here
      console.log('Start recording');
    } else {
      // Stop recording and process audio
      console.log('Stop recording');
      
      try {
        // This is a placeholder - in a real app, you'd get the audio file and upload it
        setLoading(true);
        const audioFile = { uri: 'file:///path/to/audio.m4a', name: 'audio.m4a', type: 'audio/m4a' };
        
        // Add a message showing what we're doing
        const processingMessage: Message = {
          id: Date.now().toString(),
          text: "Processing your voice message...",
          isUser: false,
          timestamp: new Date(),
        };
        
        setMessages(prev => [...prev, processingMessage]);
        
        // const response = await api.uploadAudio(audioFile);
        
        // In a real implementation, you'd process the response here
      } catch (error) {
        console.error('Error processing audio:', error);
      } finally {
        setLoading(false);
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

  return (
    <ThemedView style={styles.container}>
      <Stack.Screen options={{ title: "AI Assistant", headerShown: true }} />
      
      <FlatList
        ref={flatListRef}
        data={messages}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.messagesContainer}
        renderItem={({ item }) => (
          <View style={[
            styles.messageBubble,
            item.isUser ? styles.userBubble : styles.botBubble,
            { backgroundColor: item.isUser ? colors.tint : colorScheme === 'dark' ? '#2D3133' : '#F1F5F9' }
          ]}>
            <ThemedText style={[
              styles.messageText,
              item.isUser && { color: '#fff' }
            ]}>
              {item.text}
            </ThemedText>
            <ThemedText style={styles.timestamp}>
              {item.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </ThemedText>
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
              { backgroundColor: isRecording ? '#E53E3E' : colors.tint }
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
              { backgroundColor: query.trim() ? colors.tint : (colorScheme === 'dark' ? '#2D3133' : '#E2E8F0') },
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
  const colors = Colors[colorScheme ?? 'light'];
  
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
  timestamp: {
    fontSize: 12,
    opacity: 0.5,
    alignSelf: 'flex-end',
    marginTop: 4,
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
