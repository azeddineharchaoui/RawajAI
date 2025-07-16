import React, { useState, useEffect } from 'react';
import { StyleSheet, View, ScrollView, ActivityIndicator, TouchableOpacity, Alert, Switch } from 'react-native';
import { Stack } from 'expo-router';
import * as Haptics from 'expo-haptics';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { api, getTunnelUrl, saveTunnelUrl, clearTunnelUrl } from '@/services/apiClient';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from '@/components/ui/IconSymbol';

export default function SettingsScreen() {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];
  
  const [tunnelUrl, setTunnelUrl] = useState('');
  const [customApiUrl, setCustomApiUrl] = useState('');
  const [tunnelStatus, setTunnelStatus] = useState<{status: string, url?: string} | null>(null);
  const [language, setLanguage] = useState('en');
  const [loading, setLoading] = useState(true);
  const [testingConnection, setTestingConnection] = useState(false);
  const [useCustomEndpoint, setUseCustomEndpoint] = useState(false);
  
  // Languages available in the API
  const languages = [
    { code: 'en', name: 'English' },
    { code: 'fr', name: 'French' },
    { code: 'ar', name: 'Arabic' }
  ];

  useEffect(() => {
    const loadSettings = async () => {
      try {
        // Load saved tunnel URL
        const savedTunnelUrl = await getTunnelUrl();
        if (savedTunnelUrl) {
          setTunnelUrl(savedTunnelUrl);
          setUseCustomEndpoint(true);
        }
        
        // Fetch current tunnel status
        const status = await api.getTunnelStatus();
        setTunnelStatus(status);
      } catch (error) {
        console.error('Error loading settings:', error);
      } finally {
        setLoading(false);
      }
    };
    
    loadSettings();
  }, []);
  
  const startTunnel = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setLoading(true);
    try {
      const result = await api.startTunnel();
      setTunnelStatus(result);
      if (result.url) {
        setTunnelUrl(result.url);
        await saveTunnelUrl(result.url);
      }
      Alert.alert('Success', 'Cloudflare tunnel started successfully');
    } catch (error) {
      console.error('Error starting tunnel:', error);
      Alert.alert('Error', 'Failed to start Cloudflare tunnel');
    } finally {
      setLoading(false);
    }
  };
  
  const stopTunnel = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setLoading(true);
    try {
      await api.stopTunnel();
      setTunnelStatus({ status: 'stopped' });
      Alert.alert('Success', 'Cloudflare tunnel stopped successfully');
    } catch (error) {
      console.error('Error stopping tunnel:', error);
      Alert.alert('Error', 'Failed to stop Cloudflare tunnel');
    } finally {
      setLoading(false);
    }
  };
  
  const testConnection = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setTestingConnection(true);
    try {
      if (useCustomEndpoint && customApiUrl) {
        await saveTunnelUrl(customApiUrl);
        setTunnelUrl(customApiUrl);
      }
      
      const result = await api.testTunnel();
      if (result.status === 'ok') {
        Alert.alert('Success', 'Connection to API successful');
      } else {
        Alert.alert('Error', 'Could not connect to API');
      }
    } catch (error) {
      console.error('Error testing connection:', error);
      Alert.alert('Error', 'Failed to connect to API');
    } finally {
      setTestingConnection(false);
    }
  };
  
  const clearSettings = async () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    Alert.alert(
      'Clear Settings',
      'Are you sure you want to clear all settings? This will reset your API connection.',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            await clearTunnelUrl();
            setTunnelUrl('');
            setCustomApiUrl('');
            setUseCustomEndpoint(false);
            Alert.alert('Success', 'Settings cleared successfully');
          },
        },
      ]
    );
  };
  
  const handleLanguageChange = (lang: string) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setLanguage(lang);
    // In a real app, you'd save this preference and apply it globally
  };
  
  const toggleCustomEndpoint = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setUseCustomEndpoint(!useCustomEndpoint);
    if (!useCustomEndpoint && tunnelUrl) {
      setCustomApiUrl(tunnelUrl);
    }
  };

  if (loading) {
    return (
      <ThemedView style={styles.loadingContainer}>
        <ActivityIndicator size="large" color={colors.tint} />
        <ThemedText style={styles.loadingText}>Loading settings...</ThemedText>
      </ThemedView>
    );
  }

  return (
    <ThemedView style={styles.container}>
      <Stack.Screen options={{ title: "Settings", headerShown: true }} />
      
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.contentContainer}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>API Connection</ThemedText>
        
        <Card style={styles.card}>
          <View style={styles.settingRow}>
            <ThemedText style={styles.settingLabel}>Status</ThemedText>
            <View style={styles.statusContainer}>
              <View 
                style={[
                  styles.statusIndicator, 
                  tunnelStatus?.status === 'running' 
                    ? styles.statusActive 
                    : styles.statusInactive
                ]} 
              />
              <ThemedText>
                {tunnelStatus?.status === 'running' ? 'Connected' : 'Disconnected'}
              </ThemedText>
            </View>
          </View>
          
          {tunnelStatus?.status === 'running' && tunnelStatus?.url && (
            <View style={styles.settingRow}>
              <ThemedText style={styles.settingLabel}>URL</ThemedText>
              <ThemedText style={styles.urlText} numberOfLines={1}>{tunnelStatus.url}</ThemedText>
            </View>
          )}
          
          <View style={styles.settingRow}>
            <ThemedText style={styles.settingLabel}>Custom API Endpoint</ThemedText>
            <Switch
              value={useCustomEndpoint}
              onValueChange={toggleCustomEndpoint}
              trackColor={{ false: '#767577', true: colors.tint }}
            />
          </View>
          
          {useCustomEndpoint && (
            <View style={styles.customUrlContainer}>
              <Input
                label="API URL"
                placeholder="Enter custom API URL"
                value={customApiUrl}
                onChangeText={setCustomApiUrl}
              />
            </View>
          )}
          
          <View style={styles.buttonRow}>
            <Button
              text="Test Connection"
              onPress={testConnection}
              loading={testingConnection}
            />
            
            {tunnelStatus?.status === 'running' ? (
              <Button
                text="Stop Tunnel"
                onPress={stopTunnel}
                loading={loading}
                type="danger"
              />
            ) : (
              <Button
                text="Start Tunnel"
                onPress={startTunnel}
                loading={loading}
                type="success"
              />
            )}
          </View>
        </Card>
        
        <ThemedText type="subtitle" style={styles.sectionTitle}>Language Preferences</ThemedText>
        
        <Card style={styles.card}>
          <ThemedText style={styles.settingDescription}>
            Select your preferred language for interactions with the AI assistant and reports.
          </ThemedText>
          
          <View style={styles.languageContainer}>
            {languages.map((lang) => (
              <TouchableOpacity
                key={lang.code}
                style={[
                  styles.languageOption,
                  language === lang.code && styles.languageSelected
                ]}
                onPress={() => handleLanguageChange(lang.code)}
              >
                <ThemedText 
                  style={[
                    styles.languageText,
                    language === lang.code && styles.languageTextSelected
                  ]}>
                  {lang.name}
                </ThemedText>
              </TouchableOpacity>
            ))}
          </View>
        </Card>
        
        <ThemedText type="subtitle" style={styles.sectionTitle}>Advanced</ThemedText>
        
        <Card style={styles.card}>
          <TouchableOpacity style={styles.settingItem} onPress={clearSettings}>
            <View style={styles.settingContent}>
              <View style={styles.iconContainer}>
                <IconSymbol size={24} name="trash.fill" color="#E53E3E" />
              </View>
              <View>
                <ThemedText style={styles.settingItemTitle}>Clear Settings</ThemedText>
                <ThemedText style={styles.settingItemDescription}>
                  Reset all settings and connection information
                </ThemedText>
              </View>
            </View>
            <IconSymbol size={20} name="chevron.right" color={colors.text} />
          </TouchableOpacity>
          
          <View style={styles.separator} />
          
          <TouchableOpacity style={styles.settingItem}>
            <View style={styles.settingContent}>
              <View style={styles.iconContainer}>
                <IconSymbol size={24} name="info.circle.fill" color={colors.tint} />
              </View>
              <View>
                <ThemedText style={styles.settingItemTitle}>About</ThemedText>
                <ThemedText style={styles.settingItemDescription}>
                  Supply Chain AI Agent v1.0.0
                </ThemedText>
              </View>
            </View>
            <IconSymbol size={20} name="chevron.right" color={colors.text} />
          </TouchableOpacity>
        </Card>
      </ScrollView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
  },
  scrollView: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 32,
  },
  sectionTitle: {
    marginVertical: 16,
  },
  card: {
    padding: 16,
    marginBottom: 16,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  settingLabel: {
    fontSize: 16,
  },
  settingDescription: {
    marginBottom: 16,
    opacity: 0.8,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusIndicator: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  statusActive: {
    backgroundColor: '#38A169',
  },
  statusInactive: {
    backgroundColor: '#E53E3E',
  },
  urlText: {
    maxWidth: '60%',
    fontSize: 12,
    opacity: 0.8,
  },
  customUrlContainer: {
    marginBottom: 16,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  halfButton: {
    flex: 1,
    marginHorizontal: 4,
  },
  startButton: {
    backgroundColor: '#38A169',
  },
  stopButton: {
    backgroundColor: '#E53E3E',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    textAlign: 'center',
    fontSize: 14,
  },
  languageContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 8,
  },
  languageOption: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 20,
    marginRight: 8,
    marginBottom: 8,
    backgroundColor: 'rgba(150,150,150,0.1)',
  },
  languageSelected: {
    backgroundColor: Colors.light.tint,
  },
  languageText: {
    fontSize: 14,
  },
  languageTextSelected: {
    color: '#fff',
    fontWeight: 'bold',
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
  },
  settingContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(150,150,150,0.1)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  settingItemTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  settingItemDescription: {
    fontSize: 12,
    opacity: 0.6,
    marginTop: 4,
  },
  separator: {
    height: 1,
    backgroundColor: 'rgba(150,150,150,0.2)',
    marginVertical: 8,
  },
});
