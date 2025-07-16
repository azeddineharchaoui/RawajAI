import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, TextInput, TouchableOpacity, Alert } from 'react-native';
import { Stack } from 'expo-router';
import * as DocumentPicker from 'expo-document-picker';
import * as FileSystem from 'expo-file-system';
import * as Haptics from 'expo-haptics';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { api } from '@/services/apiClient';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

export default function DocumentsScreen() {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];
  
  const [documentText, setDocumentText] = useState('');
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<{id: string; name: string; preview: string}[]>([
    {id: '1', name: 'Supply Chain Policy.pdf', preview: 'This document outlines the company\'s supply chain policies...'},
    {id: '2', name: 'Vendor Agreement.docx', preview: 'Standard terms and conditions for vendor relationships...'},
    {id: '3', name: 'Logistics Manual.pdf', preview: 'Procedures for logistics operations including shipping...'},
  ]);

  const uploadDocument = async () => {
    if (!documentText.trim()) {
      Alert.alert('Error', 'Please enter document text');
      return;
    }

    setLoading(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    
    try {
      await api.addDocument(documentText);
      
      // Add document to local state
      const newDoc = {
        id: Date.now().toString(),
        name: `Document ${documents.length + 1}.txt`,
        preview: documentText.substring(0, 100) + '...',
      };
      
      setDocuments([...documents, newDoc]);
      setDocumentText('');
      
      Alert.alert('Success', 'Document uploaded and processed successfully');
    } catch (error) {
      console.error('Error uploading document:', error);
      Alert.alert('Error', 'Failed to upload document');
    } finally {
      setLoading(false);
    }
  };

  const pickDocument = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: ['application/pdf', 'text/plain', 'application/msword', 
               'application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
        copyToCacheDirectory: true,
      });
      
      if (!result.canceled && result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        
        // For demonstration, we'll just show the file name
        // In a real app, you'd upload the file to the server
        Alert.alert('Document Selected', `File name: ${asset.name}`);
        
        // If it's a text file, we could try to read it
        if (asset.name.endsWith('.txt')) {
          try {
            const text = await FileSystem.readAsStringAsync(asset.uri);
            setDocumentText(text);
          } catch (e) {
            console.error('Error reading file:', e);
          }
        } else {
          // For other file types (PDF, DOC), we'd need server-side processing
          // Here we just show a placeholder
          setDocumentText(`[Content from ${asset.name} would be processed on the server]`);
        }
      }
    } catch (error) {
      console.error('Document picking error:', error);
    }
  };

  const deleteDocument = (id: string) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    
    Alert.alert(
      'Delete Document',
      'Are you sure you want to delete this document?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: () => {
            // In a real app, you'd call an API to delete the document
            setDocuments(documents.filter(doc => doc.id !== id));
          },
        },
      ]
    );
  };

  return (
    <ThemedView style={styles.container}>
      <Stack.Screen options={{ title: "Knowledge Base", headerShown: true }} />
      
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.contentContainer}>
        <ThemedText type="subtitle">Add Documents to Knowledge Base</ThemedText>
        
        <Card style={styles.uploadCard}>
          <ThemedText style={styles.label}>Document Text</ThemedText>
          <TextInput
            style={[
              styles.textarea, 
              { 
                color: colors.text,
                backgroundColor: colorScheme === 'dark' ? '#1A1D1E' : '#F9FAFB',
                borderColor: colorScheme === 'dark' ? '#2D3133' : '#D1D5DB',
              }
            ]}
            placeholder="Enter or paste document text here..."
            placeholderTextColor={colorScheme === 'dark' ? '#9CA3AF' : '#6B7280'}
            multiline={true}
            numberOfLines={8}
            textAlignVertical="top"
            value={documentText}
            onChangeText={setDocumentText}
          />
          
          <View style={styles.buttonRow}>
            <Button 
              text="Pick Document"
              onPress={pickDocument}
              type="secondary"
            />
            <Button
              text="Upload Document"
              onPress={uploadDocument}
              loading={loading}
            />
          </View>
        </Card>
        
        <ThemedText type="subtitle" style={styles.sectionTitle}>Knowledge Base Documents</ThemedText>
        
        {documents.length === 0 ? (
          <Card style={styles.emptyCard}>
            <ThemedText style={styles.emptyText}>No documents in the knowledge base yet</ThemedText>
          </Card>
        ) : (
          documents.map(doc => (
            <Card key={doc.id} style={styles.documentCard}>
              <View style={styles.documentHeader}>
                <View style={styles.iconContainer}>
                  <IconSymbol size={24} name="doc.text.fill" color={colors.tint} />
                </View>
                <View style={styles.documentInfo}>
                  <ThemedText style={styles.documentTitle}>{doc.name}</ThemedText>
                  <ThemedText style={styles.documentPreview} numberOfLines={2}>
                    {doc.preview}
                  </ThemedText>
                </View>
                <TouchableOpacity 
                  style={styles.deleteButton}
                  onPress={() => deleteDocument(doc.id)}
                >
                  <IconSymbol size={20} name="trash.fill" color="#E53E3E" />
                </TouchableOpacity>
              </View>
            </Card>
          ))
        )}
      </ScrollView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 32,
  },
  uploadCard: {
    padding: 16,
    marginVertical: 16,
  },
  label: {
    fontSize: 16,
    marginBottom: 8,
  },
  textarea: {
    minHeight: 120,
    borderWidth: 1,
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 16,
  },
  buttonRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  pickButton: {
    flex: 1,
    marginRight: 8,
  },
  uploadButton: {
    flex: 1,
    marginLeft: 8,
  },
  sectionTitle: {
    marginTop: 24,
    marginBottom: 16,
  },
  emptyCard: {
    padding: 24,
    alignItems: 'center',
  },
  emptyText: {
    opacity: 0.6,
  },
  documentCard: {
    marginBottom: 12,
    padding: 16,
  },
  documentHeader: {
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
  documentInfo: {
    flex: 1,
  },
  documentTitle: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  documentPreview: {
    marginTop: 4,
    fontSize: 14,
    opacity: 0.7,
  },
  deleteButton: {
    padding: 8,
  }
});
