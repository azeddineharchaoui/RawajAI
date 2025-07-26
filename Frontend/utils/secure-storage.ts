import * as SecureStore from "expo-secure-store"
import { Platform } from "react-native"

// Wrapper pour SecureStore compatible web/mobile - VERSION CORRIGÉE
export const secureStorage = {
  async getItemAsync(key: string): Promise<string | null> {
    try {
      if (Platform.OS === "web") {
        return localStorage.getItem(key)
      }
      return await SecureStore.getItemAsync(key)
    } catch (error) {
      console.warn(`Error getting item ${key}:`, error)
      return null
    }
  },

  async setItemAsync(key: string, value: string): Promise<void> {
    try {
      if (Platform.OS === "web") {
        localStorage.setItem(key, value)
        return
      }
      await SecureStore.setItemAsync(key, value)
    } catch (error) {
      console.warn(`Error setting item ${key}:`, error)
    }
  },

  async deleteItemAsync(key: string): Promise<void> {
    try {
      if (Platform.OS === "web") {
        localStorage.removeItem(key)
        return
      }
      await SecureStore.deleteItemAsync(key)
    } catch (error) {
      console.warn(`Error deleting item ${key}:`, error)
    }
  },

  // Alias pour compatibilité avec l'ancien code - CORRIGÉ
  async getValueWithKeyAsync(key: string): Promise<string | null> {
    return this.getItemAsync(key)
  },

  async setValueWithKeyAsync(key: string, value: string): Promise<void> {
    return this.setItemAsync(key, value)
  },
}

// Export par défaut pour compatibilité
export default secureStorage
