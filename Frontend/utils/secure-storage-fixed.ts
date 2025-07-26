import * as SecureStore from "expo-secure-store"
import { Platform } from "react-native"

// Version CORRIGÉE du secure storage
export const secureStorage = {
  async getItemAsync(key: string): Promise<string | null> {
    try {
      if (Platform.OS === "web") {
        return localStorage.getItem(key)
      }
      // Utiliser la bonne méthode pour expo-secure-store
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
}

// Export par défaut
export default secureStorage
