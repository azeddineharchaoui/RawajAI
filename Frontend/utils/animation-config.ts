import { Platform } from "react-native"

// Animation configuration compatible with web
export const getAnimationConfig = (useNativeDriver = true) => ({
  useNativeDriver: Platform.OS !== "web" && useNativeDriver,
})

// Helper for common animations
export const createAnimation = (config: any) => ({
  ...config,
  useNativeDriver: Platform.OS !== "web" && config.useNativeDriver !== false,
})
