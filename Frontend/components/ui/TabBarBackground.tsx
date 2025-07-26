import { Platform } from "react-native"
import { View } from "react-native"
import { useSafeAreaInsets } from "react-native-safe-area-context"

// Hook requis par le système de navigation
export function useBottomTabOverflow() {
  const insets = useSafeAreaInsets()
  return insets.bottom
}

export default function TabBarBackground() {
  const insets = useSafeAreaInsets()

  if (Platform.OS === "web") {
    // Version web simplifiée sans shadow warnings
    return (
      <View
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: "rgba(255, 255, 255, 0.95)",
          borderTopWidth: 1,
          borderTopColor: "#e0e0e0",
        }}
      />
    )
  }

  // Version mobile avec support des insets
  return (
    <View
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: -insets.bottom,
        backgroundColor: "rgba(255, 255, 255, 0.95)",
      }}
    />
  )
}
