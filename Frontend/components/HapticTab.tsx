"use client"

import React from "react"

import type { ReactNode } from "react"
import { Platform } from "react-native"
import { TouchableOpacity, type TouchableOpacityProps } from "react-native"
import * as Haptics from "expo-haptics"

interface HapticTabProps extends TouchableOpacityProps {
  children: ReactNode
}

export function HapticTab({ children, style, onPress, ...props }: HapticTabProps) {
  const handlePress = (event: any) => {
    // Ajouter un feedback haptique sur mobile
    if (Platform.OS !== "web") {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)
    }

    if (onPress) {
      onPress(event)
    }
  }

  // Fix deprecated properties for web
  const fixedStyle = React.useMemo(() => {
    if (Platform.OS === "web" && style && typeof style === "object") {
      const webStyle = Array.isArray(style) ? style : [style]
      return webStyle.map((s) => {
        if (!s || typeof s !== "object") return s

        const fixedS = { ...s }
        // Remove shadow properties on web
        delete fixedS.shadowColor
        delete fixedS.shadowOffset
        delete fixedS.shadowOpacity
        delete fixedS.shadowRadius
        delete fixedS.elevation

        return fixedS
      })
    }
    return style
  }, [style])

  // Fix pointerEvents prop for web
  const fixedProps = { ...props }
  if ("pointerEvents" in fixedProps && Platform.OS === "web") {
    const currentStyle = Array.isArray(fixedStyle) ? fixedStyle : [fixedStyle]
    const newStyle = [...currentStyle, { pointerEvents: fixedProps.pointerEvents }]
    delete fixedProps.pointerEvents

    return (
      <TouchableOpacity style={newStyle} onPress={handlePress} {...fixedProps}>
        {children}
      </TouchableOpacity>
    )
  }

  return (
    <TouchableOpacity style={fixedStyle} onPress={handlePress} {...fixedProps}>
      {children}
    </TouchableOpacity>
  )
}
