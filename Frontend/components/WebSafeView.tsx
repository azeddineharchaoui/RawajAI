"use client"

import React from "react"
import { View, type ViewProps, Platform } from "react-native"

interface WebSafeViewProps extends ViewProps {
  children: React.ReactNode
}

export const WebSafeView: React.FC<WebSafeViewProps> = ({ style, children, ...props }) => {
  // Fix all deprecated properties for web
  const fixedStyle = React.useMemo(() => {
    if (Platform.OS !== "web" || !style) return style

    const styleArray = Array.isArray(style) ? style : [style]

    return styleArray.map((s) => {
      if (!s || typeof s !== "object") return s

      const fixedS = { ...s }

      // Remove all shadow properties on web
      delete fixedS.shadowColor
      delete fixedS.shadowOffset
      delete fixedS.shadowOpacity
      delete fixedS.shadowRadius
      delete fixedS.elevation

      return fixedS
    })
  }, [style])

  // Fix pointerEvents prop for web
  const fixedProps = { ...props }
  if ("pointerEvents" in fixedProps && Platform.OS === "web") {
    const currentStyle = Array.isArray(fixedStyle) ? fixedStyle : [fixedStyle]
    const newStyle = [...currentStyle, { pointerEvents: fixedProps.pointerEvents }]
    delete fixedProps.pointerEvents

    return (
      <View style={newStyle} {...fixedProps}>
        {children}
      </View>
    )
  }

  return (
    <View style={fixedStyle} {...fixedProps}>
      {children}
    </View>
  )
}
