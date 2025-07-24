"use client"

import React from "react"
import { TouchableOpacity, type TouchableOpacityProps, Platform } from "react-native"

interface SafeTouchableOpacityProps extends TouchableOpacityProps {
  children: React.ReactNode
}

export const SafeTouchableOpacity: React.FC<SafeTouchableOpacityProps> = ({ style, children, ...props }) => {
  // Fix deprecated properties
  const fixedStyle = React.useMemo(() => {
    if (!style || typeof style !== "object") return style

    const styleArray = Array.isArray(style) ? style : [style]

    return styleArray.map((s) => {
      if (!s || typeof s !== "object") return s

      const fixedS = { ...s }

      // Convert shadow* to boxShadow on web
      if (Platform.OS === "web") {
        const shadowProps = ["shadowColor", "shadowOffset", "shadowOpacity", "shadowRadius"]
        const hasShadow = shadowProps.some((prop) => prop in fixedS)

        if (hasShadow) {
          const shadowColor = fixedS.shadowColor || "#000"
          const shadowOffset = fixedS.shadowOffset || { width: 0, height: 2 }
          const shadowOpacity = fixedS.shadowOpacity || 0.25
          const shadowRadius = fixedS.shadowRadius || 3.84

          const alpha = Math.round(shadowOpacity * 255)
            .toString(16)
            .padStart(2, "0")
          // @ts-ignore
          fixedS.boxShadow = `${shadowOffset.width}px ${shadowOffset.height}px ${shadowRadius}px ${shadowColor}${alpha}`

          shadowProps.forEach((prop) => delete fixedS[prop])
        }
      }

      return fixedS
    })
  }, [style])

  // Fix pointerEvents prop
  const fixedProps = { ...props }
  if ("pointerEvents" in fixedProps && Platform.OS === "web") {
    const currentStyle = Array.isArray(fixedStyle) ? fixedStyle : [fixedStyle]
    const newStyle = [...currentStyle, { pointerEvents: fixedProps.pointerEvents }]
    delete fixedProps.pointerEvents

    return (
      <TouchableOpacity style={newStyle} {...fixedProps}>
        {children}
      </TouchableOpacity>
    )
  }

  return (
    <TouchableOpacity style={fixedStyle} {...fixedProps}>
      {children}
    </TouchableOpacity>
  )
}
