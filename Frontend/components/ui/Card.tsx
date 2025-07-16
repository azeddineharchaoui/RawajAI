import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

interface CardProps {
  title?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  style?: object;
  variant?: 'default' | 'elevated' | 'bordered';
}

export const Card = ({
  title,
  children,
  footer,
  style,
  variant = 'default',
}: CardProps) => {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];

  const cardStyles = {
    default: {
      backgroundColor: colorScheme === 'dark' ? '#1A1D1E' : '#FFFFFF',
      borderWidth: 0,
      shadowOpacity: 0,
    },
    elevated: {
      backgroundColor: colorScheme === 'dark' ? '#1A1D1E' : '#FFFFFF',
      borderWidth: 0,
      shadowOpacity: 0.1,
    },
    bordered: {
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderColor: colorScheme === 'dark' ? '#2D3133' : '#E1E8ED',
      shadowOpacity: 0,
    },
  };

  const variantStyle = cardStyles[variant];

  return (
    <View
      style={[
        styles.card,
        {
          backgroundColor: variantStyle.backgroundColor,
          borderWidth: variantStyle.borderWidth,
          borderColor: variantStyle.borderColor,
        },
        variant === 'elevated' && styles.elevated,
        style,
      ]}
    >
      {title && (
        <Text style={[styles.title, { color: colors.text }]}>
          {title}
        </Text>
      )}
      <View style={styles.content}>{children}</View>
      {footer && (
        <View style={[styles.footer, { borderTopColor: colorScheme === 'dark' ? '#2D3133' : '#E1E8ED' }]}>
          {footer}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    borderRadius: 12,
    overflow: 'hidden',
    marginVertical: 8,
  },
  elevated: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowRadius: 8,
    elevation: 4,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    padding: 16,
    paddingBottom: 8,
  },
  content: {
    padding: 16,
  },
  footer: {
    padding: 16,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#E1E8ED',
  },
});
