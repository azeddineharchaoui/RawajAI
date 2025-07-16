import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { IconSymbol } from './IconSymbol';

type AlertType = 'info' | 'success' | 'warning' | 'error';

interface AlertProps {
  type: AlertType;
  title?: string;
  message: string;
}

export const Alert = ({ type, title, message }: AlertProps) => {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];

  const alertStyles = {
    info: {
      backgroundColor: colorScheme === 'dark' ? '#1E3A8A' : '#DBEAFE',
      iconName: 'info.circle.fill',
      iconColor: '#2563EB',
      titleColor: colorScheme === 'dark' ? '#BFDBFE' : '#1E40AF',
      textColor: colorScheme === 'dark' ? '#93C5FD' : '#1E3A8A',
    },
    success: {
      backgroundColor: colorScheme === 'dark' ? '#064E3B' : '#D1FAE5',
      iconName: 'checkmark.circle.fill',
      iconColor: '#10B981',
      titleColor: colorScheme === 'dark' ? '#6EE7B7' : '#065F46',
      textColor: colorScheme === 'dark' ? '#34D399' : '#064E3B',
    },
    warning: {
      backgroundColor: colorScheme === 'dark' ? '#78350F' : '#FEF3C7',
      iconName: 'exclamationmark.triangle.fill',
      iconColor: '#F59E0B',
      titleColor: colorScheme === 'dark' ? '#FCD34D' : '#92400E',
      textColor: colorScheme === 'dark' ? '#FBBF24' : '#78350F',
    },
    error: {
      backgroundColor: colorScheme === 'dark' ? '#7F1D1D' : '#FEE2E2',
      iconName: 'xmark.circle.fill',
      iconColor: '#EF4444',
      titleColor: colorScheme === 'dark' ? '#FECACA' : '#991B1B',
      textColor: colorScheme === 'dark' ? '#F87171' : '#7F1D1D',
    },
  };

  const style = alertStyles[type];

  return (
    <View style={[styles.container, { backgroundColor: style.backgroundColor }]}>
      <View style={styles.iconContainer}>
        <IconSymbol name={style.iconName} size={24} color={style.iconColor} />
      </View>
      <View style={styles.textContainer}>
        {title && (
          <Text style={[styles.title, { color: style.titleColor }]}>{title}</Text>
        )}
        <Text style={[styles.message, { color: style.textColor }]}>{message}</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    borderRadius: 8,
    padding: 16,
    marginVertical: 8,
  },
  iconContainer: {
    marginRight: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  textContainer: {
    flex: 1,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  message: {
    fontSize: 14,
    lineHeight: 20,
  },
});
