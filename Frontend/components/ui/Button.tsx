import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, StyleProp, ViewStyle } from 'react-native';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

export interface ButtonProps {
  text?: string;
  onPress: () => void;
  type?: 'primary' | 'secondary' | 'danger' | 'success';
  loading?: boolean;
  disabled?: boolean;
  fullWidth?: boolean;
  icon?: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  children?: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  text,
  onPress,
  type = 'primary',
  loading = false,
  disabled = false,
  fullWidth = false,
  icon,
  style: customStyle,
  children,
}) => {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];

  const buttonStyles = {
    primary: {
      backgroundColor: colors.tint,
      textColor: '#fff',
    },
    secondary: {
      backgroundColor: colorScheme === 'dark' ? '#2C2F33' : '#E1E8ED',
      textColor: colorScheme === 'dark' ? '#ECEDEE' : '#11181C',
    },
    danger: {
      backgroundColor: '#E53E3E',
      textColor: '#fff',
    },
    success: {
      backgroundColor: '#38A169',
      textColor: '#fff',
    },
  };

  const style = buttonStyles[type];

  return (
    <TouchableOpacity
      onPress={onPress}
      disabled={loading || disabled}
      style={[
        styles.button,
        { backgroundColor: style.backgroundColor },
        fullWidth && styles.fullWidth,
        disabled && styles.disabled,
        customStyle
      ]}
    >
      {loading ? (
        <ActivityIndicator color="#fff" />
      ) : (
        <View style={styles.contentContainer}>
          {icon && <View style={styles.iconContainer}>{icon}</View>}
          {text ? (
            <Text style={[styles.text, { color: style.textColor }]}>{text}</Text>
          ) : (
            <>
              {typeof children === 'string' ? (
                <Text style={[styles.text, { color: style.textColor }]}>{children}</Text>
              ) : (
                children
              )}
            </>
          )}
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  button: {
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 120,
  },
  fullWidth: {
    width: '100%',
  },
  disabled: {
    opacity: 0.5,
  },
  text: {
    fontWeight: '600',
    fontSize: 16,
  },
  contentContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconContainer: {
    marginRight: 8,
  },
});
