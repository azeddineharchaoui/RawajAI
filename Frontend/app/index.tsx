import React, { useEffect, useRef } from 'react';
import { StyleSheet, Animated, Easing } from 'react-native';
import { Image } from 'expo-image';
import { useRouter } from 'expo-router';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
// No need for useColorScheme as ThemedComponents handle this internally

export default function SplashScreen() {
  const router = useRouter();
  // Not using colorScheme here since ThemedText/ThemedView handle color schemes internally
  
  // Using useRef instead of state to avoid re-renders
  const opacity = useRef(new Animated.Value(0)).current;
  const scale = useRef(new Animated.Value(0.8)).current;
  
  useEffect(() => {
    Animated.parallel([
      Animated.timing(opacity, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
        easing: Easing.ease,
      }),
      Animated.timing(scale, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
        easing: Easing.elastic(1.3),
      })
    ]).start();
    
    // Navigate to the main screen after a delay
    const timer = setTimeout(() => {
      router.replace('/(tabs)');
    }, 2000);
    
    return () => clearTimeout(timer);
  }, [opacity, scale, router]);
  
  return (
    <ThemedView style={styles.container}>
      <Animated.View style={[styles.logoContainer, { opacity, transform: [{ scale }] }]}>
        <Image
          source={require('@/assets/images/partial-react-logo.png')}
          style={styles.logo}
          contentFit="contain"
        />
        <ThemedText type="title" style={styles.title}>Supply Chain AI</ThemedText>
        <ThemedText style={styles.subtitle}>Intelligent Supply Chain Management</ThemedText>
      </Animated.View>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 24,
  },
  logoContainer: {
    alignItems: 'center',
  },
  logo: {
    width: 120,
    height: 120,
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    opacity: 0.7,
  }
});
