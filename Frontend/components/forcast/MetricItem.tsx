import React from 'react';
import { StyleSheet, View } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

const MetricItem = ({ label, value }: { label: string; value: string | number }) => {
    const theme = useColorScheme();
    const isDark = theme === 'dark';

    return (
        <View style={[styles.metricItem, { backgroundColor: isDark ? '#1f2937' : '#f9fafb' }]}>
            <ThemedText style={[styles.metricLabel, { color: isDark ? '#cbd5e1' : '#334155' }]}>{label}</ThemedText>
            <ThemedText style={[styles.metricValue, { color: isDark ? '#f9fafb' : '#111827' }]}>{value}</ThemedText>
        </View>
    );
};

const styles = StyleSheet.create({
    metricItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 12,
        paddingHorizontal: 16,
        marginBottom: 10,
        borderRadius: 10,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 3,
        elevation: 2,
    },
    metricLabel: {
        fontSize: 16,
        fontWeight: '500',
    },
    metricValue: {
        fontSize: 16,
        fontWeight: '700',
    },
});

export default MetricItem;
