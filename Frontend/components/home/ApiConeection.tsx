import React from 'react'
import { ActivityIndicator, StyleSheet, View, ScrollView, TouchableOpacity, ImageBackground, Dimensions } from 'react-native';
import { useState, useEffect } from 'react';
import { ThemedText } from '@/components/ThemedText';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { api } from '@/services/api';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

const ApiConeection = () => {
    const [loading, setLoading] = useState(true);
    const colorScheme = useColorScheme() as 'light' | 'dark';
    const colors = Colors[colorScheme ?? 'light'];
    const [tunnelStatus, setTunnelStatus] = useState<{ status: string, url?: string } | null>(null);

    const startTunnel = async () => {
        setLoading(true);
        try {
            const result = await api.startTunnel();
            console.log('Tunnel started:', result);
            setTunnelStatus(result);
        } catch (error) {
            console.log('Error starting tunnel:', error);
        } finally {
            setLoading(false);
        }
    };
    useEffect(()=>{startTunnel()},[])
    return (
        <Card style={styles.connectionCard}>
            <ThemedText type="subtitle">API Connection Status</ThemedText>
            {loading ? (
                <ActivityIndicator size="small" color={colors.tint} style={styles.loader} />
            ) : tunnelStatus?.status === 'already_running' ? (
                <>
                    <View style={styles.statusContainer}>
                        <View style={[styles.statusIndicator, styles.statusActive]} />
                        <ThemedText>Connected to API</ThemedText>
                    </View>
                    <ThemedText style={styles.urlText} numberOfLines={1}>
                        {tunnelStatus.url}
                    </ThemedText>
                </>
            ) : (
                <>
                    <View style={styles.statusContainer}>
                        <View style={[styles.statusIndicator, styles.statusInactive]} />
                        <ThemedText>Not Connected</ThemedText>
                    </View>
                    <Button
                        text="Connect to API"
                        onPress={startTunnel}
                        loading={loading}
                        style={styles.connectButton}
                    />
                </>
            )}
        </Card>
    )
}

const styles = StyleSheet.create({
    logo: {
        flex: 1,
        width: '100%',
        height: "100%",
    },
    connectionCard: {
        marginBottom: 16,
        padding: 16,
    },
    loader: {
        marginTop: 8,
    },
    statusContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 8,
        marginBottom: 4,
    },
    statusIndicator: {
        width: 10,
        height: 10,
        borderRadius: 5,
        marginRight: 8,
    },
    statusActive: {
        backgroundColor: '#38A169',
    },
    statusInactive: {
        backgroundColor: '#E53E3E',
    },
    urlText: {
        fontSize: 12,
        opacity: 0.7,
    },
    connectButton: {
        marginTop: 8,
    },
    cardsScroll: {
        marginHorizontal: -16,
        paddingHorizontal: 16,
        marginBottom: 16,
    },
    statCard: {
        width: 120,
        height: 120,
        padding: 12,
        marginRight: 12,
        alignItems: 'center',
        justifyContent: 'center',
    },
    iconContainer: {
        width: 40,
        height: 40,
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 8,
    },
    icon: {
        width: 24,
        height: 24,
        alignItems: 'center',
        justifyContent: 'center',
    },
    iconInner: {
        width: 12,
        height: 12,
        borderRadius: 6,
    },
    statValue: {
        fontSize: 24,
        fontWeight: 'bold',
    },
    statTitle: {
        fontSize: 12,
        opacity: 0.7,
        textAlign: 'center',
    },
    sectionContainer: {
        marginBottom: 24,
        gap: 16,
    },
    actionGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginHorizontal: -8,
    },
    actionButton: {
        width: '50%',
        padding: 8,
        alignItems: 'center',
    },
    actionIcon: {
        width: 56,
        height: 56,
        borderRadius: 28,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 8,
    },
    actionTitle: {
        textAlign: 'center',
        fontSize: 14,
    },
    activityItem: {
        flexDirection: 'row',
        paddingVertical: 12,
        borderBottomWidth: StyleSheet.hairlineWidth,
        borderBottomColor: 'rgba(150, 150, 150, 0.2)',
    },
    activityDot: {
        width: 10,
        height: 10,
        borderRadius: 5,
        backgroundColor: '#3182CE',
        marginTop: 6,
        marginRight: 12,
    },
    activityContent: {
        flex: 1,
    },
    activityDesc: {
        opacity: 0.7,
        marginTop: 2,
        fontSize: 14,
    },
    activityTime: {
        fontSize: 12,
        opacity: 0.5,
        marginTop: 4,
    },
});


export default ApiConeection