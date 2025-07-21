import React from 'react'
import { ActivityIndicator, StyleSheet, View, ScrollView, TouchableOpacity, ImageBackground, Dimensions } from 'react-native';
import { useState, useEffect } from 'react';
import { ThemedText } from '@/components/ThemedText';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { api } from '@/services/api';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { Link, Link2Off, LucideLink2Off } from 'lucide-react-native';
import { red } from 'react-native-reanimated/lib/typescript/Colors';

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
    useEffect(() => { startTunnel() }, [])
    return (
        <>
            <View style={styles.circles}>
                {loading ? (
                    <ActivityIndicator size="small" color={colors.tint} style={styles.loader} />
                ) : tunnelStatus?.status === 'already_running' ? (
                    <Link color={"green"} />
                ) : (
                    <LucideLink2Off color={"red"} onPress={startTunnel} />
                )}
            </View>
        </>
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
    circles: {
        justifyContent: 'center',
        alignItems: 'center',
        width: 70,
        height: 70,
        borderRadius: "90px",
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
    }
});


export default ApiConeection