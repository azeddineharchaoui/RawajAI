import { Image } from 'expo-image';
import { StyleSheet, View, TouchableOpacity } from 'react-native';

import { ThemedText } from '@/components/ThemedText';
import { Card } from '@/components/ui/Card';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

// Stat Card Component
const StatCard = ({ title, value, icon, color = '#38A169', onPress }: {
    title: string;
    value: string;
    icon: number|string;
    color?: string;
    onPress: () => void;
}) => {
    const colorScheme = useColorScheme();
    const colors = Colors[colorScheme ?? 'light'];

    return (
        <TouchableOpacity onPress={onPress}>
            <Card style={styles.statCard}>
                <View style={[styles.iconContainer, { backgroundColor: color + '20' }]}>
                    <View style={styles.icon}>
                        {/* Using a colored View since we can't easily colorize IconSymbol */}
                        <Image source={icon} style={{ width: 50, height: 50 }} />
                    </View>
                </View>
                <ThemedText type="defaultSemiBold" style={styles.statValue}>{value}</ThemedText>
                <ThemedText style={styles.statTitle}>{title}</ThemedText>
            </Card>
        </TouchableOpacity>
    );
};

export default StatCard;


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
