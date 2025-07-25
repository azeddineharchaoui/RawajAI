import { Colors } from "@/constants/Colors";
import { ThemedText } from "../ThemedText";
import { IconSymbol } from "../ui/IconSymbol";
import { View, StyleSheet, useColorScheme } from "react-native";

type InventoryStatus = 'ok' | 'low' | 'out';

interface InventoryItemProps {
    id: string;
    name: string;
    quantity: number;
    status: InventoryStatus;
}

const STATUS_CONFIG = {
    ok: {
        color: '#38A169', // green
        label: 'In Stock',
        icon: "checkmark.circle.fill"
    },
    low: {
        color: '#DD6B20', // orange
        label: 'Low Stock',
        icon: "exclamationmark.circle.fill"
    },
    out: {
        color: '#E53E3E', // red
        label: 'Out of Stock',
        icon: "xmark.circle.fill"
    }
} as const;

const InventoryItem = ({ id, name, quantity, status }: InventoryItemProps) => {
    const colorScheme = useColorScheme();
    const colors = Colors[colorScheme ?? 'light'];
    const statusConfig = STATUS_CONFIG[status];

    return (
        <View style={[styles.inventoryItem, { borderBottomColor: 'rgba(150,150,150,0.2)' }]}>
            <View style={styles.itemContent}>
                <ThemedText style={styles.itemName} numberOfLines={1} ellipsizeMode="tail">
                    {name}
                </ThemedText>
                <ThemedText style={[styles.itemId, { opacity: 0.6 }]}>
                    #{id}
                </ThemedText>
            </View>

            <View style={styles.itemContent}>
                <ThemedText style={styles.itemQuantity}>
                    {quantity}
                </ThemedText>
                <View style={[
                    styles.itemStatus,
                    {
                        backgroundColor: `${statusConfig.color}20`,
                        borderColor: statusConfig.color
                    }
                ]}>
                    <IconSymbol
                        size={12}
                        name={statusConfig.icon}
                        color={statusConfig.color}
                    />
                    <ThemedText style={[styles.itemStatusText, { color: statusConfig.color }]}>
                        {statusConfig.label}
                    </ThemedText>
                </View>
            </View>
        </View>
    );
};

const styles = StyleSheet.create({
    inventoryItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingVertical: 12,
        borderBottomWidth: StyleSheet.hairlineWidth,
    },
    itemContent: {
        justifyContent: 'center',
        flexShrink: 1,
    },
    itemName: {
        fontWeight: '600',
        fontSize: 16,
        maxWidth: 200,
    },
    itemId: {
        fontSize: 12,
        marginTop: 4,
    },
    itemQuantity: {
        fontSize: 16,
        fontWeight: '600',
        textAlign: 'right',
    },
    itemStatus: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 12,
        marginTop: 4,
        alignSelf: 'flex-end',
        borderWidth: StyleSheet.hairlineWidth,
        gap: 4,
    },
    itemStatusText: {
        fontSize: 12,
        fontWeight: '500',
    },
});

export default InventoryItem;