import React from 'react';
import { View, StyleSheet, useColorScheme } from 'react-native';
import { WebView } from 'react-native-webview';
import { Colors } from '@/constants/Colors';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { ThemedText } from '@/components/ThemedText';

type LocationProductBreakdown = {
    smartphone: number;
    laptop: number;
    tablet: number;
    headphones: number;
    smartwatch: number;
};

type LocationDetails = {
    capacity: number;
    capacity_utilization: number;
    total_inventory: number;
    products: LocationProductBreakdown;
};
interface optimization_results {
    status: string;
    total_cost: number;
    eoq?: number;
    reorder_point?: number;
    safety_stock?: number;
    locations?: {
        warehouse_a: LocationDetails;
        warehouse_b: LocationDetails;
        warehouse_c: LocationDetails;
    };
}

interface OptimizationData {
    optimization_results: optimization_results
    eoq?: number;
    reorder_point?: number;
    safety_stock?: number;
    total_cost?: number;
    recommendations?: string;
}

interface OptimizeInventorySectionProps {
    productId: string;
    holdingCost: string;
    orderingCost: string;
    leadTime: string;
    serviceLevel: string;
    error: string;
    loading: boolean;
    plotHtml: string;
    optimizationData: OptimizationData | null;

    setProductId: (value: string) => void;
    setHoldingCost: (value: string) => void;
    setOrderingCost: (value: string) => void;
    setLeadTime: (value: string) => void;
    setServiceLevel: (value: string) => void;
    optimizeInventory: () => void;
}

const OptimizeInventorySection: React.FC<OptimizeInventorySectionProps> = ({
    productId,
    holdingCost,
    orderingCost,
    leadTime,
    serviceLevel,
    error,
    loading,
    plotHtml,
    optimizationData,
    setProductId,
    setHoldingCost,
    setOrderingCost,
    setLeadTime,
    setServiceLevel,
    optimizeInventory,
}) => {
    const colorScheme = useColorScheme();
    const colors = Colors[colorScheme ?? 'light'];
    // console.log("gdhagdjhasgdjhasgdhgas");

    console.log(optimizationData);

    return (
        <>
            <Card style={styles.formCard}>
                <Input
                    label="Product ID"
                    placeholder="Enter product ID"
                    value={productId}
                    onChangeText={setProductId}
                    containerStyle={styles.input}
                />

                <View style={styles.rowInputs}>
                    <View style={styles.halfInput}>
                        <Input
                            label="Holding Cost"
                            placeholder="10"
                            value={holdingCost}
                            onChangeText={setHoldingCost}
                            keyboardType="numeric"
                            containerStyle={styles.input}
                        />
                    </View>
                    <View style={styles.halfInput}>
                        <Input
                            label="Ordering Cost"
                            placeholder="100"
                            value={orderingCost}
                            onChangeText={setOrderingCost}
                            keyboardType="numeric"
                            containerStyle={styles.input}
                        />
                    </View>
                </View>

                <View style={styles.rowInputs}>
                    <View style={styles.halfInput}>
                        <Input
                            label="Lead Time (days)"
                            placeholder="7"
                            value={leadTime}
                            onChangeText={setLeadTime}
                            keyboardType="numeric"
                            containerStyle={styles.input}
                        />
                    </View>
                    <View style={styles.halfInput}>
                        <Input
                            label="Service Level (0-1)"
                            placeholder="0.95"
                            value={serviceLevel}
                            onChangeText={setServiceLevel}
                            keyboardType="numeric"
                            containerStyle={styles.input}
                        />
                    </View>
                </View>

                {error ? <ThemedText style={styles.errorText}>{error}</ThemedText> : null}

                <Button
                    text="Optimize Inventory"
                    onPress={optimizeInventory}
                    loading={loading}
                    fullWidth
                />
            </Card>

            {optimizationData && (
                <Card style={styles.resultsCard}>
                    <ThemedText type="subtitle">Optimization Results</ThemedText>

                    <ResultRow label="Economic Order Quantity" value={`${optimizationData.eoq?.toFixed(2) || '-'}`} unit="units" />
                    <ResultRow label="Reorder Point" value={`${optimizationData.reorder_point?.toFixed(2) || '-'}`} unit="units" />
                    <ResultRow label="Safety Stock" value={`${optimizationData.safety_stock?.toFixed(2) || '-'}`} unit="units" />
                    <ResultRow label="Total Cost" value={`$${optimizationData.optimization_results.total_cost?.toFixed(2) || '-'}`} />

                    <ThemedText
                        type="subtitle"
                        style={{
                            marginTop: 16,
                            fontWeight: 'bold',
                            fontSize: 16,
                            color: colors.tint,
                        }}
                    >
                        Recommendations
                    </ThemedText>

                    {/* Styled recommendation content */}
                    <ThemedText
                        style={{
                            marginTop: 8,
                            fontSize: 19,
                            lineHeight: 30,
                            color: '#555',
                        }}
                    >
                        {optimizationData.recommendations || 'No recommendations available.'}
                    </ThemedText>
                </Card>
            )}

            {plotHtml && (
                <Card style={styles.chartCard}>
                    <ThemedText type="subtitle">Inventory Level Projection</ThemedText>
                    <View style={styles.chartContainer}>
                        <WebView
                            source={{ html: plotHtml }}
                            style={styles.webView}
                            scrollEnabled={false}
                            javaScriptEnabled
                            domStorageEnabled
                            onError={(e) => console.error('WebView error:', e.nativeEvent)}
                            renderError={(errorName) => (
                                <View style={styles.errorContainer}>
                                    <ThemedText style={styles.errorText}>Error loading chart: {errorName}</ThemedText>
                                </View>
                            )}
                        />
                    </View>
                </Card>
            )}
        </>
    );
};

const ResultRow = ({
    label,
    value,
    unit,
}: {
    label: string;
    value: string;
    unit?: string;
}) => (
    <View style={styles.resultRow}>
        <ThemedText style={styles.resultLabel}>{label}:</ThemedText>
        <ThemedText style={styles.resultValue}>{value} {unit || ''}</ThemedText>
    </View>
);

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    scrollView: {
        flex: 1,
    },
    contentContainer: {
        padding: 16,
        paddingBottom: 32,
    },
    tabContainer: {
        flexDirection: 'row',
        marginBottom: 16,
    },
    tab: {
        flex: 1,
        paddingVertical: 12,
        alignItems: 'center',
        borderBottomWidth: 2,
        borderBottomColor: 'transparent',
    },
    activeTab: {
        borderBottomColor: Colors.light.tint,
    },
    tabText: {
        fontSize: 16,
    },
    activeTabText: {
        fontWeight: 'bold',
        color: Colors.light.tint,
    },
    formCard: {
        marginBottom: 16,
        padding: 16,
    },
    input: {
        marginBottom: 12,
    },
    rowInputs: {
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    halfInput: {
        width: '48%',
    },
    button: {
        marginTop: 8,
    },
    errorText: {
        color: '#E53E3E',
        marginVertical: 8,
    },
    resultsCard: {
        marginBottom: 16,
        padding: 16,
    },
    resultRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginVertical: 8,
    },
    resultLabel: {
        opacity: 0.8,
    },
    resultValue: {
        fontWeight: 'bold',
    },
    chartCard: {
        marginBottom: 16,
        padding: 16,
    },
    chartContainer: {
        height: 250,
        marginTop: 8,
    },
    webView: {
        flex: 1,
        backgroundColor: 'transparent',
    },
    inventoryCard: {
        marginBottom: 16,
        padding: 16,
    },
    inventoryList: {
        marginTop: 8,
    },
    inventoryItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: 'rgba(150,150,150,0.2)',
    },
    itemContent: {
        justifyContent: 'center',
    },
    itemName: {
        fontWeight: 'bold',
        fontSize: 16,
    },
    itemId: {
        fontSize: 12,
        opacity: 0.6,
        marginTop: 4,
    },
    itemQuantity: {
        fontSize: 16,
        fontWeight: 'bold',
        textAlign: 'right',
    },
    itemStatus: {
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 12,
        marginTop: 4,
        alignSelf: 'flex-end',
    },
    itemStatusText: {
        fontSize: 12,
    },
    loader: {
        marginVertical: 20,
    },
    errorContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
});

export default OptimizeInventorySection;
