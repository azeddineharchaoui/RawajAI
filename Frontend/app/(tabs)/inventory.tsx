import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, ActivityIndicator, TouchableOpacity } from 'react-native';
import { Stack } from 'expo-router';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Card } from '@/components/ui/Card';
import InventoryItem from '@/components/Inventory/InventoryItem';
import { api } from '@/services/apiClient';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

import OptimizeInventorySection from '@/components/Inventory/RenderOptimizeContent';

export default function InventoryScreen() {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];

  const [productId, setProductId] = useState('');
  const [holdingCost, setHoldingCost] = useState('10');
  const [orderingCost, setOrderingCost] = useState('100');
  const [leadTime, setLeadTime] = useState('7');
  const [serviceLevel, setServiceLevel] = useState('0.95');
  const [loading, setLoading] = useState(false);
  interface OptimizationResult {
    eoq?: number;
    reorder_point?: number;
    safety_stock?: number;
    total_cost?: number;
    chart_data?: any;
  }
  const [optimizationData, setOptimizationData] = useState<OptimizationResult | null>(null);
  const [error, setError] = useState('');
  const [plotHtml, setPlotHtml] = useState('');
  const [activeTab, setActiveTab] = useState('optimize'); // 'optimize' or 'view'

  const optimizeInventory = async () => {
    if (!productId) {
      setError('Please enter a product ID');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await api.optimizeInventory({
        forecast_data: [], // This would come from a previous forecast
        product_id: productId,
        holding_cost: parseFloat(holdingCost) || 10,
        ordering_cost: parseFloat(orderingCost) || 100,
        lead_time: parseInt(leadTime) || 7,
        service_level: parseFloat(serviceLevel) || 0.95,
      });

      setOptimizationData(response);

      // If we have chart data, create a simple plot
      if (response.chart_data) {
        const htmlContent = generatePlotHtml(response.chart_data);
        setPlotHtml(htmlContent);
      }
    } catch (error) {
      console.error('Inventory optimization error:', error);
      setError('Failed to optimize inventory. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const generatePlotHtml = (chartData: any) => {
    // This creates a simple Plotly chart
    return `
      <!DOCTYPE html>
      <html>
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width, initial-scale=1.0">
          <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
          <style>
            body { margin: 0; padding: 0; background: ${colorScheme === 'dark' ? '#151718' : '#fff'}; }
            #chart { width: 100%; height: 100%; }
          </style>
        </head>
        <body>
          <div id="chart"></div>
          <script>
            const data = ${JSON.stringify(chartData || [])};
            const layout = {
              margin: { t: 10, r: 10, l: 50, b: 50 },
              paper_bgcolor: '${colorScheme === 'dark' ? '#151718' : '#fff'}',
              plot_bgcolor: '${colorScheme === 'dark' ? '#151718' : '#fff'}',
              font: { color: '${colorScheme === 'dark' ? '#ECEDEE' : '#11181C'}' },
              xaxis: { showgrid: true, gridcolor: 'rgba(150,150,150,0.1)' },
              yaxis: { showgrid: true, gridcolor: 'rgba(150,150,150,0.1)' },
              showlegend: true,
              legend: { bgcolor: 'rgba(0,0,0,0)' }
            };
            Plotly.newPlot('chart', data, layout, { responsive: true });
          </script>
        </body>
      </html>
    `;
  };
  //  Fake inventory items for demonstration
  interface InventoryItemType {
    id: string;
    name: string;
    quantity: number;
    status: 'ok' | 'low' | 'out';
  }
  const fakeInventoryItems: InventoryItemType[] = [
    {
      id: 'SKU-1001',
      name: 'Product A',
      quantity: 120,
      status: 'ok',
    },
    {
      id: 'SKU-1002',
      name: 'Product B',
      quantity: 42,
      status: 'low',
    },
    {
      id: 'SKU-1003',
      name: 'Product C',
      quantity: 0,
      status: 'out',
    },
  ];

  const renderTabs = () => (
    <View style={styles.tabContainer}>
      <TouchableOpacity
        style={[styles.tab, activeTab === 'optimize' && styles.activeTab]}
        onPress={() => setActiveTab('optimize')}>
        <ThemedText
          style={[
            styles.tabText,
            activeTab === 'optimize' && styles.activeTabText
          ]}>
          Optimize
        </ThemedText>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.tab, activeTab === 'view' && styles.activeTab]}
        onPress={() => setActiveTab('view')}>
        <ThemedText
          style={[
            styles.tabText,
            activeTab === 'view' && styles.activeTabText
          ]}>
          Current Stock
        </ThemedText>
      </TouchableOpacity>
    </View>
  );


  const renderViewContent = (item: InventoryItemType[]) => (
    <Card style={styles.inventoryCard}>
      <ThemedText type="subtitle">Current Inventory</ThemedText>

      {loading ? (
        <ActivityIndicator size="large" color={colors.tint} style={styles.loader} />
      ) : (
        <View style={styles.inventoryList}>
          {/* This would be populated from an API call to get inventory */}
          {item.map((item) => (
            <InventoryItem
              key={item.id}
              id={item.id}
              name={item.name}
              quantity={item.quantity}
              status={item.status}
            />
          ))}
        </View>
      )}
    </Card>
  );

  return (
    <ThemedView style={styles.container}>
      <Stack.Screen options={{ title: "Inventory Management", headerShown: true }} />

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.contentContainer}>
        <ThemedText type="subtitle">Inventory Optimization</ThemedText>

        {renderTabs()}

        {activeTab === 'optimize' ? <OptimizeInventorySection productId={productId} holdingCost={holdingCost} orderingCost={orderingCost} leadTime={leadTime} serviceLevel={serviceLevel} error={error} loading={loading} plotHtml={plotHtml} optimizationData={optimizationData}
          setProductId={setProductId}
          setHoldingCost={setHoldingCost}
          setOrderingCost={setOrderingCost}
          setLeadTime={setLeadTime}
          setServiceLevel={setServiceLevel}
          optimizeInventory={() => optimizeInventory()}
          // used fake data for demonstration
        /> : renderViewContent(fakeInventoryItems)}
      </ScrollView>
    </ThemedView>
  );
}

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
