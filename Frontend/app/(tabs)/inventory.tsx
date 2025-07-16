import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, ActivityIndicator, TouchableOpacity } from 'react-native';
import { Stack } from 'expo-router';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { api } from '@/services/apiClient';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
// Using IconSymbol for inventory item status icons
import { IconSymbol } from '@/components/ui/IconSymbol';
import { WebView } from 'react-native-webview';

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

  const renderOptimizeContent = () => (
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
      
      {/* Results */}
      {optimizationData && (
        <Card style={styles.resultsCard}>
          <ThemedText type="subtitle">Optimization Results</ThemedText>
          
          <View style={styles.resultRow}>
            <ThemedText style={styles.resultLabel}>Economic Order Quantity:</ThemedText>
            <ThemedText style={styles.resultValue}>
              {optimizationData.eoq?.toFixed(2) || '-'} units
            </ThemedText>
          </View>
          
          <View style={styles.resultRow}>
            <ThemedText style={styles.resultLabel}>Reorder Point:</ThemedText>
            <ThemedText style={styles.resultValue}>
              {optimizationData.reorder_point?.toFixed(2) || '-'} units
            </ThemedText>
          </View>
          
          <View style={styles.resultRow}>
            <ThemedText style={styles.resultLabel}>Safety Stock:</ThemedText>
            <ThemedText style={styles.resultValue}>
              {optimizationData.safety_stock?.toFixed(2) || '-'} units
            </ThemedText>
          </View>
          
          <View style={styles.resultRow}>
            <ThemedText style={styles.resultLabel}>Total Cost:</ThemedText>
            <ThemedText style={styles.resultValue}>
              ${optimizationData.total_cost?.toFixed(2) || '-'}
            </ThemedText>
          </View>
        </Card>
      )}
      
      {/* Chart visualization */}
      {plotHtml ? (
        <Card style={styles.chartCard}>
          <ThemedText type="subtitle">Inventory Level Projection</ThemedText>
          <View style={styles.chartContainer}>
            <WebView
              source={{ html: plotHtml }}
              style={styles.webView}
              scrollEnabled={false}
              javaScriptEnabled={true}
              domStorageEnabled={true}
              onError={(e) => console.error('WebView error:', e.nativeEvent)}
              renderError={(errorName) => (
                <View style={styles.errorContainer}>
                  <ThemedText style={styles.errorText}>Error loading chart: {errorName}</ThemedText>
                </View>
              )}
            />
          </View>
        </Card>
      ) : null}
    </>
  );
  
  const renderViewContent = () => (
    <Card style={styles.inventoryCard}>
      <ThemedText type="subtitle">Current Inventory</ThemedText>
      
      {loading ? (
        <ActivityIndicator size="large" color={colors.tint} style={styles.loader} />
      ) : (
        <View style={styles.inventoryList}>
          {/* This would be populated from an API call to get inventory */}
          <InventoryItem 
            id="SKU-1001" 
            name="Product A" 
            quantity={256} 
            status="ok" 
          />
          <InventoryItem 
            id="SKU-1002" 
            name="Product B" 
            quantity={42} 
            status="low" 
          />
          <InventoryItem 
            id="SKU-1003" 
            name="Product C" 
            quantity={0} 
            status="out" 
          />
          <InventoryItem 
            id="SKU-1004" 
            name="Product D" 
            quantity={128} 
            status="ok" 
          />
          <InventoryItem 
            id="SKU-1005" 
            name="Product E" 
            quantity={18} 
            status="low" 
          />
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
        
        {activeTab === 'optimize' ? renderOptimizeContent() : renderViewContent()}
      </ScrollView>
    </ThemedView>
  );
}

// Inventory Item Component
const InventoryItem = ({ id, name, quantity, status }: { 
  id: string;
  name: string;
  quantity: number;
  status: 'ok' | 'low' | 'out';
}) => {
  const statusColors = {
    ok: '#38A169', // green
    low: '#DD6B20', // orange
    out: '#E53E3E', // red
  };
  
  const statusLabels = {
    ok: 'In Stock',
    low: 'Low Stock',
    out: 'Out of Stock',
  };

  const statusIcons: Record<string, any> = {
    ok: "checkmark.circle.fill",
    low: "exclamationmark.circle.fill",
    out: "xmark.circle.fill",
  };
  
  return (
    <View style={styles.inventoryItem}>
      <View style={styles.itemContent}>
        <ThemedText style={styles.itemName}>{name}</ThemedText>
        <ThemedText style={styles.itemId}>{id}</ThemedText>
      </View>
      <View style={styles.itemContent}>
        <ThemedText style={styles.itemQuantity}>{quantity}</ThemedText>
        <View style={[styles.itemStatus, { backgroundColor: statusColors[status] + '20' }]}>
          <IconSymbol size={12} name={statusIcons[status]} color={statusColors[status]} />
          <ThemedText style={[styles.itemStatusText, { color: statusColors[status] }]}>
            {statusLabels[status]}
          </ThemedText>
        </View>
      </View>
    </View>
  );
};

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
