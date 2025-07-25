import React, { useState, useEffect } from 'react';
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

interface Product {
  product_id: string;
  category: string;
  total_inventory: number;
  avg_cost: number;
  inventory_by_location: Record<string, number>;
  avg_lead_time: number;
}

interface ProductsResponse {
  products: Product[];
  total_count: number;
}

export default function InventoryScreen() {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];
  
  const [productId, setProductId] = useState('');
  const [holdingCost, setHoldingCost] = useState('10');
  const [orderingCost, setOrderingCost] = useState('100');
  const [leadTime, setLeadTime] = useState('7');
  const [serviceLevel, setServiceLevel] = useState('0.95');
  const [loading, setLoading] = useState(false);
  const [products, setProducts] = useState<Product[]>([]);
  const [productsLoading, setProductsLoading] = useState(false);
  interface OptimizationResult {
    eoq?: number;
    reorder_point?: number;
    safety_stock?: number;
    total_cost?: number;
    chart_data?: any;
    recommendations?: string | string[];
    optimization_results?: any;
    product_id?: string;
  }
  const [optimizationData, setOptimizationData] = useState<OptimizationResult | null>(null);
  const [error, setError] = useState('');
  const [plotHtml, setPlotHtml] = useState('');
  const [activeTab, setActiveTab] = useState('optimize'); // 'optimize' or 'view'
  
  // Fetch products from API when the component mounts
  useEffect(() => {
    fetchProducts();
  }, []);
  
  // Reload products when the tab changes to 'view'
  useEffect(() => {
    if (activeTab === 'view') {
      fetchProducts();
    }
  }, [activeTab]);
  
  // Function to fetch products from the API
  const fetchProducts = async () => {
    try {
      setProductsLoading(true);
      const response = await api.getProducts() as ProductsResponse;
      if (response && response.products) {
        setProducts(response.products);
      }
    } catch (error) {
      console.error('Error fetching products:', error);
      setError('Failed to fetch products. Please try again.');
    } finally {
      setProductsLoading(false);
    }
  };

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
            // Process chart data to ensure it's well-formed
            let chartData = ${JSON.stringify(chartData || [])};
            let useYAxis2 = false;
            
            // If the data is empty or not in the expected format, create a default chart
            if (!Array.isArray(chartData) || chartData.length === 0) {
              // Create a bar chart of EOQ, Reorder Point, and Safety Stock
              chartData = [{
                x: ['EOQ', 'Reorder Point', 'Safety Stock'],
                y: [
                  ${optimizationData?.eoq || 0},
                  ${optimizationData?.reorder_point || 0},
                  ${optimizationData?.safety_stock || 0}
                ],
                type: 'bar',
                marker: {
                  color: ['rgba(55, 83, 109, 0.7)', 'rgba(26, 118, 255, 0.7)', 'rgba(56, 161, 105, 0.7)']
                },
                name: 'Inventory Parameters'
              }];
            } else {
              // Check if any trace is using yaxis2
              useYAxis2 = chartData.some(trace => trace.yaxis === 'y2');
              
              // Ensure all traces have proper configuration
              chartData = chartData.map(trace => {
                // Ensure traces have type
                if (!trace.type) {
                  trace.type = 'bar';
                }
                
                // Ensure lines+markers mode for scatter traces
                if (trace.type === 'scatter' && !trace.mode) {
                  trace.mode = 'lines+markers';
                }
                
                // Enhance markers for scatter traces
                if (trace.type === 'scatter' && !trace.marker) {
                  trace.marker = {size: 8};
                }
                
                return trace;
              });
            }
            
            // Check for inventory parameter charts (EOQ, reorder point, safety stock)
            let hasParameterChart = chartData.some(trace => 
              trace.name === 'Inventory Parameters' || 
              (trace.x && Array.isArray(trace.x) && trace.x.includes('EOQ'))
            );
            
            let hasCostChart = chartData.some(trace => 
              trace.name === 'Annual Cost Breakdown' || 
              (trace.x && Array.isArray(trace.x) && trace.x.includes('Holding Cost'))
            );
            
            // Enhanced layout with better defaults and support for multiple charts
            const layout = {
              margin: { t: 40, r: useYAxis2 ? 60 : 30, l: 60, b: hasParameterChart ? 200 : 70 },
              paper_bgcolor: '${colorScheme === 'dark' ? '#151718' : '#fff'}',
              plot_bgcolor: '${colorScheme === 'dark' ? '#151718' : '#fff'}',
              font: { color: '${colorScheme === 'dark' ? '#ECEDEE' : '#11181C'}' },
              grid: hasParameterChart ? { 
                rows: hasCostChart ? 3 : 2, 
                columns: 1,
                roworder: 'top to bottom',
                pattern: 'independent'
              } : undefined,
              xaxis: { 
                showgrid: true, 
                gridcolor: 'rgba(150,150,150,0.1)',
                title: 'Locations',
                domain: [0, 1]
              },
              yaxis: { 
                showgrid: true, 
                gridcolor: 'rgba(150,150,150,0.1)',
                title: 'Inventory Units',
                domain: hasParameterChart ? [0.6, 1] : [0, 1]
              },
              showlegend: true,
              legend: { 
                bgcolor: 'rgba(0,0,0,0)',
                orientation: 'h',
                y: hasParameterChart ? -0.1 : -0.2
              },
              title: {
                text: 'Inventory Optimization',
                font: {
                  size: 18
                }
              }
            };
            
            // Add second y-axis if needed
            if (useYAxis2) {
              layout.yaxis2 = {
                title: 'Utilization %',
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                range: [0, 100]
              };
            }
            
            // Add parameter chart axes if needed
            if (hasParameterChart) {
              layout.xaxis2 = {
                title: 'Inventory Parameters',
                domain: [0, 0.45],
                anchor: 'y3'
              };
              layout.yaxis3 = {
                title: 'Units',
                domain: [0.25, 0.45],
                anchor: 'x2'
              };
            }
            
            // Add cost chart axes if needed
            if (hasCostChart) {
              layout.xaxis3 = {
                title: 'Cost Components',
                domain: [0.55, 1],
                anchor: 'y4'
              };
              layout.yaxis4 = {
                title: 'Annual Cost ($)',
                domain: [0.25, 0.45],
                anchor: 'x3'
              };
            }
            
            Plotly.newPlot('chart', chartData, layout, { responsive: true });
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
        <View style={styles.input}>
          <ThemedText style={{marginBottom: 8, fontWeight: '500'}}>Product ID</ThemedText>
          <View style={styles.selectContainer}>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {products.length > 0 ? (
                products.map((product) => (
                  <TouchableOpacity
                    key={product.product_id}
                    style={[
                      styles.productButton,
                      productId === product.product_id && styles.selectedProduct
                    ]}
                    onPress={() => {
                      setProductId(product.product_id);
                      // Also set the lead time from the product data
                      setLeadTime(product.avg_lead_time.toString());
                    }}
                  >
                    <ThemedText style={[
                      styles.productButtonText,
                      productId === product.product_id && styles.selectedProductText
                    ]}>
                      {product.product_id}
                    </ThemedText>
                  </TouchableOpacity>
                ))
              ) : (
                <Input
                  placeholder="Enter product ID"
                  value={productId}
                  onChangeText={setProductId}
                  containerStyle={{flex: 1}}
                />
              )}
            </ScrollView>
          </View>
        </View>
        
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
      
      {/* Optimization Results */}
      {optimizationData && !loading && (
        <Card style={styles.resultsCard}>
          <ThemedText type="subtitle">Optimization Results</ThemedText>
          
          <View style={styles.resultsGrid}>
            <View style={styles.resultItem}>
              <ThemedText style={styles.resultLabel}>Economic Order Quantity</ThemedText>
              <ThemedText style={styles.resultValue}>
                {optimizationData.eoq ? Math.round(optimizationData.eoq) : 'N/A'} units
              </ThemedText>
            </View>
            
            <View style={styles.resultItem}>
              <ThemedText style={styles.resultLabel}>Reorder Point</ThemedText>
              <ThemedText style={styles.resultValue}>
                {optimizationData.reorder_point ? Math.round(optimizationData.reorder_point) : 'N/A'} units
              </ThemedText>
            </View>
            
            <View style={styles.resultItem}>
              <ThemedText style={styles.resultLabel}>Safety Stock</ThemedText>
              <ThemedText style={styles.resultValue}>
                {optimizationData.safety_stock ? Math.round(optimizationData.safety_stock) : 'N/A'} units
              </ThemedText>
            </View>
            
            <View style={styles.resultItem}>
              <ThemedText style={styles.resultLabel}>Annual Total Cost</ThemedText>
              <ThemedText style={styles.resultValue}>
                ${optimizationData.total_cost ? Math.round(optimizationData.total_cost).toLocaleString() : 'N/A'}
              </ThemedText>
            </View>
          </View>
          
          {/* Recommendations Section */}
          {optimizationData.recommendations && (
            <View style={styles.recommendationsContainer}>
              <ThemedText style={styles.recommendationsTitle}>Recommendations</ThemedText>
              
              <View style={styles.recommendationsList}>
                {typeof optimizationData.recommendations === 'string' ? (
                  optimizationData.recommendations.split('\n').map((recommendation, index) => (
                    <View key={index} style={styles.recommendationItem}>
                      <ThemedText style={styles.recommendationText}>
                        • {recommendation.trim()}
                      </ThemedText>
                    </View>
                  ))
                ) : Array.isArray(optimizationData.recommendations) ? (
                  optimizationData.recommendations.map((recommendation, index) => (
                    <View key={index} style={styles.recommendationItem}>
                      <ThemedText style={styles.recommendationText}>
                        • {recommendation}
                      </ThemedText>
                    </View>
                  ))
                ) : null}
              </View>
            </View>
          )}
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
              originWhitelist={['*']}
              onError={(e) => console.error('WebView error:', e.nativeEvent)}
              onHttpError={(e) => console.error('WebView HTTP error:', e.nativeEvent)}
              renderError={(errorName) => (
                <View style={styles.errorContainer}>
                  <ThemedText style={styles.errorText}>Error loading chart: {errorName}</ThemedText>
                  <Button 
                    text="Reload" 
                    onPress={() => optimizeInventory()} 
                    style={{marginTop: 10}}
                  />
                </View>
              )}
              startInLoadingState={true}
              renderLoading={() => (
                <ActivityIndicator size="small" color={colors.tint} />
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
      
      {productsLoading ? (
        <ActivityIndicator size="large" color={colors.tint} style={styles.loader} />
      ) : (
        <View style={styles.inventoryList}>
          {products.length > 0 ? (
            products.map((product) => {
              // Determine the status based on inventory level
              let status: 'ok' | 'low' | 'out' = 'ok';
              if (product.total_inventory <= 0) {
                status = 'out';
              } else if (product.total_inventory < 100) { // Assuming less than 100 is "low"
                status = 'low';
              }
              
              return (
                <InventoryItem 
                  key={product.product_id}
                  id={product.product_id}
                  name={`${product.product_id.charAt(0).toUpperCase() + product.product_id.slice(1)}`}
                  quantity={product.total_inventory}
                  status={status}
                  category={product.category}
                />
              );
            })
          ) : error ? (
            <ThemedText style={styles.errorText}>{error}</ThemedText>
          ) : (
            <ThemedText style={{textAlign: 'center', marginVertical: 20, opacity: 0.6}}>No products found</ThemedText>
          )}
        </View>
      )}
      
      <Button
        text="Refresh Products"
        onPress={fetchProducts}
        fullWidth
        style={{ marginTop: 16 }}
      />
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
const InventoryItem = ({ id, name, quantity, status, category }: { 
  id: string;
  name: string;
  quantity: number;
  status: 'ok' | 'low' | 'out';
  category?: string;
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
        {category && <ThemedText style={styles.itemCategory}>{category}</ThemedText>}
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
  resultsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    marginTop: 12,
  },
  resultItem: {
    width: '48%',
    marginBottom: 16,
    padding: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.03)',
    borderRadius: 8,
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginVertical: 8,
  },
  resultLabel: {
    fontSize: 14,
    marginBottom: 4,
    opacity: 0.8,
  },
  resultValue: {
    fontSize: 16,
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
  noDataText: {
    textAlign: 'center',
    marginVertical: 20,
    opacity: 0.6,
  },
  itemCategory: {
    fontSize: 12,
    opacity: 0.6,
    marginTop: 2,
    fontStyle: 'italic',
  },
  recommendationsContainer: {
    marginTop: 16,
    borderTopWidth: 1,
    borderTopColor: 'rgba(150,150,150,0.2)',
    paddingTop: 16,
  },
  recommendationsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  recommendationsList: {
    marginTop: 8,
  },
  recommendationDivider: {
    height: 1,
    backgroundColor: 'rgba(150,150,150,0.2)',
    marginVertical: 15,
  },
  recommendationItem: {
    flexDirection: 'row',
    marginBottom: 12,
    paddingRight: 10,
    alignItems: 'flex-start',
  },
  bulletPoint: {
    marginRight: 8,
    fontSize: 16,
    opacity: 0.8,
  },
  recommendationText: {
    flex: 1,
    fontSize: 14,
    lineHeight: 20,
  },
  selectContainer: {
    flexDirection: 'row',
    marginBottom: 10,
  },
  productButton: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 16,
    marginRight: 8,
    backgroundColor: 'rgba(150,150,150,0.1)',
    borderWidth: 1,
    borderColor: 'transparent',
  },
  selectedProduct: {
    backgroundColor: Colors.light.tint + '20',
    borderColor: Colors.light.tint,
  },
  productButtonText: {
    fontSize: 14,
  },
  selectedProductText: {
    color: Colors.light.tint,
    fontWeight: 'bold',
  },
});
