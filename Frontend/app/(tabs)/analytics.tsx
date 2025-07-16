import React, { useState } from 'react';
import { StyleSheet, View, ScrollView, TouchableOpacity, TextInput } from 'react-native';
import { Stack } from 'expo-router';
import { WebView } from 'react-native-webview';

import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { api } from '@/services/apiClient';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';

type AnalyticsTab = 'anomalies' | 'scenarios' | 'reports';

interface OptimizationResult {
  chart_data?: any;
  anomalies?: any[];
  scenarios?: any;
  message?: string;
}

export default function AnalyticsScreen() {
  const colorScheme = useColorScheme();
  
  const [activeTab, setActiveTab] = useState<AnalyticsTab>('anomalies');
  const [productId, setProductId] = useState('');
  const [scenario, setScenario] = useState('');
  const [reportType, setReportType] = useState<'forecast' | 'inventory'>('forecast');
  const [language, setLanguage] = useState('en');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [error, setError] = useState('');
  const [plotHtml, setPlotHtml] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');

  // Handle anomaly detection
  const detectAnomalies = async () => {
    if (!productId) {
      setError('Please enter a product ID');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    setPlotHtml('');
    
    try {
      const response = await api.detectAnomalies({
        product_id: productId,
        language,
      });
      
      setResult(response);
      
      if (response.chart_data) {
        const htmlContent = generatePlotHtml(response.chart_data);
        setPlotHtml(htmlContent);
      }
    } catch (error) {
      console.error('Anomaly detection error:', error);
      setError('Failed to detect anomalies. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle scenario analysis
  const analyzeScenario = async () => {
    if (!productId || !scenario) {
      setError('Please enter both product ID and scenario description');
      return;
    }
    
    setLoading(true);
    setError('');
    setResult(null);
    setPlotHtml('');
    
    try {
      const response = await api.analyzeScenario({
        product_id: productId,
        scenario,
        language,
      });
      
      setResult(response);
      
      if (response.chart_data) {
        const htmlContent = generatePlotHtml(response.chart_data);
        setPlotHtml(htmlContent);
      }
    } catch (error) {
      console.error('Scenario analysis error:', error);
      setError('Failed to analyze scenario. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  // Handle report generation
  const generateReport = async () => {
    if (!productId) {
      setError('Please enter a product ID');
      return;
    }
    
    setLoading(true);
    setError('');
    setPdfUrl('');
    
    try {
      const response = await api.generateReport({
        report_type: reportType,
        product_id: productId,
        language,
      });
      
      if (response.success && response.report_url) {
        setPdfUrl(response.report_url);
      } else {
        // Show the error but provide a fallback
        if (response.error) {
          setError(`Report generation issue: ${response.error}`);
        } else {
          setError('No report URL returned. Server may be having issues generating the report.');
        }
        
        // If we're in development mode, show a mock PDF viewer
        if (__DEV__) {
          setResult({
            message: 'Using mock PDF view - this would show the actual report in production.'
          });
        }
      }
    } catch (error) {
      console.error('Report generation error:', error);
      setError('Failed to generate report. Please try again.');
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
        style={[styles.tab, activeTab === 'anomalies' && styles.activeTab]}
        onPress={() => setActiveTab('anomalies')}>
        <ThemedText 
          style={[
            styles.tabText, 
            activeTab === 'anomalies' && styles.activeTabText
          ]}>
          Anomalies
        </ThemedText>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.tab, activeTab === 'scenarios' && styles.activeTab]}
        onPress={() => setActiveTab('scenarios')}>
        <ThemedText 
          style={[
            styles.tabText, 
            activeTab === 'scenarios' && styles.activeTabText
          ]}>
          Scenarios
        </ThemedText>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.tab, activeTab === 'reports' && styles.activeTab]}
        onPress={() => setActiveTab('reports')}>
        <ThemedText 
          style={[
            styles.tabText, 
            activeTab === 'reports' && styles.activeTabText
          ]}>
          Reports
        </ThemedText>
      </TouchableOpacity>
    </View>
  );

  const renderAnomalyContent = () => (
    <>
      <Card style={styles.formCard}>
        <Input
          label="Product ID"
          placeholder="Enter product ID"
          value={productId}
          onChangeText={setProductId}
        />
        
        <View style={styles.rowInputs}>
          <View style={styles.fullInput}>
            <Input
              label="Language"
              placeholder="en"
              value={language}
              onChangeText={setLanguage}
            />
          </View>
        </View>
        
        {error ? <ThemedText style={styles.errorText}>{error}</ThemedText> : null}
        
        <Button
          text="Detect Anomalies"
          onPress={detectAnomalies}
          loading={loading}
          fullWidth
        />
      </Card>
      
      {/* Results */}
      {result?.anomalies && (
        <Card style={styles.resultsCard}>
          <ThemedText type="subtitle">Detected Anomalies</ThemedText>
          
          {result.anomalies.map((anomaly, index) => (
            <View key={index} style={styles.anomalyItem}>
              <View style={styles.anomalyHeader}>
                <ThemedText style={styles.anomalyDate}>{anomaly.date || `Point ${index + 1}`}</ThemedText>
                <View style={styles.anomalyBadge}>
                  <ThemedText style={styles.anomalyBadgeText}>Anomaly</ThemedText>
                </View>
              </View>
              <ThemedText style={styles.anomalyDescription}>
                {anomaly.description || `Unusual value detected: ${anomaly.value}`}
              </ThemedText>
              <View style={styles.anomalyStats}>
                <ThemedText style={styles.anomalyStatLabel}>Value:</ThemedText>
                <ThemedText style={styles.anomalyStatValue}>{anomaly.value?.toFixed(2) || '-'}</ThemedText>
                <ThemedText style={styles.anomalyStatLabel}>Expected:</ThemedText>
                <ThemedText style={styles.anomalyStatValue}>{anomaly.expected?.toFixed(2) || '-'}</ThemedText>
                <ThemedText style={styles.anomalyStatLabel}>Deviation:</ThemedText>
                <ThemedText style={styles.anomalyStatValue}>{anomaly.deviation?.toFixed(2) || '-'}</ThemedText>
              </View>
            </View>
          ))}
        </Card>
      )}
      
      {/* Chart visualization */}
      {plotHtml ? (
        <Card style={styles.chartCard}>
          <ThemedText type="subtitle">Anomaly Visualization</ThemedText>
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
  
  const renderScenarioContent = () => (
    <>
      <Card style={styles.formCard}>
        <Input
          label="Product ID"
          placeholder="Enter product ID"
          value={productId}
          onChangeText={setProductId}
        />
        
        <View style={styles.textareaContainer}>
          <ThemedText style={styles.textareaLabel}>Scenario Description</ThemedText>
          <TextInput
            style={styles.textarea}
            placeholder="Describe your scenario. E.g., 'What if demand increases by 20% in Q3?'"
            value={scenario}
            onChangeText={setScenario}
            multiline={true}
            numberOfLines={4}
            placeholderTextColor={colorScheme === 'dark' ? 'rgba(255,255,255,0.4)' : 'rgba(0,0,0,0.4)'}
          />
        </View>
        
        <View style={styles.rowInputs}>
          <View style={styles.fullInput}>
            <Input
              label="Language"
              placeholder="en"
              value={language}
              onChangeText={setLanguage}
            />
          </View>
        </View>
        
        {error ? <ThemedText style={styles.errorText}>{error}</ThemedText> : null}
        
        <Button
          text="Analyze Scenario"
          onPress={analyzeScenario}
          loading={loading}
          fullWidth
        />
      </Card>
      
      {/* Results */}
      {result?.message && (
        <Card style={styles.resultsCard}>
          <ThemedText type="subtitle">Scenario Analysis</ThemedText>
          <ThemedText style={styles.scenarioResult}>{result.message}</ThemedText>
        </Card>
      )}
      
      {/* Chart visualization */}
      {plotHtml ? (
        <Card style={styles.chartCard}>
          <ThemedText type="subtitle">Scenario Visualization</ThemedText>
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
  
  const renderReportContent = () => (
    <>
      <Card style={styles.formCard}>
        <Input
          label="Product ID"
          placeholder="Enter product ID"
          value={productId}
          onChangeText={setProductId}
        />
        
        <View style={styles.rowInputs}>
          <View style={styles.fullInput}>
            <ThemedText style={styles.textareaLabel}>Report Type</ThemedText>
            <View style={styles.reportTypeContainer}>
              <TouchableOpacity
                style={[styles.reportTypeButton, reportType === 'forecast' && styles.reportTypeSelected]}
                onPress={() => setReportType('forecast')}
              >
                <ThemedText style={reportType === 'forecast' ? styles.reportTypeTextSelected : styles.reportTypeText}>
                  Forecast Report
                </ThemedText>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.reportTypeButton, reportType === 'inventory' && styles.reportTypeSelected]}
                onPress={() => setReportType('inventory')}
              >
                <ThemedText style={reportType === 'inventory' ? styles.reportTypeTextSelected : styles.reportTypeText}>
                  Inventory Report
                </ThemedText>
              </TouchableOpacity>
            </View>
          </View>
        </View>
        
        <View style={styles.rowInputs}>
          <View style={styles.fullInput}>
            <Input
              label="Language"
              placeholder="en"
              value={language}
              onChangeText={setLanguage}
            />
          </View>
        </View>
        
        {error ? <ThemedText style={styles.errorText}>{error}</ThemedText> : null}
        
        <Button
          text="Generate Report"
          onPress={generateReport}
          loading={loading}
          fullWidth
        />
      </Card>
      
      {/* PDF Viewer */}
      {pdfUrl && (
        <Card style={styles.pdfCard}>
          <ThemedText type="subtitle">Generated Report</ThemedText>
          <View style={styles.pdfContainer}>
            <WebView
              source={{ 
                uri: pdfUrl,
                headers: {
                  'Accept': 'application/pdf',
                  'Content-Type': 'application/pdf'
                }
              }}
              style={styles.webView}
              originWhitelist={['*']}
              javaScriptEnabled={true}
              domStorageEnabled={true}
              startInLoadingState={true}
              scalesPageToFit={true}
              key="webViewKey"
              onError={(e) => console.error('WebView error:', e.nativeEvent)}
              onHttpError={(e) => console.error('WebView HTTP error:', e.nativeEvent)}
              renderError={(errorName) => (
                <View style={styles.errorContainer}>
                  <ThemedText style={styles.errorText}>Error loading PDF: {errorName}</ThemedText>
                  <Button text="Retry" onPress={() => setReportType(reportType)} type="primary" />
                </View>
              )}
            />
          </View>
          <Button 
            text="Download PDF"
            onPress={() => { /* Open PDF in external viewer */ }}
            fullWidth
          />
        </Card>
      )}
    </>
  );

  return (
    <ThemedView style={styles.container}>
      <Stack.Screen options={{ title: "Supply Chain Analytics", headerShown: true }} />
      
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.contentContainer}>
        <ThemedText type="subtitle">Advanced Supply Chain Analytics</ThemedText>
        
        {renderTabs()}
        
        {activeTab === 'anomalies' && renderAnomalyContent()}
        {activeTab === 'scenarios' && renderScenarioContent()}
        {activeTab === 'reports' && renderReportContent()}
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
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
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
  rowInputs: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 8,
    marginBottom: 8,
  },
  halfInput: {
    width: '48%',
  },
  fullInput: {
    width: '100%',
  },
  buttonText: {
    color: '#fff',
    fontWeight: 'bold',
    textAlign: 'center',
    fontSize: 16,
  },
  errorText: {
    color: '#E53E3E',
    marginVertical: 8,
  },
  resultsCard: {
    marginBottom: 16,
    padding: 16,
  },
  chartCard: {
    marginBottom: 16,
    padding: 16,
  },
  chartContainer: {
    height: 300,
    marginTop: 8,
  },
  webView: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  textareaContainer: {
    marginVertical: 8,
  },
  textareaLabel: {
    fontSize: 14,
    marginBottom: 6,
    opacity: 0.8,
  },
  textarea: {
    borderWidth: 1,
    borderColor: 'rgba(150,150,150,0.3)',
    borderRadius: 8,
    padding: 12,
    minHeight: 100,
    textAlignVertical: 'top',
    fontSize: 16,
  },
  anomalyItem: {
    marginTop: 12,
    padding: 12,
    borderRadius: 8,
    backgroundColor: 'rgba(229, 62, 62, 0.1)',
  },
  anomalyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  anomalyDate: {
    fontWeight: 'bold',
  },
  anomalyBadge: {
    backgroundColor: '#E53E3E',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  anomalyBadgeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: 'bold',
  },
  anomalyDescription: {
    marginBottom: 8,
  },
  anomalyStats: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  anomalyStatLabel: {
    width: '25%',
    fontSize: 12,
    opacity: 0.7,
  },
  anomalyStatValue: {
    width: '25%',
    fontSize: 12,
    fontWeight: 'bold',
  },
  scenarioResult: {
    lineHeight: 22,
  },
  pdfCard: {
    marginBottom: 16,
    padding: 16,
  },
  pdfContainer: {
    height: 500,
    marginVertical: 16,
  },
  reportTypeContainer: {
    flexDirection: 'row',
    marginBottom: 16,
  },
  reportTypeButton: {
    flex: 1,
    padding: 12,
    borderWidth: 1,
    borderColor: 'rgba(150,150,150,0.3)',
    alignItems: 'center',
  },
  reportTypeSelected: {
    backgroundColor: Colors.light.tint,
    borderColor: Colors.light.tint,
  },
  reportTypeText: {
    fontSize: 14,
  },
  reportTypeTextSelected: {
    fontSize: 14,
    color: 'white',
    fontWeight: 'bold',
  },
});
