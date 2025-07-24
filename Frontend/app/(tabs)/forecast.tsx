import React, { useState, useEffect } from 'react';
import { StyleSheet, View, ScrollView, ActivityIndicator, Text, TextInput } from 'react-native';
import { Stack } from 'expo-router';
import { WebView } from 'react-native-webview';
import { Platform } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { api } from '@/services/apiClient';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { useForm, Controller } from "react-hook-form"
import * as yup from 'yup';
import { yupResolver } from '@hookform/resolvers/yup';

interface ForecastResult {
  chart_data?: any;
  metrics?: {
    accuracy?: number;
    rmse?: number;
    mape?: number;
  };
}

export default function ForecastScreen() {
  const colorScheme = useColorScheme();
  const colors = Colors[colorScheme ?? 'light'];

  const [productId, setProductId] = useState('');
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState<ForecastResult | null>(null);
  const [error, setError] = useState<any>('');
  const [plotHtml, setPlotHtml] = useState('');

  const schema = yup.object().shape({
    productId: yup.string().required('Product ID is required'),
    days: yup.number().min(1, 'Days must be at least 1').max(365, 'Days cannot exceed 365').required('Days is required').default(30),
  });

  type FormData = {
    productId: string;
    days: number;
  };


  const {
    control,
    handleSubmit,
    formState: { errors },
  } = useForm<FormData>({
    resolver: yupResolver(schema),
  })


  const generateForecast = async (data: FormData) => {
    const { productId, days } = data;
    setProductId(productId);

    if (!productId) {
      setError('Please enter a product ID');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await api.getForecast({
        product_id: productId,
        days: days,
      });
      console.log(response);
      

      setForecastData(response);
      // If we have chart data, create a simple plot
      if (response.chart_data) {
        const htmlContent = generatePlotHtml(response.chart_data);
        setPlotHtml(htmlContent);
      }
    } catch (error: any) {
      console.log('Forecast error:', error);
      setError(error.message || 'An error occurred while generating the forecast');
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
            const data = ${JSON.stringify(chartData)};
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
  return (

    <ThemedView style={styles.container}>

      <ScrollView style={styles.scrollView} contentContainerStyle={styles.contentContainer}>
        <View
          style={{
            marginVertical: 40,
            marginHorizontal: 20,
            paddingHorizontal: 20,
            paddingVertical: 10,
            backgroundColor: colors ? '#1f2937' : '#f0f4ff',
            borderRadius: 12,
            shadowColor: '#000',
            shadowOpacity: 0.1,
            shadowOffset: { width: 0, height: 2 },
            shadowRadius: 5,
            elevation: 3,
          }}
        >
          <Text
            style={{
              fontSize: 20,
              fontWeight: '600',
              color: colors ? '#ffffff' : '#1e3a8a',
            }}
          >
            📈 Forecast Results
          </Text>
          <Text
            style={{
              fontSize: 14,
              color: colors ? '#cbd5e1' : '#475569',
              marginTop: 4,
            }}
          >
            Predicted trends and performance metrics
          </Text>
        </View>
        <View
          style={{
            flexDirection: 'column',
            justifyContent: 'center',
            alignSelf: 'center',
            width: '90%',
            paddingVertical: 16,
          }}
        >
          <Text style={styles.labelStyle}>Product ID</Text>
          <Controller
            control={control}
            name="productId"
            render={({ field: { onChange, value } }) => (
              <TextInput
                placeholder="Enter Product ID"
                placeholderTextColor={Colors ? '#888' : '#999'}
                style={styles.inputStyle}
                value={value}
                onChangeText={onChange}
              />
            )}
          />
          {errors.productId && <Text style={styles.errorStyle}>{errors.productId.message}</Text>}

          <Text style={styles.labelStyle}>Forecast Period (Days)</Text>
          <Controller
            control={control}
            name="days"
            render={({ field: { onChange, value } }) => (
              <TextInput
                keyboardType="numeric"
                placeholder="30"
                placeholderTextColor={Colors ? '#888' : '#999'}
                style={styles.inputStyle}
                value={value !== undefined && value !== null ? String(value) : '30'}
                onChangeText={onChange}
              />
            )}
          />
          {errors.days && <Text style={styles.errorStyle}>{errors.days.message}</Text>}

          {loading ? (
            <ActivityIndicator size="large" color={colors.tint} style={{ marginTop: 32 }} />
          ) : (
            <Button
              onPress={handleSubmit(generateForecast)}
              style={{
                marginTop: 24,
                backgroundColor: '#4A90E2',
                paddingVertical: 12,
                borderRadius: 8,
              }}
              disabled={loading}
              text="Generate Forecast"
            />
          )}
        </View>
        {loading && (
          <ActivityIndicator size="large" color={colors.tint} style={styles.loader} />
        )}

        {forecastData && !loading && (
          <View style={{ marginTop: 20, padding: 10 }}>
            <Card>
              <ThemedText type="subtitle">Forecast Chart</ThemedText>

              <View style={{ height: 300, marginVertical: 10 }}>
                {Platform.OS !== 'web' ? (
                  <WebView
                    source={{ html: plotHtml }}
                    style={{ flex: 1 }}
                    scrollEnabled={false}
                    javaScriptEnabled={true}
                    domStorageEnabled={true}
                    originWhitelist={['*']}
                    onError={(e) => console.error('WebView error:', e.nativeEvent)}
                    renderError={(errorName) => (
                      <View style={{ alignItems: 'center', justifyContent: 'center' }}>
                        <ThemedText style={{ color: 'red' }}>Error loading chart: {errorName}</ThemedText>
                      </View>
                    )}
                  />
                ) : (
                  <ThemedText>
                    Web platform does not support WebView. Chart preview is unavailable on web.
                  </ThemedText>
                )}
              </View>
            </Card>

            <Card style={{ marginTop: 20 }}>
              <ThemedText type="subtitle">Forecast Metrics</ThemedText>

              {forecastData.metrics && (
                <View style={{ marginVertical: 10 }}>
                  <MetricItem
                    label="Accuracy"
                    value={`${Math.round((forecastData.metrics?.accuracy || 0) * 100) / 100}%`}
                  />
                  <MetricItem
                    label="RMSE"
                    value={Math.round((forecastData.metrics?.rmse || 0) * 100) / 100}
                  />
                  <MetricItem
                    label="MAPE"
                    value={`${Math.round((forecastData.metrics?.mape || 0) * 100) / 100}%`}
                  />
                </View>
              )}

              <Button onPress={() => console.log('Generate report for:', productId)}>
                Generate PDF Report
              </Button>
            </Card>
          </View>
        )}
      </ScrollView>
    </ThemedView>
  );
}

const MetricItem = ({ label, value }: { label: string; value: string | number }) => (
  <View style={styles.metricItem}>
    <ThemedText style={styles.metricLabel}>{label}</ThemedText>
    <ThemedText style={styles.metricValue}>{value}</ThemedText>
  </View>
);

const styles = StyleSheet.create({
  inputStyle: {
    width: '100%',
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: Colors ? '#444' : '#ccc',
    backgroundColor: Colors ? '#1c1c1e' : '#fff',
    color: Colors ? '#f2f2f2' : '#1c1c1e',
    marginTop: 6,
  },

  labelStyle: {
    marginTop: 16,
    marginBottom: 6,
    fontSize: 16,
    fontWeight: '500',
    color: Colors ? '#dde3e7ff' : '#333',
  },

  errorStyle: {
    color: '#E53E3E',
    fontSize: 13,
    marginTop: 4,
  },
  container: {
    flex: 1,
  },
  inputs: {
    width: '100%',
    paddingTop: 5,
    paddingBottom: 5,
    paddingLeft: 10,
    paddingRight: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: 'rgba(209, 197, 197, 0.43)',
    backgroundColor: 'transparent',
    color: '#dde3e7ff',
  },
  label: {
    marginBottom: 8,
    color: '#dde3e7ff',
    fontSize: 16,
    fontWeight: '500',
    marginTop: 16,
  },
  scrollView: {
    flex: 1,
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 40,
  },
  formCard: {
    padding: 16,
    marginTop: 16,
    marginBottom: 24,
  },
  loader: {
    marginVertical: 32,
  },
  resultsContainer: {
    gap: 16,
  },
  chartCard: {
    padding: 16,
  },
  webViewContainer: {
    height: 300,
    marginTop: 16,
    borderRadius: 8,
    overflow: 'hidden',
  },
  webView: {
    flex: 1,
    backgroundColor: 'transparent',
  },
  metricsCard: {
    padding: 16,
  },
  metrics: {
    marginVertical: 16,
  },
  metricItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(150, 150, 150, 0.2)',
  },
  metricLabel: {
    opacity: 0.7,
  },
  metricValue: {
    fontWeight: 'bold',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    color: '#E53E3E',
    marginVertical: 8,
  },
});
