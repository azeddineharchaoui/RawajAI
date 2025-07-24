"use client"
import { useState } from "react"
import { StyleSheet, View, ScrollView, TouchableOpacity, TextInput } from "react-native"
import { Stack } from "expo-router"
import { WebView } from "react-native-webview"
import { ThemedText } from "@/components/ThemedText"
import { ThemedView } from "@/components/ThemedView"
import { Card } from "@/components/ui/Card"
import { Button } from "@/components/ui/Button"
import { Input } from "@/components/ui/Input"
import { Colors } from "@/constants/Colors"
import { useColorScheme } from "@/hooks/useColorScheme"
import { _DEV_ } from "react-native"
import { Platform } from 'react-native';
type AnalyticsTab = "anomalies" | "scenarios" | "reports"

interface ProductInfo {
  id: string
  name: string
  stock: number
  price: number
  category?: string
  supplier?: string
  last_updated?: string
  demand_forecast?: number
  reorder_point?: number
  safety_stock?: number
}

interface OptimizationResult {
  chart_data?: any
  anomalies?: any[]
  scenarios?: any
  message?: string
  success?: boolean
}

export default function AnalyticsScreen() {
  const colorScheme = useColorScheme()

  // États principaux
  const [activeTab, setActiveTab] = useState<AnalyticsTab>("anomalies")
  const [productId, setProductId] = useState("")
  const [scenario, setScenario] = useState("")
  const [reportType, setReportType] = useState<"forecast" | "inventory">("forecast")
  const [language, setLanguage] = useState("en")
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<OptimizationResult | null>(null)
  const [error, setError] = useState("")
  const [plotHtml, setPlotHtml] = useState("")
  const [pdfUrl, setPdfUrl] = useState("")

  // État pour les informations produit
  const [productInfo, setProductInfo] = useState<ProductInfo | null>(null)
  const [productLoading, setProductLoading] = useState(false)

  // Configuration API
  const API_BASE_URL = "https://meals-recognised-encouraged-organizing.trycloudflare.com"
  
 const fetchProductInfoSilently = async (productIdToFetch: string): Promise<ProductInfo | null> => {
  try {
    console.log("🔄 Récupération silencieuse des infos produit:", productIdToFetch);

    const response = await fetch(`${API_BASE_URL}/api/product-info`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ productId: productIdToFetch.trim() }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();

    if (result.success && result.data) {
      console.log("✅ Infos produit récupérées silencieusement:", result.data.name);
      return result.data;
    } else {
      console.warn("⚠ Produit non trouvé lors de la récupération silencieuse");
      return null;
    }
  } catch (error) {
    console.error("❌ Erreur lors de la récupération silencieuse:", error);
    return null;
  }
};

  const getProductInfo = async () => {
    if (!productId.trim()) {
      setError("Please enter a product ID")
      return
    }

    setProductLoading(true)
    setError("")
    setProductInfo(null)

    try {
const response = await fetch(`${API_BASE_URL}/api/product-info`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({ productId: productId.trim() }),
      })

      if (!response.ok) {
  throw new Error(`HTTP error! status: ${response.status}`)      }

      const result = await response.json()

      if (result.success && result.data) {
        console.log("Produit trouvé:", result.data)
        setProductInfo(result.data)
      } else {
        console.warn("⚠ Produit non trouvé")
        setError(result.error || "Product not found")
        setProductInfo(null)
      }
    } catch (error) {
      console.error(" Erreur lors de la requête produit:", error)
      setError("Failed to fetch product information")
      setProductInfo(null)
    } finally {
      setProductLoading(false)
    }
  }

  // Handle anomaly detection avec récupération automatique des infos produit
  const detectAnomalies = async () => {
    if (!productId.trim()) {
      setError("Please enter a product ID")
      return
    }

    setLoading(true)
    setError("")
    setResult(null)
    setPlotHtml("")

    try {
      console.log("🔍 Détection d'anomalies pour:", productId.trim())

      // 1. Récupérer les infos produit en arrière-plan si pas déjà disponibles
      if (!productInfo || productInfo.id !== productId.trim().toUpperCase()) {
        console.log("📦 Récupération automatique des infos produit...")
        const fetchedProductInfo = await fetchProductInfoSilently(productId.trim())
        if (fetchedProductInfo) {
          setProductInfo(fetchedProductInfo)
        }
      }

      // 2. Procéder à la détection d'anomalies
     const response = await fetch(`${API_BASE_URL}/api/detect-anomalies`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  body: JSON.stringify({
    product_id: productId.trim(),
    language: language,
  }),
});

      if (!response.ok) {
throw new Error(`HTTP error! status: ${response.status}`)
      }

      const responseData = await response.json()
      console.log("📊 Réponse anomalies:", responseData)

      if (responseData.success) {
        setResult(responseData)

        if (responseData.chart_data) {
          const htmlContent = generatePlotHtml(responseData.chart_data, "Anomaly Detection Results")
          setPlotHtml(htmlContent)
        }
      } else {
        setError(responseData.message || "Failed to detect anomalies")
      }
    } catch (error) {
      console.error("❌ Anomaly detection error:", error)
setError(`Failed to detect anomalies: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  // Handle scenario analysis avec récupération automatique des infos produit
  const analyzeScenario = async () => {
    if (!productId.trim() || !scenario.trim()) {
      setError("Please enter both product ID and scenario description")
      return
    }

    setLoading(true)
    setError("")
    setResult(null)
    setPlotHtml("")

    try {
      console.log("🎯 Analyse de scénario pour:", productId.trim())

      // 1. Récupérer les infos produit en arrière-plan si pas déjà disponibles
      if (!productInfo || productInfo.id !== productId.trim().toUpperCase()) {
        console.log("📦 Récupération automatique des infos produit...")
        const fetchedProductInfo = await fetchProductInfoSilently(productId.trim())
        if (fetchedProductInfo) {
          setProductInfo(fetchedProductInfo)
        }
      }

      // 2. Procéder à l'analyse de scénario
const response = await fetch(`${API_BASE_URL}/api/analyze-scenario`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          product_id: productId.trim(),
          scenario: scenario.trim(),
          language: language,
        }),
      })

      if (!response.ok) {
  throw new Error(`HTTP error! status: ${response.status}`)      }

      const responseData = await response.json()
      console.log("📈 Réponse scénario:", responseData)

      if (responseData.success) {
        setResult(responseData)

        if (responseData.chart_data) {
          const htmlContent = generatePlotHtml(responseData.chart_data, "Scenario Analysis Results")
          setPlotHtml(htmlContent)
        }
      } else {
        setError(responseData.message || "Failed to analyze scenario")
      }
    } catch (error) {
      console.error(" Scenario analysis error:", error)
setError(`Failed to analyze scenario: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const generateReport = async () => {
    if (!productId.trim()) {
      setError("Please enter a product ID")
      return
    }

    setLoading(true)
    setError("")
    setPdfUrl("")

    try {
      console.log(" Génération de rapport pour:", productId.trim())

      // 1. Récupérer les infos produit en arrière-plan si pas déjà disponibles
      if (!productInfo || productInfo.id !== productId.trim().toUpperCase()) {
        console.log("📦 Récupération automatique des infos produit...")
        const fetchedProductInfo = await fetchProductInfoSilently(productId.trim())
        if (fetchedProductInfo) {
          setProductInfo(fetchedProductInfo)
        }
      }

      const response = await fetch(`${API_BASE_URL}/api/generate-report`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Accept: "application/json",
  },
  body: JSON.stringify({
    report_type: reportType,
    product_id: productId.trim(),
    language: language,
  }),
});

      if (!response.ok) {
  throw new Error(`HTTP error! status: ${response.status}`)      }

      const responseData = await response.json()
      console.log("Réponse rapport:", responseData)

      if (responseData.success && responseData.report_url) {
        setPdfUrl(responseData.report_url)
      } else {
        if (responseData.error) {
setError(`Report generation issue: ${responseData.error}`)
        } else {
          setError("No report URL returned. Server may be having issues generating the report.")
        }

        if (_DEV_) {
          setResult({
            message: "Using mock PDF view - this would show the actual report in production.",
          })
        }
      }
    } catch (error) {
      console.error(" Report generation error:", error)
setError(`Failed to generate report: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

 const generatePlotHtml = (chartData: any, title = "Chart") => {
  return `
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
        <style>
          body {
            margin: 0;
            padding: 10px;
            background: ${colorScheme === "dark" ? "#151718" : "#fff"};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          }
          #chart { width: 100%; height: calc(100vh - 60px); }
          .chart-title {
            text-align: center;
            color: ${colorScheme === "dark" ? "#ECEDEE" : "#11181C"};
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
          }
        </style>
      </head>
      <body>
        <div class="chart-title">${title}</div>
        <div id="chart"></div>
        <script>
          try {
            const data = ${JSON.stringify(chartData || [])};
            console.log(' Données du graphique:', data);
            
            if (!data || data.length === 0) {
              document.getElementById('chart').innerHTML = 
                '<div style="text-align: center; padding: 50px; color: #666;">No chart data available</div>';
            } else {
              const layout = {
                margin: { t: 20, r: 20, l: 60, b: 60 },
                paper_bgcolor: '${colorScheme === "dark" ? "#151718" : "#fff"}',
                plot_bgcolor: '${colorScheme === "dark" ? "#151718" : "#fff"}',
                font: {
                  color: '${colorScheme === "dark" ? "#ECEDEE" : "#11181C"}',
                  family: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
                },
                xaxis: {
                  showgrid: true,
                  gridcolor: 'rgba(150,150,150,0.2)',
                  title: { font: { size: 14 } }
                },
                yaxis: {
                  showgrid: true,
                  gridcolor: 'rgba(150,150,150,0.2)',
                  title: { font: { size: 14 } }
                },
                showlegend: true,
                legend: {
                  bgcolor: 'rgba(0,0,0,0)',
                  bordercolor: 'rgba(0,0,0,0)'
                },
                hovermode: 'closest'
              };

              Plotly.newPlot('chart', data, layout, {
                responsive: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
              });

              console.log('✅ Graphique créé avec succès');
            }
          } catch (error) {
            console.error(' Erreur création graphique:', error);
            document.getElementById('chart').innerHTML =
              '<div style="text-align: center; padding: 50px; color: #E53E3E;">Error loading chart: ' + error.message + '</div>';
          }
        </script>
      </body>
    </html>
  `;
};

  // Composant pour afficher les informations produit
  const renderProductInfo = () => {
    if (!productInfo) return null

    return (
      <Card style={styles.productInfoCard}>
        <ThemedText type="subtitle" style={styles.productInfoTitle}>
          📦 Product Information
        </ThemedText>
        <View style={styles.productInfoGrid}>
          <View style={styles.productInfoItem}>
            <ThemedText style={styles.productInfoLabel}>Name:</ThemedText>
            <ThemedText style={styles.productInfoValue}>{productInfo.name}</ThemedText>
          </View>
          <View style={styles.productInfoItem}>
            <ThemedText style={styles.productInfoLabel}>Stock:</ThemedText>
            <ThemedText style={[styles.productInfoValue, { color: productInfo.stock < 10 ? "#E53E3E" : "#38A169" }]}>
              {productInfo.stock} units
            </ThemedText>
          </View>
          <View style={styles.productInfoItem}>
            <ThemedText style={styles.productInfoLabel}>Price:</ThemedText>
            <ThemedText style={styles.productInfoValue}>{productInfo.price} MAD</ThemedText>
          </View>
          {productInfo.category && (
            <View style={styles.productInfoItem}>
              <ThemedText style={styles.productInfoLabel}>Category:</ThemedText>
              <ThemedText style={styles.productInfoValue}>{productInfo.category}</ThemedText>
            </View>
          )}
          {productInfo.supplier && (
            <View style={styles.productInfoItem}>
              <ThemedText style={styles.productInfoLabel}>Supplier:</ThemedText>
              <ThemedText style={styles.productInfoValue}>{productInfo.supplier}</ThemedText>
            </View>
          )}
          {productInfo.reorder_point && (
            <View style={styles.productInfoItem}>
              <ThemedText style={styles.productInfoLabel}>Reorder Point:</ThemedText>
              <ThemedText style={styles.productInfoValue}>{productInfo.reorder_point} units</ThemedText>
            </View>
          )}
          {productInfo.safety_stock && (
            <View style={styles.productInfoItem}>
              <ThemedText style={styles.productInfoLabel}>Safety Stock:</ThemedText>
              <ThemedText style={styles.productInfoValue}>{productInfo.safety_stock} units</ThemedText>
            </View>
          )}
          {productInfo.demand_forecast && (
            <View style={styles.productInfoItem}>
              <ThemedText style={styles.productInfoLabel}>Demand Forecast:</ThemedText>
              <ThemedText style={styles.productInfoValue}>{productInfo.demand_forecast} units/month</ThemedText>
            </View>
          )}
        </View>
        {productInfo.last_updated && (
          <ThemedText style={styles.lastUpdated}>
            Last updated: {new Date(productInfo.last_updated).toLocaleDateString()}
          </ThemedText>
        )}
      </Card>
    )
  }

  const renderTabs = () => (
    <View style={styles.tabContainer}>
      <TouchableOpacity
        style={[styles.tab, activeTab === "anomalies" && styles.activeTab]}
        onPress={() => setActiveTab("anomalies")}
      >
        <ThemedText style={[styles.tabText, activeTab === "anomalies" && styles.activeTabText]}>
          🔍 Anomalies
        </ThemedText>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.tab, activeTab === "scenarios" && styles.activeTab]}
        onPress={() => setActiveTab("scenarios")}
      >
        <ThemedText style={[styles.tabText, activeTab === "scenarios" && styles.activeTabText]}>
          🎯 Scenarios
        </ThemedText>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.tab, activeTab === "reports" && styles.activeTab]}
        onPress={() => setActiveTab("reports")}
      >
        <ThemedText style={[styles.tabText, activeTab === "reports" && styles.activeTabText]}>📊 Reports</ThemedText>
      </TouchableOpacity>
    </View>
  )

  const renderAnomalyContent = () => (
    <>
      <Card style={styles.formCard}>
        <Input
          label="Product ID"
          placeholder="Enter product ID (e.g., PROD001)"
          value={productId}
          onChangeText={setProductId}
        />
        <View style={styles.rowInputs}>
          <View style={styles.fullInput}>
            <Input label="Language" placeholder="en, fr, es..." value={language} onChangeText={setLanguage} />
          </View>
        </View>
        {error ? <ThemedText style={styles.errorText}>{error}</ThemedText> : null}
        <View style={styles.buttonRow}>
          <Button text="Get Product Info" onPress={getProductInfo} loading={productLoading} style={styles.halfButton} />
          <Button text="Detect Anomalies" onPress={detectAnomalies} loading={loading} style={styles.halfButton} />
        </View>
      </Card>

      {/* Chart visualization - AFFICHÉ EN PREMIER */}
{plotHtml ? (
  <Card style={styles.chartCard}>
    <ThemedText type="subtitle">📈 Anomaly Visualization</ThemedText>
    <View style={styles.chartContainer}>
      {Platform.OS === 'web' ? (
        <iframe
          srcDoc={plotHtml}
          style={{ width: '100%', height: 400, border: 'none' }}
          sandbox="allow-scripts" 
        />
      ) : (
        <WebView
          source={{ html: plotHtml }}
          style={styles.webView}
          scrollEnabled={false}
          javaScriptEnabled={true}
          domStorageEnabled={true}
          onError={(e) =>
            console.error("WebView error:", e.nativeEvent)
          }
        />
      )}
    </View>
  </Card>
) : null}


      {/* Informations produit */}
      {renderProductInfo()}

      {/* Results */}
      {result?.anomalies && (
        <Card style={styles.resultsCard}>
          <ThemedText type="subtitle">🚨 Detected Anomalies</ThemedText>
          {result.anomalies.map((anomaly, index) => (
            <View key={index} style={styles.anomalyItem}>
              <View style={styles.anomalyHeader}>
<ThemedText style={styles.anomalyDate}>
  {anomaly.date || `Point ${index + 1}`}
</ThemedText>
                <View style={styles.anomalyBadge}>
                  <ThemedText style={styles.anomalyBadgeText}>Anomaly</ThemedText>
                </View>
              </View>
              <ThemedText style={styles.anomalyDescription}>
{anomaly.description || `Unusual value detected: ${anomaly.value}`}
              </ThemedText>
              <View style={styles.anomalyStats}>
                <ThemedText style={styles.anomalyStatLabel}>Value:</ThemedText>
                <ThemedText style={styles.anomalyStatValue}>{anomaly.value?.toFixed(2) || "-"}</ThemedText>
                <ThemedText style={styles.anomalyStatLabel}>Expected:</ThemedText>
                <ThemedText style={styles.anomalyStatValue}>{anomaly.expected?.toFixed(2) || "-"}</ThemedText>
                <ThemedText style={styles.anomalyStatLabel}>Deviation:</ThemedText>
                <ThemedText style={styles.anomalyStatValue}>{anomaly.deviation?.toFixed(2) || "-"}</ThemedText>
              </View>
            </View>
          ))}
        </Card>
      )}
    </>
  )

  const renderScenarioContent = () => (
    <>
      <Card style={styles.formCard}>
        <Input label="Product ID" placeholder="Enter product ID" value={productId} onChangeText={setProductId} />
        <View style={styles.textareaContainer}>
          <ThemedText style={styles.textareaLabel}>Scenario Description</ThemedText>
          <TextInput
            style={[
              styles.textarea,
              {
                color: colorScheme === "dark" ? "#ECEDEE" : "#11181C",
                borderColor: colorScheme === "dark" ? "rgba(255,255,255,0.3)" : "rgba(0,0,0,0.3)",
              },
            ]}
            placeholder="Describe your scenario. E.g., 'What if demand increases by 20% in Q3?'"
            value={scenario}
            onChangeText={setScenario}
            multiline={true}
            numberOfLines={4}
            placeholderTextColor={colorScheme === "dark" ? "rgba(255,255,255,0.4)" : "rgba(0,0,0,0.4)"}
          />
        </View>
        <View style={styles.rowInputs}>
          <View style={styles.fullInput}>
            <Input label="Language" placeholder="en, fr, es..." value={language} onChangeText={setLanguage} />
          </View>
        </View>
        {error ? <ThemedText style={styles.errorText}>{error}</ThemedText> : null}
        <View style={styles.buttonRow}>
          <Button text="Get Product Info" onPress={getProductInfo} loading={productLoading} style={styles.halfButton} />
          <Button text="Analyze Scenario" onPress={analyzeScenario} loading={loading} style={styles.halfButton} />
        </View>
      </Card>

      {/* Chart visualization - AFFICHÉ EN PREMIER */}
      {plotHtml ? (
        <Card style={styles.chartCard}>
          <ThemedText type="subtitle">📈 Scenario Visualization</ThemedText>
          <View style={styles.chartContainer}>
            <WebView
              source={{ html: plotHtml }}
              style={styles.webView}
              scrollEnabled={false}
              javaScriptEnabled={true}
              domStorageEnabled={true}
              onError={(e) => console.error("WebView error:", e.nativeEvent)}
              onMessage={(event) => console.log("WebView message:", event.nativeEvent.data)}
              renderError={(errorName) => (
                <View style={styles.errorContainer}>
                  <ThemedText style={styles.errorText}>Error loading chart: {errorName}</ThemedText>
                </View>
              )}
            />
          </View>
        </Card>
      ) : null}

      {/* Informations produit */}
      {renderProductInfo()}

      {/* Results */}
      {result?.message && (
        <Card style={styles.resultsCard}>
          <ThemedText type="subtitle">🎯 Scenario Analysis</ThemedText>
          <ThemedText style={styles.scenarioResult}>{result.message}</ThemedText>
        </Card>
      )}
    </>
  )

  const renderReportContent = () => (
    <>
      <Card style={styles.formCard}>
        <Input label="Product ID" placeholder="Enter product ID" value={productId} onChangeText={setProductId} />
        <View style={styles.rowInputs}>
          <View style={styles.fullInput}>
            <ThemedText style={styles.textareaLabel}>Report Type</ThemedText>
            <View style={styles.reportTypeContainer}>
              <TouchableOpacity
                style={[styles.reportTypeButton, reportType === "forecast" && styles.reportTypeSelected]}
                onPress={() => setReportType("forecast")}
              >
                <ThemedText style={reportType === "forecast" ? styles.reportTypeTextSelected : styles.reportTypeText}>
                   Forecast Report
                </ThemedText>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.reportTypeButton, reportType === "inventory" && styles.reportTypeSelected]}
                onPress={() => setReportType("inventory")}
              >
                <ThemedText style={reportType === "inventory" ? styles.reportTypeTextSelected : styles.reportTypeText}>
                   Inventory Report
                </ThemedText>
              </TouchableOpacity>
            </View>
          </View>
        </View>
        <View style={styles.rowInputs}>
          <View style={styles.fullInput}>
            <Input label="Language" placeholder="en, fr, es..." value={language} onChangeText={setLanguage} />
          </View>
        </View>
        {error ? <ThemedText style={styles.errorText}>{error}</ThemedText> : null}
        <View style={styles.buttonRow}>
          <Button text="Get Product Info" onPress={getProductInfo} loading={productLoading} style={styles.halfButton} />
          <Button text="Generate Report" onPress={generateReport} loading={loading} style={styles.halfButton} />
        </View>
      </Card>

      

      {/* Informations produit */}
      {renderProductInfo()}

      {/* PDF Viewer */}
      {pdfUrl && (
        <Card style={styles.pdfCard}>
          <ThemedText type="subtitle"> Generated Report</ThemedText>
          <View style={styles.pdfContainer}>
            <WebView
              source={{
                uri: pdfUrl,
                headers: {
                  Accept: "application/pdf",
                  "Content-Type": "application/pdf",
                },
              }}
              style={styles.webView}
              originWhitelist={["*"]}
              javaScriptEnabled={true}
              domStorageEnabled={true}
              startInLoadingState={true}
              scalesPageToFit={true}
              onError={(e) => console.error("WebView error:", e.nativeEvent)}
              onHttpError={(e) => console.error("WebView HTTP error:", e.nativeEvent)}
              renderError={(errorName) => (
                <View style={styles.errorContainer}>
                  <ThemedText style={styles.errorText}>Error loading PDF: {errorName}</ThemedText>
                  <Button text="Retry" onPress={() => generateReport()} />
                </View>
              )}
            />
          </View>
          <Button
            text=" Download PDF"
            onPress={() => {
              /* Open PDF in external viewer */
            }}
            fullWidth
          />
        </Card>
      )}
    </>
  )

  return (
    <ThemedView style={styles.container}>
      <Stack.Screen options={{ title: " Supply Chain Analytics", headerShown: true }} />
      <ScrollView style={styles.scrollView} contentContainerStyle={styles.contentContainer}>
        <ThemedText type="subtitle" style={styles.mainTitle}>
           Advanced Supply Chain Analytics
        </ThemedText>
        {renderTabs()}
        {activeTab === "anomalies" && renderAnomalyContent()}
        {activeTab === "scenarios" && renderScenarioContent()}
        {activeTab === "reports" && renderReportContent()}
      </ScrollView>
    </ThemedView>
  )
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
    justifyContent: "center",
    alignItems: "center",
    padding: 20,
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 32,
  },
  mainTitle: {
    textAlign: "center",
    marginBottom: 16,
  },
  tabContainer: {
    flexDirection: "row",
    marginBottom: 16,
    backgroundColor: "rgba(150,150,150,0.1)",
    borderRadius: 12,
    padding: 4,
  },
  tab: {
    flex: 1,
    paddingVertical: 12,
    alignItems: "center",
    borderRadius: 8,
  },
  activeTab: {
    backgroundColor: Colors.light.tint,
  },
  tabText: {
    fontSize: 14,
    fontWeight: "500",
  },
  activeTabText: {
    fontWeight: "bold",
    color: "white",
  },
  formCard: {
    marginBottom: 16,
    padding: 16,
  },
  productInfoCard: {
    marginBottom: 16,
    padding: 16,
    backgroundColor: "rgba(56, 161, 105, 0.1)",
    borderLeftWidth: 4,
    borderLeftColor: "#38A169",
  },
  productInfoTitle: {
    marginBottom: 12,
    color: "#38A169",
  },
  productInfoGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
  },
  productInfoItem: {
    width: "48%",
    marginBottom: 12,
  },
  productInfoLabel: {
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 2,
  },
  productInfoValue: {
    fontSize: 14,
    fontWeight: "bold",
  },
  lastUpdated: {
    fontSize: 12,
    opacity: 0.6,
    textAlign: "center",
    marginTop: 8,
    fontStyle: "italic",
  },
  buttonRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 16,
  },
  halfButton: {
    width: "48%",
  },
  rowInputs: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 8,
    marginBottom: 8,
  },
  halfInput: {
    width: "48%",
  },
  fullInput: {
    width: "100%",
  },
  errorText: {
    color: "#E53E3E",
    marginVertical: 8,
    textAlign: "center",
  },
  resultsCard: {
    marginBottom: 16,
    padding: 16,
  },
  chartCard: {
    marginBottom: 16,
    padding: 16,
    backgroundColor: "rgba(66, 153, 225, 0.1)",
    borderLeftWidth: 4,
    borderLeftColor: "#4299E1",
  },
  chartContainer: {
    height: 350,
    marginTop: 12,
    borderRadius: 8,
    overflow: "hidden",
  },
  webView: {
    flex: 1,
    backgroundColor: "transparent",
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
    borderRadius: 8,
    padding: 12,
    minHeight: 100,
    textAlignVertical: "top",
    fontSize: 16,
  },
  anomalyItem: {
    marginTop: 12,
    padding: 12,
    borderRadius: 8,
    backgroundColor: "rgba(229, 62, 62, 0.1)",
    borderLeftWidth: 3,
    borderLeftColor: "#E53E3E",
  },
  anomalyHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  anomalyDate: {
    fontWeight: "bold",
  },
  anomalyBadge: {
    backgroundColor: "#E53E3E",
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  anomalyBadgeText: {
    color: "white",
    fontSize: 12,
    fontWeight: "bold",
  },
  anomalyDescription: {
    marginBottom: 8,
    lineHeight: 20,
  },
  anomalyStats: {
    flexDirection: "row",
    flexWrap: "wrap",
  },
  anomalyStatLabel: {
    width: "25%",
    fontSize: 12,
    opacity: 0.7,
  },
  anomalyStatValue: {
    width: "25%",
    fontSize: 12,
    fontWeight: "bold",
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
    borderRadius: 8,
    overflow: "hidden",
  },
  reportTypeContainer: {
    flexDirection: "row",
    marginBottom: 16,
    borderRadius: 8,
    overflow: "hidden",
  },
  reportTypeButton: {
    flex: 1,
    padding: 12,
    borderWidth: 1,
    borderColor: "rgba(150,150,150,0.3)",
    alignItems: "center",
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
    color: "white",
    fontWeight: "bold",
  },
})
