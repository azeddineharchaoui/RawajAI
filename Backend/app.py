import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={r"/api/*": {"origins": "*"}})

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "product.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    SAMPLE_PRODUCTS = json.load(f)
def generate_sample_data():
    dates = pd.date_range(start='2023-01-01', end='2024-01-15', freq='D')
    data = []
    for date in dates:
        for product_id in SAMPLE_PRODUCTS.keys():
            base_demand = SAMPLE_PRODUCTS[product_id]["demand_forecast"] / 30
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            noise = np.random.normal(0, 0.1)
            demand = max(0, base_demand * seasonal_factor * (1 + noise))
            if random.random() < 0.02:
                demand *= random.choice([0.1, 3.0])
            data.append({
                'date': date,
                'product_id': product_id,
                'demand': demand,
                'stock_level': max(0, SAMPLE_PRODUCTS[product_id]["stock"] - random.randint(0, 10)),
                'price': SAMPLE_PRODUCTS[product_id]["price"] * (1 + random.uniform(-0.05, 0.05))
            })
    return pd.DataFrame(data)

sample_data = generate_sample_data()

@app.route("/api/product-info", methods=["POST", "OPTIONS"])
def product_info():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    product_id = data.get("productId", "").strip().upper()
    print(f"🔍 Recherche du produit: {product_id}")
    if not product_id:
        return jsonify({"success": False, "message": "Product ID is required"}), 400
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            products = json.load(f)
        # product = next((p for p in products if str(p.get("id")).upper() == product_id), None)
        product = SAMPLE_PRODUCTS.get(product_id)

        if product:
            print(f" Produit trouvé dans JSON: {product.get('name', 'N/A')}")
            return jsonify({"success": True, "data": product})
    except FileNotFoundError:
        print("⚠ Fichier JSON non trouvé, utilisation des données simulées")
    if product_id in SAMPLE_PRODUCTS:
        product_info = SAMPLE_PRODUCTS[product_id].copy()
        print(f" Produit trouvé dans données simulées: {product_info['name']}")
        return jsonify({"success": True, "data": product_info})
    else:
        print(f" Produit non trouvé: {product_id}")
        available_products = list(SAMPLE_PRODUCTS.keys())
        return jsonify({
            "success": False, 
            "error": "Product not found",
            "available_products": available_products
        }), 404

@app.route("/api/detect-anomalies", methods=["POST", "OPTIONS"])
def detect_anomalies():
    if request.method == "OPTIONS":
        return '', 200
    print("Route /api/detect-anomalies appelée")  # Debug
    try:
        data = request.get_json()
        product_id = data.get("product_id", "").strip().upper()
        print(f"🔍 Détection d'anomalies pour: {product_id}")
        if not product_id:
            return jsonify({"success": False, "message": "Product ID is required"}), 400
        if product_id not in SAMPLE_PRODUCTS:
            return jsonify({"success": False, "message": f"Product {product_id} not found"}), 404
        product_data = sample_data[sample_data['product_id'] == product_id].copy()
        product_data = product_data.sort_values('date').tail(60)
        anomalies = []
        anomaly_dates = []
        anomaly_values = []
        for i in range(min(8, len(product_data))):
            if random.random() < 0.4:
                row = product_data.iloc[-(i+1)]
                expected_value = row['demand'] * 0.85
                anomalies.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'value': float(row['demand']),
                    'expected': float(expected_value),
                    'deviation': float(abs(row['demand'] - expected_value)),
                    'description': f'Unusual demand pattern detected on {row["date"].strftime("%Y-%m-%d")}'
                })
                anomaly_dates.append(row['date'].strftime('%Y-%m-%d'))
                anomaly_values.append(float(row['demand']))
        chart_data = [
            {
                'x': product_data['date'].dt.strftime('%Y-%m-%d').tolist(),
                'y': product_data['demand'].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Demand',
                'line': {'color': '#4299E1', 'width': 2},
                'marker': {'size': 6}
            }
        ]
        if anomalies:
            chart_data.append({
                'x': anomaly_dates,
                'y': anomaly_values,
                'type': 'scatter',
                'mode': 'markers',
                'name': 'Anomalies',
                'marker': {
                    'color': '#E53E3E', 
                    'size': 12,
                    'symbol': 'diamond'
                }
            })
        print(f"✅ Trouvé {len(anomalies)} anomalies pour {product_id}")
        return jsonify({
            'success': True,
            'anomalies': anomalies,
            'chart_data': chart_data,
            'message': f'Found {len(anomalies)} anomalies for product {product_id}'
        })
    except Exception as e:
        print(f" Erreur dans detect_anomalies: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error detecting anomalies: {str(e)}'
        }), 500

@app.route("/api/analyze-scenario", methods=["POST", "OPTIONS"])
def analyze_scenario():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        product_id = data.get('product_id', '').strip().upper()
        scenario = data.get('scenario', '').strip()
        print(f" Analyse de scénario pour: {product_id}")
        if not product_id or not scenario:
            return jsonify({
                'success': False,
                'message': 'Product ID and scenario are required'
            }), 400
        if product_id not in SAMPLE_PRODUCTS:
            return jsonify({
                'success': False,
                'message': f'Product {product_id} not found'
            }), 404
        product_info = SAMPLE_PRODUCTS[product_id]
        base_demand = product_info["demand_forecast"]
        scenario_multiplier = 1.2 if "increase" in scenario.lower() else 0.8
        dates = pd.date_range(start=datetime.now(), periods=30, freq='D')
        scenario_data = []
        for i, date in enumerate(dates):
            current_demand = base_demand * scenario_multiplier * (1 + random.uniform(-0.1, 0.1))
            scenario_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'projected_demand': current_demand
            })
        chart_data = [{
            'x': [d['date'] for d in scenario_data],
            'y': [d['projected_demand'] for d in scenario_data],
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Projected Demand',
            'line': {'color': '#38A169', 'width': 3},
            'marker': {'size': 8}
        }]
        analysis_message = f"""
Scenario Analysis for {product_info['name']}:

Scenario: {scenario}

Current monthly demand: {base_demand} units
Projected monthly demand: {base_demand * scenario_multiplier:.1f} units
Impact: {((scenario_multiplier - 1) * 100):+.1f}%

Recommendations:
- Current stock level: {product_info['stock']} units
- Reorder point: {product_info['reorder_point']} units
- Consider adjusting safety stock based on scenario impact
        """
        print(f"Analyse de scénario terminée pour {product_id}")
        return jsonify({
            'success': True,
            'message': analysis_message,
            'chart_data': chart_data,
            'scenario_data': scenario_data
        })
    except Exception as e:
        print(f" Erreur dans analyze_scenario: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error analyzing scenario: {str(e)}'
        }), 500

@app.route("/api/generate-report", methods=["POST", "OPTIONS"])
def generate_report():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json()
        product_id = data.get('product_id', '').strip().upper()
        report_type = data.get('report_type', 'forecast')
        print(f" Génération de rapport pour: {product_id}")
        if not product_id:
            return jsonify({
                'success': False,
                'message': 'Product ID is required'
            }), 400
        if product_id not in SAMPLE_PRODUCTS:
            return jsonify({
                'success': False,
                'message': f'Product {product_id} not found'
            }), 404
        product_info = SAMPLE_PRODUCTS[product_id]
        mock_report_url = f"data:text/html;charset=utf-8,<html><body><h1>{report_type.title()} Report</h1><h2>Product: {product_info['name']}</h2><p>Stock: {product_info['stock']} units</p><p>Price: {product_info['price']} MAD</p><p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></body></html>"
        print(f" Rapport généré pour {product_id}")
        return jsonify({
            'success': True,
            'report_url': mock_report_url,
            'message': f'{report_type.title()} report generated successfully'
        })
    except Exception as e:
        print(f" Erreur dans generate_report: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error generating report: {str(e)}'
        }), 500

@app.route("/anomaly_detection", methods=["POST", "OPTIONS"])
def anomaly_detection():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json()
    product_id = data.get("product_id")
    return detect_anomalies()

@app.route("/api/status", methods=["GET", "OPTIONS"])
def status():
    if request.method == 'OPTIONS':
        return '', 200
    return jsonify({
        'status': 'running',
        'message': 'Backend Flask opérationnel',
        'products_available': len(SAMPLE_PRODUCTS),
        'data_records': len(sample_data),
        'endpoints': [
            '/api/product-info',
            '/api/detect-anomalies', 
            '/api/analyze-scenario',
            '/api/generate-report'
        ],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/')
def home():
    return f'''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Supply Chain AI - Backend Flask</title>
</head>
<body>
    <h1>Avec succès</h1>
</body>
</html>

    '''

if __name__ == "__main__":
    print(" BACKEND FLASK DÉMARRÉ")
    print("=" * 50)
    print(" Données générées:", len(sample_data), "enregistrements")
    print(" Produits disponibles:", len(SAMPLE_PRODUCTS))
    print(" Endpoints: product-info, detect-anomalies, analyze-scenario, generate-report")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
