from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Pour permettre les requêtes depuis le frontend

@app.route('/')
def hello():
    return "Backend RAwajAI is running!"

@app.route('/api/test')
def test():
    return jsonify({"message": "API is working", "status": "success"})

@app.route('/api/status')
def api_status():
    return jsonify({
        "status": "connected",
        "message": "API connectée avec succès"
    })

@app.route('/api/dashboard')
def dashboard():
    return jsonify({
        "products": 15,
        "low_stock": 4,
        "anomalies": 2,
        "alerts": 3,
        "status": "connected"
    })

@app.route('/api/products')
def products():
    # Données d'exemple
    return jsonify({
        "products": [
            {"id": 1, "name": "Produit 1", "stock": 10},
            {"id": 2, "name": "Produit 2", "stock": 5},
            {"id": 3, "name": "Produit 3", "stock": 0}
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)