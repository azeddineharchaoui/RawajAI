#!/usr/bin/env python3
"""
 RawaJAI FastAPI Server - Version finale
Serveur API complet avec CORS configuré
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os
from datetime import datetime

print(" INITIALISATION RAWAJAI FASTAPI SERVER")
print("=" * 50)

# Créer l'application FastAPI
app = FastAPI(
    title=" RawaJAI Supply Chain API",
    description="Assistant IA pour la gestion de supply chain",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configuration CORS TRÈS PERMISSIVE pour le développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8082",
        "http://127.0.0.1:8082", 
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"  # Permet tout pour le dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Modèles de données
class QuestionRequest(BaseModel):
    query: str
    language: str = "fr"

class QuestionResponse(BaseModel):
    query: str
    response: str
    language: str

# Routes API
@app.get("/")
def root():
    return {
        "🚀": "RawaJAI FastAPI Server",
        "status": " ONLINE",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "http://localhost:5000/docs",
            "health": "http://localhost:5000/health", 
            "ask": "http://localhost:5000/ask",
            "tunnel": "http://localhost:5000/tunnel/status"
        },
        "message": "🎉 API FastAPI complètement fonctionnelle !"
    }

@app.get("/health")
def health():
    return {
        "status": "✅ healthy",
        "service": "RawaJAI FastAPI Backend",
        "api_type": "FastAPI",
        "port": 5000,
        "timestamp": datetime.now().isoformat(),
        "cors": "enabled",
        "endpoints_available": True
    }

@app.get("/tunnel/status")
def tunnel_status():
    return {
        "status": "✅ online",
        "message": "Backend FastAPI accessible via localhost:5000",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """
     Endpoint principal pour l'assistant IA
    """
    print(f"📝 Question reçue: '{request.query}' (langue: {request.language})")
    
    query = request.query.lower().strip()
    
    # Réponses intelligentes avec emojis
    if "bonjour" in query or "salut" in query or "hello" in query:
        response = "🎉 **Bonjour !** Je suis votre assistant RawaJAI.\n\n✨ Comment puis-je vous aider avec votre **supply chain** aujourd'hui ?\n\n💡 Tapez 'aide' pour voir mes capacités !"
        
    elif "aide" in query or "help" in query:
        response = """🔧 **Je peux vous aider avec :**

• 📊 **Prévisions de demande** - Analyser les tendances futures
• 📦 **Optimisation d'inventaire** - Calculer les stocks optimaux  
• 📈 **Analyse de données** - Interpréter vos métriques
• 🚚 **Gestion supply chain** - Optimiser vos flux logistiques
• 💡 **Conseils stratégiques** - Améliorer vos processus
• 🔍 **Détection d'anomalies** - Identifier les problèmes

**Que voulez-vous explorer ?** """

    elif "merci" in query or "thank" in query:
        response = "😊 **De rien !** C'est un plaisir de vous aider.\n\nN'hésitez pas pour d'autres questions sur votre supply chain ! 💪"
        
    elif "forecast" in query or "prévision" in query:
        response = "📊 **Prévisions de demande**\n\nJe peux analyser vos données historiques et prédire la demande future avec des algorithmes avancés.\n\n🔍 **Avez-vous des données spécifiques à analyser ?**"
        
    elif "inventory" in query or "inventaire" in query or "stock" in query:
        response = "📦 **Optimisation d'inventaire**\n\nJe calcule les niveaux optimaux de stock en tenant compte :\n• Coûts de stockage\n• Demande prévue\n• Délais d'approvisionnement\n• Niveau de service souhaité\n\n🎯 **Quels produits vous intéressent ?**"
        
    elif "supply chain" in query or "chaîne" in query or "logistique" in query:
        response = "🚚 **Supply Chain Management**\n\nJe peux vous aider à optimiser :\n• 📋 Planification des approvisionnements\n• 🏭 Gestion de production\n• 📦 Distribution et livraison\n• 💰 Réduction des coûts\n• ⚡ Amélioration de l'efficacité\n\n🎯 **Sur quel aspect voulez-vous vous concentrer ?**"
        
    else:
        response = f"""📝 **Question reçue :** "{request.query}"

🤖 Je suis spécialisé en **supply chain management** et je peux vous aider !

💡 **Suggestions :**
• Tapez **"aide"** pour voir toutes mes capacités
• Demandez-moi des **"prévisions"** 
• Parlez-moi d'**"inventaire"** ou de **"stock"**
• Posez des questions sur la **"supply chain"**

🚀 **Comment puis-je vous aider concrètement ?**"""
    
    print(f"✅ Réponse envoyée ({len(response)} caractères)")
    
    return QuestionResponse(
        query=request.query,
        response=response,
        language=request.language
    )

# Gestion des erreurs
@app.exception_handler(404)
def not_found_handler(request, exc):
    return {"error": "Endpoint non trouvé", "available_endpoints": ["/", "/docs", "/health", "/ask", "/tunnel/status"]}

@app.exception_handler(500)
def server_error_handler(request, exc):
    return {"error": "Erreur serveur", "message": str(exc)}

# Point d'entrée principal
if __name__ == "__main__":
    print("\n🚀 LANCEMENT DU SERVEUR RAWAJAI FASTAPI")
    print("=" * 60)
    print("📡 API URL: http://localhost:5000")
    print("📚 Documentation: http://localhost:5000/docs")
    print("🏥 Health Check: http://localhost:5000/health")
    print("🤖 Assistant: POST http://localhost:5000/ask")
    print("🔄 Auto-reload: Activé")
    print("⏹️  Arrêt: Ctrl+C")
    print("=" * 60)
    print()
    
    try:
        uvicorn.run(
            "fastapi_server:app",
            host="0.0.0.0",
            port=5000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n Serveur arrêté par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)
