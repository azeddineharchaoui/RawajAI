"""
RawaJAI Backend API Server
Serveur FastAPI pour l'assistant IA
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import sys
import os

print(" DÉMARRAGE API RAWAJAI")
print("=" * 40)

# Créer l'application FastAPI
app = FastAPI(
    title=" RawaJAI API",
    description="Assistant IA pour la Supply Chain",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        "🚀": "RawaJAI Backend API",
        "status": "✅ ONLINE",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health", 
            "ask": "/ask",
            "tunnel": "/tunnel/status"
        },
        "message": "API FastAPI fonctionnelle !"
    }

@app.get("/health")
def health():
    return {
        "status": "✅ healthy",
        "service": "RawaJAI Backend",
        "api": "FastAPI",
        "port": 5000
    }

@app.get("/tunnel/status")
def tunnel_status():
    return {
        "status": "✅ online",
        "message": "Backend accessible via localhost:5000"
    }

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """
    🤖 Endpoint principal pour l'assistant IA
    """
    print(f"📝 Question reçue: {request.query}")
    
    query = request.query.lower().strip()
    
    # Réponses intelligentes
    if "bonjour" in query or "salut" in query or "hello" in query:
        response = "🎉 Bonjour ! Je suis votre assistant RawaJAI. Comment puis-je vous aider avec votre supply chain aujourd'hui ?"
        
    elif "aide" in query or "help" in query:
        response = """🔧 Je peux vous aider avec :
        
• 📊 **Prévisions de demande** - Analyser les tendances
• 📦 **Optimisation d'inventaire** - Calculer les stocks optimaux  
• 📈 **Analyse de données** - Interpréter vos métriques
• 🚚 **Gestion supply chain** - Optimiser vos flux
• 💡 **Conseils stratégiques** - Améliorer vos processus

Que voulez-vous explorer ?"""

    elif "merci" in query or "thank" in query:
        response = "😊 De rien ! C'est un plaisir de vous aider. N'hésitez pas pour d'autres questions !"
        
    elif "forecast" in query or "prévision" in query:
        response = "📊 **Prévisions de demande** : Je peux analyser vos données historiques et prédire la demande future. Avez-vous des données spécifiques à analyser ?"
        
    elif "inventory" in query or "inventaire" in query or "stock" in query:
        response = "📦 **Optimisation d'inventaire** : Je calcule les niveaux optimaux de stock en tenant compte des coûts et de la demande. Quels produits vous intéressent ?"
        
    elif "supply chain" in query or "chaîne" in query:
        response = "🚚 **Supply Chain** : Je peux vous aider à optimiser vos flux, réduire les coûts et améliorer l'efficacité. Sur quel aspect voulez-vous vous concentrer ?"
        
    else:
        response = f"""📝 J'ai reçu votre question : "{request.query}"

🤖 Je suis spécialisé en **supply chain management**. 

💡 **Suggestions** :
• Tapez "aide" pour voir mes capacités
• Demandez-moi des "prévisions" 
• Parlez-moi d'"inventaire"
• Posez des questions sur la "supply chain"

Comment puis-je vous aider ?"""
    
    print(f"✅ Réponse envoyée: {response[:50]}...")
    
    return QuestionResponse(
        query=request.query,
        response=response,
        language=request.language
    )

# Point d'entrée principal
if __name__ == "__main__":
    print("\n🚀 LANCEMENT DU SERVEUR RAWAJAI")
    print("=" * 50)
    print("📡 API URL: http://localhost:5000")
    print("📚 Documentation: http://localhost:5000/docs")
    print("🔄 Auto-reload: Activé")
    print("⏹️  Arrêt: Ctrl+C")
    print("=" * 50)
    print()
    
    try:
        uvicorn.run(
            "api_server:app",
            host="0.0.0.0",
            port=5000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n Serveur arrêté par l'utilisateur")
    except Exception as e:
        print(f" Erreur: {e}")
