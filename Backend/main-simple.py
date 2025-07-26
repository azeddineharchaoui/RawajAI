from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
from datetime import datetime

app = FastAPI(title="RawaJAI Backend API", version="1.0.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8082",  # Expo web
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:8082",
        "http://127.0.0.1:3000",
        "*"  # Pour le développement
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    query: str
    language: str = "en"

class QuestionResponse(BaseModel):
    query: str
    response: str
    language: str

# Réponses prédéfinies pour simuler l'IA
RESPONSES = {
    "bonjour": "Bonjour ! Je suis ravi de vous aider. Comment puis-je vous assister avec votre supply chain aujourd'hui ?",
    "hello": "Hello! I'm happy to help you. How can I assist you with your supply chain today?",
    "aide": "Je peux vous aider avec :\n• Prévisions de demande\n• Optimisation d'inventaire\n• Analyse de données\n• Gestion de la supply chain\n\nQue souhaitez-vous savoir ?",
    "help": "I can help you with:\n• Demand forecasting\n• Inventory optimization\n• Data analysis\n• Supply chain management\n\nWhat would you like to know?",
    "forecast": "Pour les prévisions, je peux analyser vos données historiques et prédire la demande future. Avez-vous des données spécifiques à analyser ?",
    "inventory": "Pour l'optimisation d'inventaire, je peux vous aider à calculer les niveaux optimaux de stock. Quels sont vos produits principaux ?",
    "supply chain": "La supply chain implique la gestion des flux de produits, d'informations et de finances. Sur quel aspect souhaitez-vous vous concentrer ?",
}

@app.get("/")
async def root():
    return {
        "message": "RawaJAI Backend API is running!",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RawaJAI Backend",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint pour poser des questions à l'assistant IA
    """
    try:
        query_lower = request.query.lower().strip()
        
        # Chercher une réponse correspondante
        response_text = None
        for keyword, response in RESPONSES.items():
            if keyword in query_lower:
                response_text = response
                break
        
        # Réponse par défaut si aucun mot-clé trouvé
        if not response_text:
            if request.language == "fr":
                response_text = f"J'ai bien reçu votre question : '{request.query}'. Je suis spécialisé dans la supply chain. Vous pouvez me demander de l'aide sur les prévisions, l'inventaire, ou l'analyse de données."
            else:
                response_text = f"I received your question: '{request.query}'. I specialize in supply chain management. You can ask me about forecasting, inventory, or data analysis."
        
        return QuestionResponse(
            query=request.query,
            response=response_text,
            language=request.language
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement: {str(e)}")

@app.get("/tunnel/status")
async def tunnel_status():
    return {
        "status": "online",
        "message": "Backend accessible via localhost:5000",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🚀 Démarrage du serveur RawaJAI Backend...")
    print("📡 API disponible sur: http://localhost:5000")
    print("📚 Documentation: http://localhost:5000/docs")
    print("🔄 Appuyez sur Ctrl+C pour arrêter")
    
    uvicorn.run(
        "main-simple:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
