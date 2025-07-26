from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="RawaJAI Backend API", version="1.0.0")

# Configuration CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8082",  # Expo web
        "http://localhost:3000",  # React dev server
        "http://127.0.0.1:8082",
        "http://127.0.0.1:3000",
        "*"  
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

@app.get("/")
async def root():
    return {"message": "RawaJAI Backend API is running!", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RawaJAI Backend"}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint pour poser des questions à l'assistant IA
    """
    try:
        if "bonjour" in request.query.lower():
            response_text = f"Bonjour ! Je suis ravi de vous aider. Vous avez dit : '{request.query}'. Comment puis-je vous assister avec votre supply chain aujourd'hui ?"
        elif "help" in request.query.lower() or "aide" in request.query.lower():
            response_text = "Je peux vous aider avec la gestion de votre supply chain, les prévisions, l'optimisation d'inventaire, et l'analyse de données. Que souhaitez-vous savoir ?"
        else:
            response_text = f"J'ai bien reçu votre question : '{request.query}'. Je suis en cours de développement et je peux vous aider avec des questions sur la supply chain, les prévisions et l'optimisation d'inventaire."
        
        return QuestionResponse(
            query=request.query,
            response=response_text,
            language=request.language
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement de la question: {str(e)}")

@app.get("/tunnel/status")
async def tunnel_status():
    return {"status": "online", "message": "Backend accessible via localhost:5000"}

if __name__ == "__main__":
    print("🚀 Démarrage du serveur RawaJAI Backend...")
    print("📡 API disponible sur: http://localhost:5000")
    print("📚 Documentation: http://localhost:5000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
