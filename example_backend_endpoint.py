from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from agents.agen_perkaba import get_perkaba_agent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: Optional[str] = None
    debug_mode: bool = False

class ChatResponse(BaseModel):
    response: str
    user_id: str
    session_id: str

@app.post("/chat/perkaba", response_model=ChatResponse)
async def chat_with_perkaba_agent(request: ChatRequest):
    try:
        # Buat agent dengan user_id dan session_id dari frontend
        agent = get_perkaba_agent(
            user_id=request.user_id,
            session_id=request.session_id or f"session_{request.user_id}",
            debug_mode=request.debug_mode
        )
        
        # Agent akan otomatis load memory lama jika ada
        # dan save conversation history untuk session berikutnya
        response = agent.run(request.message)
        
        return ChatResponse(
            response=response.content,
            user_id=request.user_id,
            session_id=agent.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """Endpoint untuk mendapatkan daftar session user"""
    # Implementasi untuk mendapatkan daftar session dari database
    # Bisa query PostgresStorage untuk mendapatkan session yang ada
    pass

@app.delete("/chat/sessions/{user_id}/{session_id}")
async def clear_session_memory(user_id: str, session_id: str):
    """Endpoint untuk menghapus memory session tertentu"""
    # Implementasi untuk menghapus memory session
    pass 