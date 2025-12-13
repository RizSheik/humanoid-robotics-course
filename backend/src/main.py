from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the app
app = FastAPI(
    title="Educational Book RAG Chatbot API",
    description="API for educational content and RAG chatbot functionality",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes (to be added later)
from backend.src.routes import content, chat, user

# Mount the routes
app.include_router(content.router, prefix="/api", tags=["content"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(user.router, prefix="/api", tags=["user"])

from backend.src.database.connection import engine, Base

@app.on_event("startup")
async def startup_event():
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@app.get("/")
def read_root():
    return {"message": "Educational Book RAG Chatbot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)