import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi_socketio import SocketManager
import uvicorn
import rag  # Ensure `rag.py` exists in the same directory

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("❌ API_KEY is missing in .env file! Please check and restart.")
    exit(1)

# Initialize FastAPI app and Socket.IO
app = FastAPI(title="📄 RAG Chatbot with Azure OpenAI")
sio = SocketManager(app=app)

@app.get("/")
def read_root():
    """API Root Endpoint"""
    return {"message": "✅ Welcome to the RAG Chatbot API!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and index it using FAISS.
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise ValueError("The uploaded file is empty.")

        msg = rag.load_and_index_document(file_bytes)
        return {"status": "success", "detail": msg}

    except Exception as e:
        logger.exception("❌ Error uploading document")
        return {"status": "error", "detail": str(e)}

@app.get("/query")
def query(q: str):
    """
    Query the indexed document for relevant information.
    """
    try:
        if not q.strip():
            raise ValueError("Query cannot be empty.")

        answer, sources = rag.get_answer(q)
        return {"answer": answer, "sources": sources}

    except Exception as e:
        logger.exception("❌ Error processing query")
        return {"error": str(e)}

# ---------------- Socket.IO Handlers ----------------

@sio.on("connect")
async def on_connect(sid, environ):
    """Handle client connection to Socket.IO"""
    logger.info(f"🔗 Client connected: {sid}")

@sio.on("ask_question")
async def on_ask_question(sid, data):
    """
    Handle live chat via Socket.IO
    """
    query_text = data.get("query", "").strip()
    if not query_text:
        await sio.emit("answer", {"error": "Query cannot be empty."}, room=sid)
        return

    logger.info(f"📥 Received query from client: {query_text}")

    try:
        answer, sources = rag.get_answer(query_text)
        await sio.emit("answer", {"answer": answer, "sources": sources}, room=sid)
    
    except Exception as e:
        logger.exception("❌ Error processing query")
        await sio.emit("answer", {"error": str(e)}, room=sid)

# ---------------- Run the Server ----------------

if __name__ == "__main__":
    logger.info("🚀 Starting FastAPI Server with Uvicorn...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)