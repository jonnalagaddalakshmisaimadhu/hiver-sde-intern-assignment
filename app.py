"""
FastAPI Server for @AppleSupport AI Customer Support Agent.
Provides REST API endpoints and serves the responsive modern Web UI.
"""
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.agent import AppleSupportAgent

app = FastAPI(
    title="@AppleSupport AI Customer Support Agent",
    description="Grounded AI Customer Support Agent for Hiver SDE Intern Assignment",
    version="1.0.0",
)

# Enable CORS for local testing flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lazy agent instance
_agent_instance = None


def get_agent() -> AppleSupportAgent:
    global _agent_instance
    if _agent_instance is None:
        print("[INFO] Initializing AppleSupportAgent for FastAPI server...")
        _agent_instance = AppleSupportAgent()
    return _agent_instance


class InquiryRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Incoming customer inquiry text")


class EvidenceItem(BaseModel):
    customer_message: str
    brand_response: str
    similarity: float


class InquiryResponse(BaseModel):
    query: str
    intent: str
    intent_confidence: float
    decision: str
    reason: str
    reply: str
    evidence: List[EvidenceItem]
    latency_ms: float


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "brand": "@AppleSupport",
        "intents_count": 8,
        "indexed_dialogues": 4800,
        "escalation_threshold": 0.38,
    }


@app.post("/api/inquire", response_model=InquiryResponse)
async def process_inquiry(req: InquiryRequest):
    """Process a customer inquiry through the full AI Agent pipeline."""
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Inquiry message cannot be empty.")

    agent = get_agent()
    t0 = time.time()
    resp = agent.process_inquiry(query)
    latency_ms = round((time.time() - t0) * 1000, 1)

    evidence_list = []
    for ev in resp.evidence:
        evidence_list.append(
            EvidenceItem(
                customer_message=ev.customer_message,
                brand_response=ev.brand_response,
                similarity=round(ev.similarity, 4),
            )
        )

    return InquiryResponse(
        query=query,
        intent=resp.intent,
        intent_confidence=round(resp.intent_confidence, 2),
        decision=resp.decision,
        reason=resp.reason,
        reply=resp.reply,
        evidence=evidence_list,
        latency_ms=latency_ms,
    )


# Serve UI static files
UI_DIR = Path(__file__).resolve().parent / "ui"
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the single-page application."""
    index_file = UI_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {"message": "UI directory not found. Please build ui/index.html."}


if __name__ == "__main__":
    import uvicorn

    print("[INFO] Starting @AppleSupport AI Agent Web Server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
