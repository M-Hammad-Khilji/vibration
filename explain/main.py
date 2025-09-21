# explain_agent/main.py
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", None)
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# Try to import LangChain OpenAI wrapper
try:
    from langchain_openai import ChatOpenAI
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

app = FastAPI(title="ExplainAgent - Bearing Report")

class ExplainRequest(BaseModel):
    fault: str
    action: str
    confidence: float = 0.0
    prompt: str = "Please summarize for maintenance engineer."

class ExplainResponse(BaseModel):
    explanation: str

@app.get("/health")
def health():
    return {"status": "ok", "llm_available": LLM_AVAILABLE}

@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest):
    # If LLM configured & package available, use it
    if LLM_AVAILABLE and NEBIUS_API_KEY:
        # create llm client
        llm = ChatOpenAI(model=LLM_MODEL, api_key=NEBIUS_API_KEY, base_url="https://api.studio.nebius.ai/v1")
        prompt = (
            f"You are an experienced vibration maintenance engineer.\n"
            f"A bearing has been detected with fault '{req.fault}' (confidence {req.confidence:.2f}). "
            f"Recommended action: {req.action}.\nUser prompt: {req.prompt}\n\n"
            "Create a short maintenance report (3-6 sentences) explaining the issue, urgency, "
            "what to inspect/replace, and any safety notes."
        )
        resp = llm.predict(prompt)
        # Normalize response
        if isinstance(resp, str):
            text = resp
        elif hasattr(resp, "content"):
            text = resp.content
        else:
            text = str(resp)
        return {"explanation": text}
    else:
        # fallback
        text = (
            f"FAULT REPORT:\nDetected: {req.fault} (confidence {req.confidence:.2f}).\n"
            f"Recommended action: {req.action}\n"
            "Notes: Inspect the bearing and consider replacement if defect confirmed. "
            "Collect vibration logs and avoid running under heavy load until inspected."
        )
        return {"explanation": text}

if __name__ == "__main__":
    port = int(os.getenv("EXPLAIN_AGENT_PORT", "8100"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
