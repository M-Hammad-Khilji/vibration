# interface_agent/main.py
import os
import tempfile
import traceback
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import numpy as np
import librosa
import soundfile as sf
import requests
import pickle
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # loads .env at repo root if present

# config via env (or fallback defaults)
MODEL_PATH = os.getenv("MODEL_PATH", str(ROOT / "model" / "bearing_cnn_model.h5"))
LE_PATH = os.getenv("LABEL_ENCODER_PATH", str(ROOT / "model" / "label_encoder.pkl"))
EXPLAIN_AGENT_URL = os.getenv("EXPLAIN_AGENT_URL", "http://localhost:8100/explain")
SR = int(os.getenv("SAMPLE_RATE", "12000"))
N_MELS = int(os.getenv("N_MELS", "128"))
SPEC_LEN = int(os.getenv("SPEC_LEN", "128"))

app = FastAPI(title="InterfaceAgent - Bearing Vibration")

# --------------------------
# Pipeline components (compact in this file)
# --------------------------
class DataAgent:
    def __init__(self, sr=SR, n_mels=N_MELS, spec_len=SPEC_LEN):
        self.sr = sr
        self.n_mels = n_mels
        self.spec_len = spec_len

    def load_audio_file(self, filepath):
        # librosa handles mp3/wav, returns mono
        y, _ = librosa.load(filepath, sr=self.sr, mono=True)
        return y

    def audio_to_melspec(self, y):
        S = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=2048, hop_length=256, n_mels=self.n_mels, power=2.0
        )
        Sdb = librosa.power_to_db(S, ref=np.max)
        # pad/trim to spec_len
        if Sdb.shape[1] < self.spec_len:
            pad = self.spec_len - Sdb.shape[1]
            Sdb = np.pad(Sdb, ((0,0),(0,pad)), mode="constant")
        else:
            Sdb = Sdb[:, :self.spec_len]
        return Sdb[..., np.newaxis].astype(np.float32)

class PreprocAgent:
    def preprocess(self, spec):
        return np.expand_dims(spec, axis=0)  # (1, n_mels, spec_len, 1)

class ModelAgent:
    def __init__(self, model_path=MODEL_PATH, le_path=LE_PATH):
        self.model = tf.keras.models.load_model(model_path)
        with open(le_path, "rb") as f:
            self.le = pickle.load(f)

    def predict(self, X):
        probs = self.model.predict(X)[0]
        idx = int(np.argmax(probs))
        label = self.le.inverse_transform([idx])[0]
        return label, probs

class DecisionAgent:
    def decide(self, label):
        mapping = {
            "Normal": "No action needed. Continue monitoring.",
            "Ball": "Ball fault suspected — schedule bearing inspection/replacement.",
            "InnerRace": "Inner race fault — inspect inner race ASAP.",
            "OuterRace": "Outer race fault — inspect outer race ASAP."
        }
        return mapping.get(label, "Unknown fault type — specialist review required.")

# instantiate objects once
data_agent = DataAgent()
preproc_agent = PreprocAgent()
model_agent = ModelAgent()
decision_agent = DecisionAgent()

# --------------------------
# API
# --------------------------
class PredictResponse(BaseModel):
    fault: str
    confidence: float
    recommended_action: str
    explanation: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), prompt: str = Form("check the status of the bearing")):
    # Save uploaded file to a temporary file
    suffix = Path(file.filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Pipeline steps
        y = data_agent.load_audio_file(tmp_path)
        spec = data_agent.audio_to_melspec(y)
        X = preproc_agent.preprocess(spec)
        label, probs = model_agent.predict(X)
        confidence = float(max(probs))
        action = decision_agent.decide(label)

        # Call explain agent
        payload = {
            "fault": label,
            "action": action,
            "confidence": confidence,
            "prompt": prompt
        }
        try:
            r = requests.post(EXPLAIN_AGENT_URL, json=payload, timeout=20)
            if r.status_code == 200:
                explanation = r.json().get("explanation", "")
            else:
                explanation = f"Explain agent responded with status {r.status_code}: {r.text}"
        except Exception as e:
            explanation = f"Failed to call explain agent: {e}"

        return {
            "fault": label,
            "confidence": confidence,
            "recommended_action": action,
            "explanation": explanation
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "fault": "error",
            "confidence": 0.0,
            "recommended_action": "error",
            "explanation": f"Processing error: {e}"
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("INTERFACE_AGENT_PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
