# agents.py
import numpy as np
import librosa
import tensorflow as tf
import os
from langchain_openai import ChatOpenAI  # updated import for new versions

# --- DataAgent ---
class DataAgent:
    def __init__(self, sr=12000, n_mels=128, spec_len=128):
        self.sr = sr
        self.n_mels = n_mels
        self.spec_len = spec_len

    def load(self, file_path):
        """Handles both MP3 and NumPy spectrograms."""
        if file_path.endswith(".mp3"):
            y, _ = librosa.load(file_path, sr=self.sr)
            # Convert to mel-spectrogram
            S = librosa.feature.melspectrogram(
                y=y, sr=self.sr, n_fft=2048, hop_length=256,
                n_mels=self.n_mels, power=2.0
            )
            Sdb = librosa.power_to_db(S, ref=np.max)

            # Pad/trim to fixed size
            if Sdb.shape[1] < self.spec_len:
                pad = self.spec_len - Sdb.shape[1]
                Sdb = np.pad(Sdb, ((0,0),(0,pad)), mode='constant')
            else:
                Sdb = Sdb[:, :self.spec_len]

            return Sdb[..., np.newaxis]  # (128, 128, 1)

        elif file_path.endswith(".npy"):
            return np.load(file_path)

        else:
            raise ValueError("Unsupported file format! Use .mp3 or .npy")


# --- PreprocAgent ---
class PreprocAgent:
    def preprocess(self, spectrogram):
        return np.expand_dims(spectrogram, axis=0)  # batch dimension


# --- ModelAgent ---
class ModelAgent:
    def __init__(self, model_path="cnn_bearing.h5", label_encoder_path="label_encoder.pkl"):
        self.model = tf.keras.models.load_model(model_path)
        import pickle
        with open(label_encoder_path, "rb") as f:
            self.le = pickle.load(f)

    def predict(self, spectrogram):
        probs = self.model.predict(spectrogram)
        pred_idx = np.argmax(probs, axis=1)[0]
        return self.le.inverse_transform([pred_idx])[0], probs[0]


# --- DecisionAgent ---
class DecisionAgent:
    def decide(self, fault_type):
        actions = {
            "Normal": "No action needed. Continue monitoring.",
            "Ball": "Ball fault detected → schedule bearing replacement.",
            "InnerRace": "Inner race fault detected → inspect inner track.",
            "OuterRace": "Outer race fault detected → inspect outer track."
        }
        return actions.get(fault_type, "Unknown fault type.")


# --- ExplainAgent ---
class ExplainAgent:
    def __init__(self, api_key=None):
        if api_key:
            self.llm = ChatOpenAI(
                model="openai/gpt-oss-20b",           # ✅ switch model
                api_key=api_key,
                base_url="https://api.studio.nebius.ai/v1"  # ✅ Nebius endpoint
            )
        else:
            self.llm = None

    def explain(self, fault_type, action):
        if self.llm:
            response = self.llm.invoke(
                f"The system detected {fault_type} fault. Recommended action: {action}. "
                f"Explain this clearly for a maintenance engineer."
            )
            return response.content
        else:
            return f"The detected fault is {fault_type}. Recommended action: {action}."
