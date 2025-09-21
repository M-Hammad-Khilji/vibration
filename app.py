# app.py
import streamlit as st
import os
from dotenv import load_dotenv
from agents import DataAgent, PreprocAgent, ModelAgent, DecisionAgent, ExplainAgent

# --- Load environment variables ---
load_dotenv()  # this reads .env file
API_KEY = os.getenv("NEBIUS_API_KEY")

st.title("🔧 Bearing Fault Diagnosis (MP3 + AI Agents)")
uploaded_file = st.file_uploader("Upload vibration data (.mp3)", type=["mp3"])

if uploaded_file is not None:
    # Save temp file
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # --- Run pipeline ---
    data_agent = DataAgent()
    preproc_agent = PreprocAgent()
    model_agent = ModelAgent("bearing_cnn_model.h5", "label_encoder.pkl")
    decision_agent = DecisionAgent()
    explain_agent = ExplainAgent(api_key=API_KEY)

    spectrogram = data_agent.load(temp_path)
    X = preproc_agent.preprocess(spectrogram)
    fault, probs = model_agent.predict(X)
    action = decision_agent.decide(fault)
    explanation = explain_agent.explain(fault, action)

    st.success(f"✅ Fault detected: **{fault}**")
    st.info(f"📌 Recommended action: {action}")
    st.write("💡 Explanation:", explanation)

    # Debug info
    st.caption(f"🔑 API key loaded: {bool(API_KEY)}")
