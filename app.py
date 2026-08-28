import streamlit as st
import json
import os
import requests
from agent_pipeline import run_agentic_pipeline

def check_and_pull_model(model_name: str) -> bool:
    """
    Prüft über die Ollama API, ob das Modell existiert.
    Wenn nicht, wird es heruntergeladen und ein Ladebalken in Streamlit angezeigt.
    """
    # Die OpenAI API URL endet auf /v1. Die native Ollama API nutzt die Basis-URL.
    base_v1_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    base_url = base_v1_url.replace("/v1", "")
    
    try:
        # 1. Existierende Modelle abfragen
        tags_res = requests.get(f"{base_url}/api/tags")
        tags_res.raise_for_status()
        models = [m["name"] for m in tags_res.json().get("models", [])]
        
        # Ollama hängt manchmal ':latest' an, wenn kein Tag angegeben wurde
        if model_name in models or f"{model_name}:latest" in models:
            return True

        # 2. Modell ist nicht da -> Lade es herunter
        st.warning(f"Modell '{model_name}' ist nicht lokal vorhanden. Lade herunter... (Das kann je nach Größe etwas dauern)")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        pull_res = requests.post(f"{base_url}/api/pull", json={"name": model_name}, stream=True)
        pull_res.raise_for_status()
        
        # 3. Stream auslesen und Fortschrittsbalken aktualisieren
        for line in pull_res.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                status = data.get("status", "Lade...")
                
                if "total" in data and "completed" in data:
                    total = data["total"]
                    completed = data["completed"]
                    if total > 0:
                        percent = completed / total
                        # Streamlit progress bar akzeptiert Float von 0.0 bis 1.0
                        progress_bar.progress(min(percent, 1.0))
                        status_text.text(f"{status}: {int(percent*100)}%")
                else:
                    status_text.text(status)
                    
        st.success(f"Modell '{model_name}' erfolgreich heruntergeladen!")
        return True
        
    except Exception as e:
        st.error(f"Fehler bei der Kommunikation mit Ollama: {e}")
        return False

def berechne_metriken(ground_truth: dict, predicted: dict):
    # Sicherstellen, dass ground_truth und predicted wirklich Dictionaries sind
    if not isinstance(ground_truth, dict): ground_truth = {}
    if not isinstance(predicted, dict): predicted = {}

    gt_set = set()
    for k, vals in ground_truth.items():
        if isinstance(vals, list):
            for v in vals:
                gt_set.add(f"{str(k).lower()}:{str(v).lower().strip()}")

    pred_set = set()
    for k, vals in predicted.items():
        if isinstance(vals, list):
            for v in vals:
                pred_set.add(f"{str(k).lower()}:{str(v).lower().strip()}")
                
    tp = len(gt_set.intersection(pred_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, tp, fp, fn

st.set_page_config(page_title="Agentic NER Pipeline", layout="wide")
st.title("🤖 Agentic AI Medical NER System")
st.markdown("Autonome Orchestrierung, Strategiewahl & Self-Refinement")

# Auswahlfeld für das Modell
selected_model = st.selectbox(
    "Wähle das LLM-Modell:",
    ("llama3", "llama3:8b", "qwen2:7b", "mistral") 
)

col1, col2 = st.columns(2)
with col1:
    text_input = st.text_area("Klinischer Text:", "Patient (m, 54) mit Verdacht auf Pneumonie. Aktuelle Medikation: 500mg Amoxicillin.")
with col2:
    gt_input = st.text_area("Ground Truth (JSON):", json.dumps({"Krankheit": ["Pneumonie"], "Medikament": ["Amoxicillin"]}, indent=2))

if st.button("Agenten-Pipeline starten", type="primary"):
    # 1. Zuerst prüfen ob das Modell da ist (und ggf. herunterladen)
    if check_and_pull_model(selected_model):
        
        # 2. Wenn das Modell bereit ist, starte die eigentliche Pipeline
        with st.spinner(f"Agenten arbeiten mit Modell '{selected_model}'..."):
            try:
                result = run_agentic_pipeline(text_input, model_name=selected_model)
                
                st.subheader("1. Orchestrator-Agent (Strategiewahl)")
                st.info(f"Der Agent wählte autonom: **{result.initial_strategy}**")
                if result.initial_strategy == "Chain-of-Thought":
                    st.write("**Gedankengang des Modells:**", result.gedankengang)
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.subheader("2. Extractor-Agent (Draft)")
                    st.json(result.initial_json)
                with col_res2:
                    st.subheader("3. Critic-Agent (Self-Refinement)")
                    st.json(result.refined_json)
                    
                st.subheader("📊 Quantitative Evaluation")
                gt_dict = json.loads(gt_input)
                
                # Evaluiere das verfeinerte Ergebnis
                p, r, f1, tp, fp, fn = berechne_metriken(gt_dict, result.refined_json)
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{p:.2f}")
                m2.metric("Recall", f"{r:.2f}")
                m3.metric("F1-Score", f"{f1:.2f}")
                st.write(f"True Positives: {tp} | False Positives: {fp} | False Negatives: {fn}")
                
            except Exception as e:
                st.error(f"Fehler bei der Pipeline-Ausführung: {e}")