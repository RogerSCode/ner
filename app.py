import streamlit as st
import json
import os
import requests
from agent_pipeline import run_agentic_pipeline, PipelineError

def check_and_pull_model(model_name: str) -> bool:
    base_v1_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    base_url = base_v1_url.replace("/v1", "")
    
    try:
        tags_res = requests.get(f"{base_url}/api/tags", timeout=10)
        tags_res.raise_for_status()
        models = [m["name"] for m in tags_res.json().get("models", [])]
        
        if model_name in models or f"{model_name}:latest" in models:
            return True

        st.warning(f"Modell '{model_name}' ist nicht lokal vorhanden. Lade herunter... (Das kann dauern)")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        pull_res = requests.post(f"{base_url}/api/pull", json={"name": model_name}, stream=True)
        pull_res.raise_for_status()
        
        for line in pull_res.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                status = data.get("status", "Lade...")
                
                if "total" in data and "completed" in data:
                    total = data["total"]
                    completed = data["completed"]
                    if total > 0:
                        percent = completed / total
                        progress_bar.progress(min(percent, 1.0))
                        status_text.text(f"{status}: {int(percent*100)}%")
                else:
                    status_text.text(status)
                    
        st.success(f"Modell '{model_name}' erfolgreich heruntergeladen!")
        return True
        
    except requests.exceptions.Timeout:
        st.error("Fehler: Zeitüberschreitung bei der Verbindung zu Ollama. Läuft der Container?")
        return False
    except Exception as e:
        st.error(f"Fehler bei der Kommunikation mit Ollama: {e}")
        return False

def berechne_metriken(ground_truth: dict, predicted: dict):
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
    
    if not text_input.strip():
        st.warning("Bitte gib einen klinischen Text ein.")
        st.stop()
        
    if len(text_input.strip()) < 10:
        st.warning("Der klinische Text ist sehr kurz. Bitte gib einen sinnvollen Satz ein.")
        st.stop()
        
    try:
        gt_dict = json.loads(gt_input)
    except json.JSONDecodeError:
        st.error("Die eingegebene Ground Truth ist kein gültiges JSON-Format. Bitte korrigieren.")
        st.stop()
    
    if check_and_pull_model(selected_model):
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
                    # Durch Pydantic-Update nutzen wir nun .model_dump()
                    st.json(result.initial_json.model_dump())
                with col_res2:
                    st.subheader("3. Critic-Agent (Self-Refinement)")
                    st.json(result.refined_json.model_dump())
                    
                st.subheader("📊 Quantitative Evaluation")
                
                # Auch bei der Metriken-Übergabe wird nun aus dem Modell ein Dictionary extrahiert
                p, r, f1, tp, fp, fn = berechne_metriken(gt_dict, result.refined_json.model_dump())
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{p:.2f}")
                m2.metric("Recall", f"{r:.2f}")
                m3.metric("F1-Score", f"{f1:.2f}")
                st.write(f"True Positives: {tp} | False Positives: {fp} | False Negatives: {fn}")
                
            except PipelineError as pe:
                st.error(f"🛑 Abbruch durch die Agenten-Pipeline: {pe}")
            except Exception as e:
                st.error(f"⚠️ Unerwarteter Systemfehler: {e}")