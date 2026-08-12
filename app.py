import streamlit as st
import json
from agent_pipeline import run_agentic_pipeline

def berechne_metriken(ground_truth: dict, predicted: dict):
    # Alles wird mit str(v) sicher zu einem String konvertiert, bevor .lower() aufgerufen wird
    gt_set = {f"{k.lower()}:{str(v).lower().strip()}" for k, vals in ground_truth.items() if isinstance(vals, list) for v in vals}
    pred_set = {f"{k.lower()}:{str(v).lower().strip()}" for k, vals in predicted.items() if isinstance(vals, list) for v in vals}
                
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

col1, col2 = st.columns(2)
with col1:
    text_input = st.text_area("Klinischer Text:", "Patient (m, 54) mit Verdacht auf Pneumonie. Aktuelle Medikation: 500mg Amoxicillin.")
with col2:
    gt_input = st.text_area("Ground Truth (JSON):", json.dumps({"Krankheit": ["Pneumonie"], "Medikament": ["Amoxicillin"]}, indent=2))

if st.button("Agenten-Pipeline starten", type="primary"):
    with st.spinner("Agenten arbeiten..."):
        try:
            result = run_agentic_pipeline(text_input)
            
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
