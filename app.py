import streamlit as st
import json
import os
import requests
import pandas as pd
from datetime import datetime
from agent_pipeline import run_agentic_pipeline, PipelineError, ALLOWED_CATEGORIES

# -------------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------------
def check_and_pull_model(model_name: str) -> bool:
    base_v1_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    base_url = base_v1_url.replace("/v1", "")
    try:
        tags_res = requests.get(f"{base_url}/api/tags", timeout=10)
        tags_res.raise_for_status()
        models = [m["name"] for m in tags_res.json().get("models", [])]
        if model_name in models or f"{model_name}:latest" in models:
            return True

        st.warning(f"Modell '{model_name}' fehlt lokal. Lade herunter... (Das kann dauern)")
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
    except Exception as e:
        st.error(f"Kommunikationsfehler mit Ollama: {e}")
        return False

def berechne_metriken(ground_truth: dict, predicted: dict):
    if not isinstance(ground_truth, dict): ground_truth = {}
    if not isinstance(predicted, dict): predicted = {}
    gt_set = set()
    for k, vals in ground_truth.items():
        if isinstance(vals, list):
            for v in vals: gt_set.add(f"{str(k).lower()}:{str(v).lower().strip()}")
    pred_set = set()
    for k, vals in predicted.items():
        if isinstance(vals, list):
            for v in vals: pred_set.add(f"{str(k).lower()}:{str(v).lower().strip()}")
            
    tp = len(gt_set.intersection(pred_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn

def dict_to_dataframe(data_dict: dict) -> pd.DataFrame:
    rows = []
    for cat in ALLOWED_CATEGORIES:
        for val in data_dict.get(cat, []):
            rows.append({"Kategorie": cat, "Extrahierter Wert": val})
    if not rows:
        rows.append({"Kategorie": "Krankheit", "Extrahierter Wert": ""})
    return pd.DataFrame(rows)

def save_to_db(original_text: str, validated_dict: dict):
    db_file = "ground_truth_db.json"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "text": original_text,
        "validated_entities": validated_dict
    }
    db_data = []
    if os.path.exists(db_file):
        with open(db_file, 'r', encoding='utf-8') as f:
            try: db_data = json.load(f)
            except: pass
    db_data.append(entry)
    with open(db_file, 'w', encoding='utf-8') as f:
        json.dump(db_data, f, indent=2, ensure_ascii=False)

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Agentic NER Pipeline", layout="wide")
st.title("🤖 Agentic AI Medical NER System")

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "current_text" not in st.session_state:
    st.session_state.current_text = ""

with st.sidebar:
    st.header("⚙️ System-Konfiguration")
    
    # Der Anwendungs-Modus trennt die Logik!
    app_mode = st.radio(
        "Wähle den Anwendungs-Modus:",
        ("1️⃣ Klinische Anwendung (Arzt)", "2️⃣ System-Evaluation (Data Science)")
    )
    
    st.divider()
    
    default_env_model = os.environ.get("DEFAULT_MODEL", "llama3")
    available_models = ["llama3", "llama3:8b", "qwen2:7b", "mistral"]
    if default_env_model not in available_models:
        available_models.insert(0, default_env_model)
    selected_model = st.selectbox("LLM-Modell:", available_models, index=available_models.index(default_env_model))


# =====================================================================
# MODUS 1: KLINISCHE ANWENDUNG (Für den Arzt, keine GT vorhanden)
# =====================================================================
if app_mode == "1️⃣ Klinische Anwendung (Arzt)":
    st.markdown("In diesem Modus können unstrukturierte Arztbriefe eingefügt werden. Die KI extrahiert die Entitäten, welche anschließend fachlich validiert und in der Datenbank gespeichert werden können.")
    
    st.header("1. Data-Prep-Frontend")
    text_input = st.text_area("Neuer klinischer Text (Arztbrief):", "Patient (m, 54) klagt über Dyspnoe. Aktuelle Medikation: 500mg Amoxicillin p.o. zweimal täglich.")
    
    st.header("2. Deployment- & Eval-Dienst (Inferenz)")
    if st.button("🚀 Text analysieren", type="primary"):
        if not text_input.strip() or len(text_input.strip()) < 10:
            st.warning("Bitte gib einen sinnvollen klinischen Text ein.")
        elif check_and_pull_model(selected_model):
            with st.spinner(f"Agenten analysieren den Text (Modell: {selected_model})..."):
                try:
                    st.session_state.pipeline_result = run_agentic_pipeline(text_input, model_name=selected_model)
                    st.session_state.current_text = text_input
                except PipelineError as pe:
                    st.error(f"🛑 Abbruch durch die Pipeline: {pe}")
                except Exception as e:
                    st.error(f"⚠️ Systemfehler: {e}")

    if st.session_state.pipeline_result and st.session_state.current_text == text_input:
        res = st.session_state.pipeline_result
        
        st.subheader("Modeling-Service (Orchestrator)")
        st.info(f"Die KI wählte autonom die Strategie: **{res.initial_strategy}**")
        with st.expander("KI-Gedankengang anzeigen (Transparenz)"):
            st.write(res.gedankengang)

        st.header("3. Human Review & Fachliche Freigabe")
        st.markdown("Prüfe die von der KI extrahierten Daten. Füge vergessene Werte hinzu oder korrigiere Fehler.")
        
        df = dict_to_dataframe(res.refined_json.model_dump())
        edited_df = st.data_editor(
            df, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "Kategorie": st.column_config.SelectboxColumn("Kategorie", options=list(ALLOWED_CATEGORIES), required=True)
            }
        )
        
        st.header("4. Ground-Truth-DB (Persistenz)")
        if st.button("💾 Als validiert markieren & in DB speichern", type="secondary"):
            edited_dict = {cat: [] for cat in ALLOWED_CATEGORIES}
            for _, row in edited_df.iterrows():
                cat = row.get("Kategorie")
                val = row.get("Extrahierter Wert")
                if pd.notna(cat) and pd.notna(val) and str(val).strip():
                    edited_dict[cat].append(str(val).strip())
                    
            save_to_db(st.session_state.current_text, edited_dict)
            st.success("Erfolgreich persistiert! Die Daten stehen nun als neuer Goldstandard in der Datenbank zur Verfügung.")


# =====================================================================
# MODUS 2: SYSTEM-EVALUATION (Für den Data Scientist / Meilenstein 3)
# =====================================================================
elif app_mode == "2️⃣ System-Evaluation (Data Science)":
    st.markdown("Dieser Modus dient der Qualitätsmessung (Precision, Recall, F1) der Agenten-Pipeline anhand eines bestehenden Goldstandards (z. B. GERNERMED).")
    
    st.header("1. Test-Datensatz laden")
    uploaded_file = st.file_uploader("Lade einen konvertierten Datensatz hoch (.json)", type=["json"])
    
    text_eval = ""
    gt_eval = "{}"
    
    if uploaded_file is not None:
        try:
            batch_data = json.load(uploaded_file)
            entry_idx = st.number_input(f"Wähle einen Datensatz (0 bis {len(batch_data)-1}):", min_value=0, max_value=len(batch_data)-1, value=0)
            text_eval = batch_data[entry_idx].get("text", "")
            gt_dict = batch_data[entry_idx].get("ground_truth", {})
            gt_eval = json.dumps(gt_dict, indent=2, ensure_ascii=False)
            
            col_t, col_g = st.columns(2)
            with col_t:
                st.text_area("Zu analysierender Text:", text_eval, disabled=True)
            with col_g:
                st.text_area("Geladene Ground Truth (Referenz):", gt_eval, disabled=True)
                
        except Exception as e:
            st.error(f"Fehler beim Lesen der Datei: {e}")
    else:
        st.info("Bitte lade eine JSON-Datei mit Text und Ground Truth hoch, um die Evaluation zu starten.")
        
    st.header("2. Benchmark-Messung durchführen")
    if st.button("📊 Evaluierung starten", type="primary") and uploaded_file is not None:
        if check_and_pull_model(selected_model):
            with st.spinner(f"Agenten arbeiten..."):
                try:
                    result = run_agentic_pipeline(text_eval, model_name=selected_model)
                    
                    st.subheader("Ergebnis der Agenten-Pipeline")
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown("**1. Draft (Extractor-Agent)**")
                        st.json(result.initial_json.model_dump())
                    with col_r2:
                        st.markdown("**2. Refined (Critic-Agent)**")
                        st.json(result.refined_json.model_dump())
                    
                    st.header("3. Quantitative Evaluation")
                    p, r, f1, tp, fp, fn = berechne_metriken(gt_dict, result.refined_json.model_dump())
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Precision", f"{p:.2f}")
                    m2.metric("Recall", f"{r:.2f}")
                    m3.metric("F1-Score", f"{f1:.2f}")
                    st.write(f"**True Positives:** {tp} | **False Positives:** {fp} | **False Negatives:** {fn}")
                    
                except PipelineError as pe:
                    st.error(f"🛑 Abbruch: {pe}")
                except Exception as e:
                    st.error(f"⚠️ Systemfehler: {e}")