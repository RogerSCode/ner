import streamlit as st
import json
import os
import requests
import pandas as pd
from datetime import datetime
from agent_pipeline import run_agentic_pipeline, PipelineError, ALLOWED_CATEGORIES
from metrics import berechne_metriken
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



def dict_to_dataframe(data_dict: dict) -> pd.DataFrame:
    """Wandelt das JSON in eine flache, bearbeitbare Tabelle um."""
    rows = []
    for cat in ALLOWED_CATEGORIES:
        for val in data_dict.get(cat, []):
            rows.append({"Kategorie": cat, "Extrahierter Wert": val})
    # Falls leer, eine leere Zeile als Startpunkt anbieten
    if not rows:
        rows.append({"Kategorie": "Medikament", "Extrahierter Wert": ""})
    return pd.DataFrame(rows)

def save_to_db(original_text: str, validated_dict: dict):
    """Speichert den freigegebenen Datensatz in der Ground-Truth-DB."""
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

# Session State initialisieren (verhindert das Verschwinden der Daten beim Editieren)
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "current_text" not in st.session_state:
    st.session_state.current_text = ""

with st.sidebar:
    st.header("⚙️ System-Konfiguration")
    
    # Der Anwendungs-Modus trennt die Logik
    app_mode = st.radio(
        "Wähle den Anwendungs-Modus:",
        ("1️⃣ Klinische Anwendung (Arzt)", "2️⃣ System-Evaluation (Data Science)")
    )
    
    st.divider()
    
    # Modellauswahl
    default_env_model = os.environ.get("DEFAULT_MODEL", "llama3")
    available_models = ["llama3", "llama3:8b", "qwen2:7b", "mistral"]
    if default_env_model not in available_models:
        available_models.insert(0, default_env_model)
    selected_model = st.selectbox("LLM-Modell:", available_models, index=available_models.index(default_env_model))

    # Strategieauswahl
    strategy_options = ["Auto", "Zero-Shot", "Few-Shot", "Chain-of-Thought"]
    selected_strategy = st.selectbox(
        "Prompting-Strategie:", 
        strategy_options, 
        index=0,
        help="Wähle 'Auto' für den Agentic Orchestrator oder erzwinge eine Strategie für die Baseline-Messung."
    )


# =====================================================================
# MODUS 1: KLINISCHE ANWENDUNG (Human-in-the-Loop, Persistenz)
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
                    st.session_state.pipeline_result = run_agentic_pipeline(
                        text_input, 
                        model_name=selected_model, 
                        forced_strategy=selected_strategy
                    )
                    st.session_state.current_text = text_input
                except PipelineError as pe:
                    st.error(f"🛑 Abbruch durch die Pipeline: {pe}")
                except Exception as e:
                    st.error(f"⚠️ Systemfehler: {e}")

    # Sobald ein Ergebnis da ist und es zum aktuellen Text passt, anzeigen
    if st.session_state.pipeline_result and st.session_state.current_text == text_input:
        res = st.session_state.pipeline_result
        
        st.subheader("Modeling-Service (Orchestrator)")
        if selected_strategy == "Auto":
            st.info(f"🤖 Die KI wählte autonom die Strategie: **{res.initial_strategy}**")
        else:
            st.info(f"👤 Strategie manuell erzwungen: **{res.initial_strategy}** (Orchestrator übersprungen)")
            
        with st.expander("KI-Gedankengang anzeigen (Transparenz)"):
            st.write(res.gedankengang)

        st.header("3. Human Review & Fachliche Freigabe")
        st.markdown("Prüfe die von der KI extrahierten Daten. Füge vergessene Werte hinzu oder korrigiere Fehler.")
        
        # DataFrame bauen und bearbeitbar machen
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
            # Wandelt den DataFrame zurück in ein Dictionary, das nur ALLOWED_CATEGORIES als Keys hat
            edited_dict = {cat: [] for cat in ALLOWED_CATEGORIES}
            for _, row in edited_df.iterrows():
                cat = row.get("Kategorie")
                val = row.get("Extrahierter Wert")
                if pd.notna(cat) and pd.notna(val) and str(val).strip():
                    edited_dict[cat].append(str(val).strip())
                    
            save_to_db(st.session_state.current_text, edited_dict)
            st.success("Erfolgreich persistiert! Die Daten stehen nun als neuer Goldstandard in der Datenbank zur Verfügung.")


# =====================================================================
# MODUS 2: SYSTEM-EVALUATION (Qualitätsmessung gegen Goldstandard)
# =====================================================================
elif app_mode == "2️⃣ System-Evaluation (Data Science)":
    st.markdown("Dieser Modus dient der Qualitätsmessung (Precision, Recall, F1) der Agenten-Pipeline anhand eines bestehenden Goldstandards (Batch-Evaluierung).")
    
    st.header("1. Test-Datensatz laden")
    uploaded_file = st.file_uploader("Lade einen konvertierten Datensatz hoch (.json)", type=["json"])
    
    if uploaded_file is not None:
        try:
            batch_data = json.load(uploaded_file)
            st.success(f"Datensatz mit {len(batch_data)} Einträgen erfolgreich geladen!")
            
            st.header("2. Batch-Evaluierung (Modell-Qualität messen)")
            st.markdown("Da lokale LLMs Zeit für die Inferenz benötigen, wähle eine repräsentative Stichprobengröße (z.B. 20-50) für deine Baseline-Messung.")
            
            sample_size = st.number_input("Wie viele Texte sollen evaluiert werden?", min_value=1, max_value=len(batch_data), value=10)
            
            if st.button("📊 Batch-Evaluierung starten", type="primary"):
                if check_and_pull_model(selected_model):
                    
                    # Globale Zähler für die Metriken über das gesamte Batch
                    global_tp, global_fp, global_fn = 0, 0, 0
                    
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    
                    with st.spinner(f"Evaluiere {sample_size} Texte mit {selected_model} (Strategie: {selected_strategy})..."):
                        
                        # Schleife über die gewünschte Anzahl an Texten
                        for i in range(sample_size):
                            text_eval = batch_data[i].get("text", "")
                            gt_dict = batch_data[i].get("ground_truth", {})
                            
                            status_text.text(f"Analysiere Text {i+1} von {sample_size}...")
                            
                            try:
                                # Pipeline aufrufen
                                result = run_agentic_pipeline(
                                    text_eval, 
                                    model_name=selected_model, 
                                    forced_strategy=selected_strategy
                                )
                                
                                # Lokale Metriken für diesen einen Text berechnen
                                _, _, _, tp, fp, fn = berechne_metriken(gt_dict, result.refined_json.model_dump())
                                
                                # Zu den globalen Metriken addieren
                                global_tp += tp
                                global_fp += fp
                                global_fn += fn
                                
                            except Exception as e:
                                st.error(f"Fehler bei Text {i+1}: {e}")
                                
                            # Fortschrittsbalken aktualisieren
                            progress_bar.progress((i + 1) / sample_size)
                            
                        status_text.text("Evaluierung abgeschlossen!")
                        
                        # Globale Metriken über das gesamte Batch berechnen
                        p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
                        r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
                        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
                        
                        # Ergebnisse präsentieren
                        st.header("3. Quantitatives Gesamtergebnis")
                        st.info(f"**Modell:** {selected_model} | **Strategie:** {selected_strategy} | **Sample Size:** {sample_size} Texte")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Globale Precision", f"{p:.2f}")
                        m2.metric("Globaler Recall", f"{r:.2f}")
                        m3.metric("Globaler F1-Score", f"{f1:.2f}")
                        
                        st.markdown("### Fehler-Analyse über das gesamte Batch")
                        st.write(f"**True Positives (Korrekt gefunden):** {global_tp}")
                        st.write(f"**False Positives (Halluzinationen):** {global_fp}")
                        st.write(f"**False Negatives (Übersehen):** {global_fn}")
                        
        except Exception as e:
            st.error(f"Fehler beim Lesen der Datei: {e}")
    else:
        st.info("Bitte lade die `app_testdaten.json` hoch, um die Evaluation zu starten.")
        
    st.header("2. Benchmark-Messung durchführen")
    if st.button("📊 Evaluierung starten", type="primary") and uploaded_file is not None:
        if check_and_pull_model(selected_model):
            with st.spinner(f"Agenten arbeiten (Strategie: {selected_strategy})..."):
                try:
                    result = run_agentic_pipeline(
                        text_eval, 
                        model_name=selected_model, 
                        forced_strategy=selected_strategy
                    )
                    
                    st.subheader("Modeling-Service (Orchestrator)")
                    if selected_strategy == "Auto":
                        st.info(f"🤖 Die KI wählte autonom die Strategie: **{result.initial_strategy}**")
                    else:
                        st.info(f"👤 Strategie manuell erzwungen: **{result.initial_strategy}**")
                        
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