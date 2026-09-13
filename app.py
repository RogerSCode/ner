"""
Streamlit-Benutzeroberfläche für das agentenbasierte medizinische NER-System.
Bietet Modi für den klinischen Einsatz (Arzt) und die Evaluation (Data Science).
"""

import streamlit as st
import json
import os
import requests
import pandas as pd
from datetime import datetime
from agent_pipeline import run_agentic_pipeline, PipelineError, ERLAUBTE_KATEGORIEN
from metrics import berechne_metriken
from config import settings

# -------------------------------------------------------------------
# Hilfsfunktionen
# -------------------------------------------------------------------
def check_and_pull_model(model_name: str) -> bool:
    """
    Prüft, ob ein lokales Ollama-Modell verfügbar ist, und lädt es andernfalls herunter.

    Args:
        model_name (str): Der Name des LLMs (z. B. 'llama3').

    Returns:
        bool: True, wenn das Modell verfügbar ist oder erfolgreich geladen wurde. False bei Fehlern.
    """
    base_v1_url = settings.ollama_base_url
    base_url = base_v1_url.replace("/v1", "")
    try:
        tags_antwort = requests.get(f"{base_url}/api/tags", timeout=10)
        tags_antwort.raise_for_status()
        modelle = [m["name"] for m in tags_antwort.json().get("models", [])]
        if model_name in modelle or f"{model_name}:latest" in modelle:
            return True

        st.warning(f"Modell '{model_name}' fehlt lokal. Lade herunter... (Das kann dauern)")
        fortschrittsbalken = st.progress(0.0)
        status_text = st.empty()
        download_antwort = requests.post(f"{base_url}/api/pull", json={"name": model_name}, stream=True)
        download_antwort.raise_for_status()
        
        for zeile in download_antwort.iter_lines():
            if zeile:
                daten = json.loads(zeile.decode('utf-8'))
                status = daten.get("status", "Lade...")
                if "total" in daten and "completed" in daten:
                    gesamt = daten["total"]
                    abgeschlossen = daten["completed"]
                    if gesamt > 0:
                        prozent = abgeschlossen / gesamt
                        fortschrittsbalken.progress(min(prozent, 1.0))
                        status_text.text(f"{status}: {int(prozent*100)}%")
                else:
                    status_text.text(status)
        st.success(f"Modell '{model_name}' erfolgreich heruntergeladen!")
        return True
    except Exception as e:
        st.error(f"Kommunikationsfehler mit Ollama: {e}")
        return False

def dict_to_dataframe(daten_dict: dict) -> pd.DataFrame:
    """
    Wandelt das strukturierte JSON in eine flache, bearbeitbare Datentabelle um.
    
    Args:
        daten_dict (dict): Die extrahierten Entitäten.
        
    Returns:
        pd.DataFrame: Tabellarische Darstellung der Daten.
    """
    zeilen = []
    for kategorie in ERLAUBTE_KATEGORIEN:
        for wert in daten_dict.get(kategorie, []):
            zeilen.append({"Kategorie": kategorie, "Extrahierter Wert": wert})
    if not zeilen:
        zeilen.append({"Kategorie": "Medikament", "Extrahierter Wert": ""})
    return pd.DataFrame(zeilen)

def save_to_db(originaltext: str, validiertes_dict: dict):
    """
    Speichert den freigegebenen Datensatz dauerhaft in der Referenzdatenbank.
    
    Args:
        originaltext (str): Der medizinische Rohtext.
        validiertes_dict (dict): Die vom Arzt geprüften und freigegebenen Entitäten.
    """
    datenbank_datei = "ground_truth_db.json"
    eintrag = {
        "timestamp": datetime.now().isoformat(),
        "text": originaltext,
        "validated_entities": validiertes_dict
    }
    datenbank_daten = []
    if os.path.exists(datenbank_datei):
        with open(datenbank_datei, 'r', encoding='utf-8') as datei:
            try: datenbank_daten = json.load(datei)
            except: pass
    datenbank_daten.append(eintrag)
    with open(datenbank_datei, 'w', encoding='utf-8') as datei:
        json.dump(datenbank_daten, datei, indent=2, ensure_ascii=False)

# -------------------------------------------------------------------
# Streamlit Benutzeroberfläche
# -------------------------------------------------------------------
st.set_page_config(page_title="Agentenbasiertes NER System", layout="wide")
st.title("🤖 Agentenbasiertes Medizinisches NER-System")

# Zustand der Sitzung initialisieren (verhindert das Verschwinden der Daten beim Editieren)
if "pipeline_ergebnis" not in st.session_state:
    st.session_state.pipeline_ergebnis = None
if "aktueller_text" not in st.session_state:
    st.session_state.aktueller_text = ""

with st.sidebar:
    st.header("⚙️ System-Konfiguration")
    
    anwendungs_modus = st.radio(
        "Wähle den Anwendungs-Modus:",
        ("1️⃣ Klinische Anwendung (Arzt)", "2️⃣ System-Evaluation (Data Science)")
    )
    
    st.divider()
    
    # Modellauswahl nutzt die validierten Konfigurationen aus config.py
    standard_modell = settings.default_model
    verfuegbare_modelle = ["llama3", "llama3:8b", "qwen2:7b", "mistral"]
    if standard_modell not in verfuegbare_modelle:
        verfuegbare_modelle.insert(0, standard_modell)
    gewaehltes_modell = st.selectbox("LLM-Modell:", verfuegbare_modelle, index=verfuegbare_modelle.index(standard_modell))

    strategie_optionen = ["Auto", "Zero-Shot", "Few-Shot", "Chain-of-Thought"]
    gewaehlte_strategie = st.selectbox(
        "Prompting-Strategie:", 
        strategie_optionen, 
        index=0,
        help="Wähle 'Auto' für den autonomen Orchestrator oder erzwinge eine Strategie für Vergleichsmessungen."
    )

if anwendungs_modus == "1️⃣ Klinische Anwendung (Arzt)":
    st.markdown("In diesem Modus können unstrukturierte Arztbriefe eingefügt werden. Die KI extrahiert die Entitäten, welche anschließend fachlich validiert und in der Datenbank gespeichert werden können.")
    
    st.header("1. Datenvorbereitung (Frontend)")
    texteingabe = st.text_area("Neuer klinischer Text (Arztbrief):", "Patient (m, 54) klagt über Dyspnoe. Aktuelle Medikation: 500mg Amoxicillin p.o. zweimal täglich.")
    
    st.header("2. Inferenz-Dienst (KI-Analyse)")
    if st.button("🚀 Text analysieren", type="primary"):
        if not texteingabe.strip() or len(texteingabe.strip()) < 10:
            st.warning("Bitte gib einen sinnvollen klinischen Text ein.")
        elif check_and_pull_model(gewaehltes_modell):
            with st.spinner(f"Agenten analysieren den Text (Modell: {gewaehltes_modell})..."):
                try:
                    st.session_state.pipeline_ergebnis = run_agentic_pipeline(
                        texteingabe, 
                        model_name=gewaehltes_modell, 
                        forced_strategy=gewaehlte_strategie
                    )
                    st.session_state.aktueller_text = texteingabe
                except PipelineError as fehler:
                    st.error(f"🛑 Abbruch durch die Pipeline: {fehler}")
                except Exception as fehler:
                    st.error(f"⚠️ Systemfehler: {fehler}")

    if st.session_state.pipeline_ergebnis and st.session_state.aktueller_text == texteingabe:
        ergebnis = st.session_state.pipeline_ergebnis
        
        st.subheader("Modellierungs-Dienst (Orchestrator)")
        if gewaehlte_strategie == "Auto":
            st.info(f"🤖 Die KI wählte autonom die Strategie: **{ergebnis.initial_strategy}**")
        else:
            st.info(f"👤 Strategie manuell erzwungen: **{ergebnis.initial_strategy}** (Orchestrator übersprungen)")
            
        with st.expander("KI-Gedankengang anzeigen (Transparenz)"):
            st.write(ergebnis.gedankengang)

        st.header("3. Manuelle Überprüfung & Fachliche Freigabe (Human-in-the-Loop)")
        st.markdown("Prüfe die von der KI extrahierten Daten. Füge vergessene Werte hinzu oder korrigiere Fehler.")
        
        # Datentabelle erstellen und interaktiv bearbeitbar machen
        daten_tabelle = dict_to_dataframe(ergebnis.refined_json.model_dump())
        bearbeitete_tabelle = st.data_editor(
            daten_tabelle, 
            num_rows="dynamic", 
            use_container_width=True,
            column_config={
                "Kategorie": st.column_config.SelectboxColumn("Kategorie", options=list(ERLAUBTE_KATEGORIEN), required=True)
            }
        )
        
        st.header("4. Referenzdatenbank (Persistenz)")
        if st.button("💾 Als validiert markieren & speichern", type="secondary"):
            validiertes_woerterbuch = {kat: [] for kat in ERLAUBTE_KATEGORIEN}
            for _, zeile in bearbeitete_tabelle.iterrows():
                kategorie = zeile.get("Kategorie")
                wert = zeile.get("Extrahierter Wert")
                if pd.notna(kategorie) and pd.notna(wert) and str(wert).strip():
                    validiertes_woerterbuch[kategorie].append(str(wert).strip())
                    
            save_to_db(st.session_state.aktueller_text, validiertes_woerterbuch)
            st.success("Erfolgreich persistiert! Die Daten stehen nun als neuer Goldstandard zur Verfügung.")

elif anwendungs_modus == "2️⃣ System-Evaluation (Data Science)":
    st.markdown("Dieser Modus dient der Qualitätsmessung der Agenten-Pipeline anhand eines bestehenden Goldstandards.")
    
    st.header("1. Test-Datensatz laden")
    hochgeladene_datei = st.file_uploader("Lade einen konvertierten Datensatz hoch (.json)", type=["json"])
    
    if hochgeladene_datei is not None:
        try:
            batch_daten = json.load(hochgeladene_datei)
            st.success(f"Datensatz mit {len(batch_daten)} Einträgen erfolgreich geladen!")
            
            st.header("2. Stapelverarbeitung zur Evaluierung (Batch-Modus)")
            st.markdown("Da lokale Sprachmodelle Zeit für die Inferenz benötigen, wähle eine repräsentative Stichprobengröße.")
            
            stichprobengroesse = st.number_input("Wie viele Texte sollen evaluiert werden?", min_value=1, max_value=len(batch_daten), value=10)
            
            if st.button("📊 Evaluierung starten", type="primary"):
                if check_and_pull_model(gewaehltes_modell):
                    
                    # Zähler für die Auswertung über den gesamten Stapel
                    gesamt_treffer, gesamt_falsch_positiv, gesamt_uebersehen = 0, 0, 0
                    
                    fortschrittsbalken = st.progress(0.0)
                    status_text = st.empty()
                    
                    with st.spinner(f"Evaluiere {stichprobengroesse} Texte mit {gewaehltes_modell} (Strategie: {gewaehlte_strategie})..."):
                        
                        for i in range(stichprobengroesse):
                            test_text = batch_daten[i].get("text", "")
                            referenz_dict = batch_daten[i].get("ground_truth", {})
                            
                            status_text.text(f"Analysiere Text {i+1} von {stichprobengroesse}...")
                            
                            try:
                                ergebnis = run_agentic_pipeline(
                                    test_text, 
                                    model_name=gewaehltes_modell, 
                                    forced_strategy=gewaehlte_strategie
                                )
                                
                                _, _, _, treffer, falsch_positiv, uebersehen = berechne_metriken(referenz_dict, ergebnis.refined_json.model_dump())
                                
                                gesamt_treffer += treffer
                                gesamt_falsch_positiv += falsch_positiv
                                gesamt_uebersehen += uebersehen
                                
                            except Exception as fehler:
                                st.error(f"Fehler bei Text {i+1}: {fehler}")
                                
                            fortschrittsbalken.progress((i + 1) / stichprobengroesse)
                            
                        status_text.text("Evaluierung abgeschlossen!")
                        
                        # Berechnung der finalen Metriken
                        precision = gesamt_treffer / (gesamt_treffer + gesamt_falsch_positiv) if (gesamt_treffer + gesamt_falsch_positiv) > 0 else 0.0
                        recall = gesamt_treffer / (gesamt_treffer + gesamt_uebersehen) if (gesamt_treffer + gesamt_uebersehen) > 0 else 0.0
                        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                        
                        st.header("3. Quantitatives Gesamtergebnis")
                        st.info(f"**Modell:** {gewaehltes_modell} | **Strategie:** {gewaehlte_strategie} | **Stichprobengröße:** {stichprobengroesse} Texte")
                        
                        spalte1, spalte2, spalte3 = st.columns(3)
                        
                        spalte1.metric(
                            "Globale Precision", 
                            f"{precision:.2f}", 
                            help="Genauigkeit: Wie viel Prozent der von der KI extrahierten Entitäten waren tatsächlich richtig? (Ein niedriger Wert bedeutet viele 'Halluzinationen' / False Positives)."
                        )
                        spalte2.metric(
                            "Globaler Recall", 
                            f"{recall:.2f}", 
                            help="Trefferquote: Wie viel Prozent der im Originaltext vorhandenen Entitäten hat die KI gefunden? (Ein niedriger Wert bedeutet, dass viel übersehen wurde / False Negatives)."
                        )
                        spalte3.metric(
                            "Globaler F1-Score", 
                            f"{f1_score:.2f}", 
                            help="Das harmonische Mittel aus Precision und Recall. Dient als einzelner, ausbalancierter Score für die Gesamtqualität des Modells."
                        )
                        
                        st.markdown("### Fehler-Analyse über den gesamten Datensatz")
                        st.write(f"**True Positives (Korrekt gefunden):** {gesamt_treffer}")
                        st.write(f"**False Positives (Halluzinationen):** {gesamt_falsch_positiv}")
                        st.write(f"**False Negatives (Übersehen):** {gesamt_uebersehen}")
                        
        except Exception as fehler:
            st.error(f"Fehler beim Lesen der Datei: {fehler}")
    else:
        st.info("Bitte lade die Testdaten hoch, um die Evaluation zu starten.")