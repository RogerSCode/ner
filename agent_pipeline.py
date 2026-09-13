"""
Kernlogik des Multi-Agenten-Systems für die Named Entity Recognition (NER).
Beinhaltet den Orchestrator-, Extractor- und Critic-Agenten.
"""

import json
from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
from pydantic import ValidationError
from models import ExtractionResult
from config import settings

# Der Client greift auf die zentralen, typsicheren Konfigurationen zu
local_client = OpenAI(
    base_url=settings.ollama_base_url,
    api_key="ollama",
    timeout=settings.api_timeout
)

class PipelineError(Exception):
    """Spezifische Fehlerklasse für Abbrüche innerhalb der Agenten-Pipeline."""
    pass

# Die erlaubten medizinischen Kategorien
ERLAUBTE_KATEGORIEN = {
    "Krankheit", "Medikament", "Wirkstaerke", "Form", 
    "Dosierung", "Verabreichungsweg", "Haeufigkeit", "Dauer"
}

def generate_with_retry_and_filter(messages: list, model_name: str, expects_gedankengang: bool = False, max_retries: int = 3) -> dict:
    """
    Sendet eine Anfrage an das lokale Sprachmodell und erzwingt eine valide JSON-Antwort.
    
    Diese Funktion implementiert einen automatischen Wiederholungsmechanismus.
    Sollte das Modell eine fehlerhafte Struktur oder unerlaubte Schlüssel generieren,
    wird es mit der entsprechenden Fehlermeldung zur iterativen Korrektur aufgefordert.
    
    Args:
        messages (list): Die Konversationshistorie (Prompts) für das Modell.
        model_name (str): Der Name des Sprachmodells.
        expects_gedankengang (bool): Gibt an, ob das Textfeld 'gedankengang' im JSON erlaubt ist.
        max_retries (int): Maximale Anzahl der Korrekturversuche.
        
    Returns:
        dict: Das validierte und gefilterte JSON-Wörterbuch.
        
    Raises:
        PipelineError: Wenn das Modell nach maximalen Versuchen scheitert oder die API ausfällt.
    """
    erlaubte_schluessel = set(ERLAUBTE_KATEGORIEN)
    if expects_gedankengang:
        erlaubte_schluessel.add("gedankengang")
        
    for versuch in range(max_retries):
        try:
            response = local_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            rohtext = response.choices[0].message.content
            
            geparstes_dict = json.loads(rohtext)
            fehler = []
            
            unbekannte_schluessel = set(geparstes_dict.keys()) - erlaubte_schluessel
            if unbekannte_schluessel:
                fehler.append(f"Unzulässige Schlüssel generiert: {list(unbekannte_schluessel)}. Erlaubt sind NUR: {list(erlaubte_schluessel)}.")
            
            for schluessel in ERLAUBTE_KATEGORIEN:
                if schluessel in geparstes_dict:
                    if not isinstance(geparstes_dict[schluessel], list):
                        fehler.append(f"Der Schlüssel '{schluessel}' muss eine Liste sein.")
                    else:
                        for element in geparstes_dict[schluessel]:
                            if not isinstance(element, str):
                                fehler.append(f"Elemente in '{schluessel}' müssen einfache Texte (Strings) sein.")

            if fehler and versuch < max_retries - 1:
                raise ValueError(" | ".join(fehler))

            gefiltertes_dict = {k: v for k, v in geparstes_dict.items() if k in erlaubte_schluessel}
            
            # Normalisierung der Listeninhalte
            for schluessel in ERLAUBTE_KATEGORIEN:
                if schluessel in gefiltertes_dict:
                    if isinstance(gefiltertes_dict[schluessel], str):
                        gefiltertes_dict[schluessel] = [gefiltertes_dict[schluessel]]
                    elif isinstance(gefiltertes_dict[schluessel], list):
                        bereinigte_liste = []
                        for element in gefiltertes_dict[schluessel]:
                            if isinstance(element, dict):
                                if "name" in element:
                                    bereinigte_liste.append(str(element["name"]))
                                elif element:
                                    bereinigte_liste.append(str(list(element.values())[0]))
                            else:
                                bereinigte_liste.append(str(element))
                        gefiltertes_dict[schluessel] = bereinigte_liste
                    else:
                        gefiltertes_dict[schluessel] = []
                else:
                    gefiltertes_dict[schluessel] = []
            
            return gefiltertes_dict
            
        except (json.JSONDecodeError, ValueError) as e:
            if versuch < max_retries - 1:
                fehlermeldung = f"Fehler in deiner Ausgabe: {e} Bitte korrigiere dies und gib AUSSCHLIESSLICH das validierte JSON-Format zurück."
                messages.append({"role": "assistant", "content": rohtext if rohtext else "{}"})
                messages.append({"role": "user", "content": fehlermeldung})
            else:
                raise PipelineError(f"Modell scheiterte nach {max_retries} Versuchen an Formatierungsfehlern. Letzter Fehler: {e}")
        
        # Spezifische Fehlerbehandlung für Netzwerk- und API-Probleme
        except APIConnectionError:
            raise PipelineError("Verbindungsfehler: Der lokale Ollama-Server ist nicht erreichbar. Bitte stelle sicher, dass Docker/Ollama läuft.")
        except APITimeoutError:
            raise PipelineError(f"Zeitüberschreitung: Das Modell '{model_name}' hat nach {settings.api_timeout} Sekunden nicht geantwortet. Ist der Rechner überlastet?")
        except APIStatusError as e:
            raise PipelineError(f"Ollama API-Fehler (Code {e.status_code}): {e.message}")
        except Exception as e:
            raise PipelineError(f"Unerwarteter Systemfehler bei der Generierung: {str(e)}")

def orchestrator_agent(text: str, model_name: str) -> str:
    """
    Analysiert den Rohtext und wählt dynamisch die optimale Prompting-Strategie aus.

    Args:
        text (str): Der medizinische Rohtext.
        model_name (str): Das zu verwendende Sprachmodell.

    Returns:
        str: Die gewählte Strategie ('Zero-Shot', 'Few-Shot' oder 'Chain-of-Thought').
    """
    system_anweisung = """Du bist der Orchestrator-Agent einer medizinischen NER-Pipeline. 
    Analysiere den Text und wähle die Prompting-Strategie:
    - 'Zero-Shot' für kurze Standardtexte.
    - 'Few-Shot' für Texte mit ungewöhnlichen Formaten/Abkürzungen.
    - 'Chain-of-Thought' für komplexe diagnostische Zusammenhänge.
    Antworte AUSSCHLIESSLICH mit exakt einem dieser drei Wörter."""
    
    try:
        response = local_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_anweisung}, {"role": "user", "content": text}],
            temperature=0.0
        )
        rohausgabe = response.choices[0].message.content.strip()
        if "Chain-of-Thought" in rohausgabe: return "Chain-of-Thought"
        if "Few-Shot" in rohausgabe: return "Few-Shot"
        return "Zero-Shot"
        
    except APIConnectionError:
        raise PipelineError("Orchestrator-Fehler: Keine Verbindung zu Ollama. Läuft der Dienst im Hintergrund?")
    except APITimeoutError:
        raise PipelineError(f"Orchestrator-Fehler: Zeitüberschreitung beim Zugriff auf Modell '{model_name}'.")
    except APIStatusError as e:
        raise PipelineError(f"Orchestrator-Fehler: API meldet Status {e.status_code}. Ist das Modell korrekt geladen?")
    except Exception as e:
        raise PipelineError(f"Unerwarteter Fehler im Orchestrator: {str(e)}")

def extractor_agent(text: str, strategy: str, model_name: str) -> dict:
    """
    Führt die initiale Informationsextraktion der Entitäten aus dem Text durch.

    Args:
        text (str): Der medizinische Rohtext.
        strategy (str): Die vom Orchestrator gewählte Prompting-Strategie.
        model_name (str): Das zu verwendende Sprachmodell.

    Returns:
        dict: Die unkorrigierten extrahierten Entitäten.
    """
    system_anweisung = f"""Du bist ein medizinischer Extraktions-Agent. Extrahiere Entitäten in folgende Kategorien: 
    {', '.join(ERLAUBTE_KATEGORIEN)}. Antworte AUSSCHLIESSLICH im gültigen JSON-Format."""
    
    nachrichten = [{"role": "system", "content": system_anweisung}]
    
    if strategy == "Few-Shot":
        nachrichten.append({"role": "user", "content": "Patient nimmt morgens 1x 500mg Ibuprofen Tablette."})
        nachrichten.append({"role": "assistant", "content": '{"Krankheit": [], "Medikament": ["Ibuprofen"], "Wirkstaerke": ["500mg"], "Form": ["Tablette"], "Dosierung": ["1x"], "Verabreichungsweg": [], "Haeufigkeit": ["morgens"], "Dauer": []}'})
    elif strategy == "Chain-of-Thought":
        system_anweisung += " Denke Schritt für Schritt. Schreibe deine Analyse zuerst in ein Feld 'gedankengang' (als Text), bevor du die Arrays befüllst."
        nachrichten[0]["content"] = system_anweisung
        
    nachrichten.append({"role": "user", "content": text})
    return generate_with_retry_and_filter(messages=nachrichten, model_name=model_name, expects_gedankengang=(strategy == "Chain-of-Thought"))

def critic_agent(text: str, initial_json: dict, model_name: str) -> dict:
    """
    Prüft das extrahierte JSON auf Fehler oder Lücken und korrigiert diese iterativ.

    Args:
        text (str): Der medizinische Originaltext.
        initial_json (dict): Das vom Extractor generierte Roh-JSON.
        model_name (str): Das zu verwendende Sprachmodell.

    Returns:
        dict: Das verfeinerte und bereinigte JSON.
    """
    system_anweisung = f"""Du bist ein strenger medizinischer Qualitätsprüfungs-Agent. 
    Du erhältst einen Originaltext und ein extrahiertes JSON.
    Prüfe: Wurden medizinische Entitäten übersehen oder falsch zugeordnet?
    REGELN:
    1. Erfinde NIEMALS neue JSON-Schlüssel! 
    2. Nutze AUSSCHLIESSLICH diese Kategorien: {', '.join(ERLAUBTE_KATEGORIEN)}.
    Gib AUSSCHLIESSLICH das korrigierte JSON-Objekt zurück."""
    
    bereinigtes_json = {k: v for k, v in initial_json.items() if k in ERLAUBTE_KATEGORIEN}
    nutzer_eingabe = f"Originaltext: {text}\n\nZu prüfendes JSON: {json.dumps(bereinigtes_json, ensure_ascii=False)}"
    
    nachrichten = [{"role": "system", "content": system_anweisung}, {"role": "user", "content": nutzer_eingabe}]
    return generate_with_retry_and_filter(messages=nachrichten, model_name=model_name, expects_gedankengang=False)

def run_agentic_pipeline(text: str, model_name: str = "llama3", forced_strategy: str = "Auto") -> ExtractionResult:
    """
    Orchestriert den gesamten Lebenszyklus der Informationsextraktion.
    
    Führt nacheinander den Orchestrator, Extractor und Critic aus und validiert
    die Ausgabe abschließend gegen das finale Pydantic-Modell.

    Args:
        text (str): Der zu verarbeitende medizinische Text.
        model_name (str): Das zu verwendende Sprachmodell.
        forced_strategy (str): Optionale Überschreibung der Agenten-Strategie.

    Returns:
        ExtractionResult: Das validierte Gesamtergebnis.
        
    Raises:
        PipelineError: Bei strukturellen oder netzwerkbedingten Abbrüchen.
    """
    # 1. Orchestrator-Phase 
    if forced_strategy and forced_strategy != "Auto":
        strategie = forced_strategy
    else:
        strategie = orchestrator_agent(text, model_name)
        
    # 2. Extraktions-Phase
    initiales_json = extractor_agent(text, strategie, model_name)
    
    # 3. Qualitätsprüfungs-Phase (Self-Refinement)
    if forced_strategy == "Zero-Shot":
        verfeinertes_json = initiales_json.copy()
    else:
        verfeinertes_json = critic_agent(text, initiales_json, model_name)
    
    # 4. Aufräumen der Datenstruktur für die Benutzeroberfläche
    roher_gedankengang = initiales_json.pop("gedankengang", "Kein Gedankengang formuliert")
    verfeinertes_json.pop("gedankengang", None)
    
    if not isinstance(roher_gedankengang, str):
        roher_gedankengang = json.dumps(roher_gedankengang, ensure_ascii=False)
    
    try:
        return ExtractionResult(
            initial_strategy=strategie,
            initial_json=initiales_json,
            refined_json=verfeinertes_json,
            gedankengang=roher_gedankengang
        )
    except ValidationError as e:
        raise PipelineError(f"Struktureller Validierungsfehler im finalen Modell: {e}")