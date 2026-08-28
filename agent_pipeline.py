import os
import json
from openai import OpenAI
from pydantic import ValidationError
from models import ExtractionResult

local_client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    timeout=120.0
)

class PipelineError(Exception):
    """Eine benutzerdefinierte Ausnahme für alle Fehler innerhalb der Agenten-Pipeline."""
    pass

def generate_with_retry_and_filter(messages: list, model_name: str, expects_gedankengang: bool = False, max_retries: int = 3) -> dict:
    """
    Führt den LLM-Aufruf aus. Prüft das JSON aktiv auf Strukturfehler.
    Gibt dem Modell präzises Feedback zu erfundenen Schlüsseln oder falschen Typen.
    Wendet nur beim allerletzten Versuch einen harten Auto-Filter als Fallback an.
    """
    allowed_keys = {"Krankheit", "Medikament"}
    if expects_gedankengang:
        allowed_keys.add("gedankengang")
        
    for attempt in range(max_retries):
        try:
            response = local_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            raw_content = response.choices[0].message.content
            
            # 1. JSON parsen (kann json.JSONDecodeError werfen)
            parsed_dict = json.loads(raw_content)
            
            # 2. Aktive Struktur-Prüfung für das Modell-Feedback
            errors = []
            
            # Prüfe auf erfundene Schlüssel
            extra_keys = set(parsed_dict.keys()) - allowed_keys
            if extra_keys:
                errors.append(f"Unzulässige Schlüssel generiert: {list(extra_keys)}. Erlaubt sind NUR: {list(allowed_keys)}.")
            
            # Prüfe auf falsche Datentypen in den Listen
            for key in ["Krankheit", "Medikament"]:
                if key in parsed_dict:
                    if not isinstance(parsed_dict[key], list):
                        errors.append(f"Der Schlüssel '{key}' muss eine Liste sein, ist aber {type(parsed_dict[key]).__name__}.")
                    else:
                        for item in parsed_dict[key]:
                            if not isinstance(item, str):
                                errors.append(f"Elemente in '{key}' müssen einfache Strings sein. Gefunden: {item} (Typ: {type(item).__name__}).")

            # Wenn wir im 1. oder 2. Versuch sind und Fehler gefunden haben, werfen wir 
            # absichtlich eine Ausnahme, um die Feedback-Schleife auszulösen.
            if errors and attempt < max_retries - 1:
                raise ValueError(" | ".join(errors))

            # === AB HIER: Erfolgreich ODER wir sind im allerletzten Versuch (Fallback-Filter greift) ===
            
            # Filtern: Alle komplett unzulässigen Keys rigoros entfernen
            filtered_dict = {k: v for k, v in parsed_dict.items() if k in allowed_keys}
            
            # Typen-Korrektur: Listen-Inhalte reparieren (Rettungsanker)
            for key in ["Krankheit", "Medikament"]:
                if key in filtered_dict:
                    if isinstance(filtered_dict[key], str):
                        filtered_dict[key] = [filtered_dict[key]]
                    elif isinstance(filtered_dict[key], list):
                        cleaned_list = []
                        for item in filtered_dict[key]:
                            if isinstance(item, dict):
                                if "name" in item:
                                    cleaned_list.append(str(item["name"]))
                                elif item:
                                    cleaned_list.append(str(list(item.values())[0]))
                            else:
                                cleaned_list.append(str(item))
                        filtered_dict[key] = cleaned_list
                    else:
                        filtered_dict[key] = []
                else:
                    filtered_dict[key] = [] # Standardmäßig leere Liste setzen
            
            return filtered_dict
            
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                # Das ist das Herzstück: Wir geben dem Modell exaktes Feedback, was es falsch gemacht hat!
                error_msg = f"Fehler in deiner Ausgabe: {e} Bitte korrigiere dies und gib AUSSCHLIESSLICH das validierte JSON-Format zurück."
                
                # Füge die kaputte Antwort und die Fehlermeldung dem Kontext hinzu
                messages.append({"role": "assistant", "content": raw_content if raw_content else "{}"})
                messages.append({"role": "user", "content": error_msg})
            else:
                # Das passiert nur, wenn das Modell nach 3 Versuchen noch immer kein lesbares JSON generiert
                raise PipelineError(f"Das Modell konnte nach {max_retries} Versuchen kein gültiges JSON generieren. Letzter Fehler: {e}")
        except Exception as e:
            raise PipelineError(f"Unerwarteter API-Fehler bei der Generierung: {str(e)}")
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                # Feedback-Schleife
                error_msg = f"Dein letzter Output war kein valides JSON. Parser-Fehler: {e}. Bitte korrigiere den Fehler und gib AUSSCHLIESSLICH gültiges JSON zurück."
                messages.append({"role": "assistant", "content": raw_content if raw_content else "{}"})
                messages.append({"role": "user", "content": error_msg})
            else:
                raise PipelineError(f"Das Modell konnte nach {max_retries} Versuchen kein gültiges JSON generieren.")
        except Exception as e:
            raise PipelineError(f"Unerwarteter API-Fehler bei der Generierung: {str(e)}")

def orchestrator_agent(text: str, model_name: str) -> str:
    system_prompt = """Du bist der Orchestrator-Agent einer medizinischen NER-Pipeline. 
    Analysiere den folgenden Text hinsichtlich Komplexität, Domäne und Sprache. 
    Wähle die am besten geeignete Prompting-Strategie:
    - 'Zero-Shot' für kurze, einfache Standardtexte.
    - 'Few-Shot' für Texte mit ungewöhnlichen Formaten oder vielen Abkürzungen.
    - 'Chain-of-Thought' für komplexe diagnostische Zusammenhänge.
    
    Antworte AUSSCHLIESSLICH mit exakt einem dieser drei Wörter. Keine Erklärungen."""
    
    try:
        response = local_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        raw_output = response.choices[0].message.content.strip()
        
        if "Chain-of-Thought" in raw_output: return "Chain-of-Thought"
        if "Few-Shot" in raw_output: return "Few-Shot"
        return "Zero-Shot"
    
    except Exception as e:
        raise PipelineError(f"Der Orchestrator-Agent konnte nicht antworten. API-Fehler: {str(e)}")

def extractor_agent(text: str, strategy: str, model_name: str) -> dict:
    system_prompt = """Du bist ein medizinischer NER-Agent. Extrahiere Entitäten in die Kategorien 
    'Krankheit' und 'Medikament'. Antworte AUSSCHLIESSLICH im gültigen JSON-Format."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if strategy == "Few-Shot":
        messages.append({"role": "user", "content": "Patient hat Migräne, nimmt Ibuprofen."})
        messages.append({"role": "assistant", "content": '{"Krankheit": ["Migräne"], "Medikament": ["Ibuprofen"]}'})
    elif strategy == "Chain-of-Thought":
        system_prompt += " Denke Schritt für Schritt. Schreibe deine Analyse zuerst in ein Feld 'gedankengang' (als einfachen Text-String), bevor du die Arrays für 'Krankheit' und 'Medikament' befüllst."
        messages[0]["content"] = system_prompt
        
    messages.append({"role": "user", "content": text})
    
    return generate_with_retry_and_filter(
        messages=messages, 
        model_name=model_name, 
        expects_gedankengang=(strategy == "Chain-of-Thought")
    )

def critic_agent(text: str, initial_json: dict, model_name: str) -> dict:
    system_prompt = """Du bist ein strenger medizinischer Qualitäts-Agent. 
    Du erhältst einen Originaltext und ein extrahiertes JSON.
    Prüfe: Wurden medizinische Entitäten übersehen?
    
    REGELN:
    1. Erfinde NIEMALS neue JSON-Schlüssel (Keys)! 
    2. Nutze AUSSCHLIESSLICH die Kategorien 'Krankheit' und 'Medikament'.
    3. Ignoriere Alter, Geschlecht oder Dosierungen.
    
    Gib AUSSCHLIESSLICH das korrigierte JSON-Objekt zurück."""
    
    # Hier nehmen wir eine Kopie, damit wir den Gedankengang nicht für den Prompt mitschicken, falls er stört
    clean_json_for_prompt = {k: v for k, v in initial_json.items() if k in ["Krankheit", "Medikament"]}
    
    user_prompt = f"Originaltext: {text}\n\nZu prüfendes JSON: {json.dumps(clean_json_for_prompt, ensure_ascii=False)}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return generate_with_retry_and_filter(
        messages=messages, 
        model_name=model_name, 
        expects_gedankengang=False
    )

def run_agentic_pipeline(text: str, model_name: str = "llama3") -> ExtractionResult:
    strategy = orchestrator_agent(text, model_name)
    initial_json = extractor_agent(text, strategy, model_name)
    refined_json = critic_agent(text, initial_json, model_name)
    
    # 1. Wir speichern den Gedankengang und ENTFERNEN (pop) ihn gleichzeitig aus dem initial_json.
    # Dadurch bleibt nur noch das strikte Format {"Krankheit": [], "Medikament": []} übrig, was Pydantic glücklich macht.
    raw_gedankengang = initial_json.pop("gedankengang", "Kein Gedankengang (da kein Chain-of-Thought)")
    refined_json.pop("gedankengang", None) # Zur Sicherheit auch beim Critic entfernen, falls er es halluziniert hat
    
    if not isinstance(raw_gedankengang, str):
        raw_gedankengang = json.dumps(raw_gedankengang, ensure_ascii=False)
    
    try:
        return ExtractionResult(
            initial_strategy=strategy,
            initial_json=initial_json,
            refined_json=refined_json,
            gedankengang=raw_gedankengang
        )
    except ValidationError as e:
        raise PipelineError(f"Struktureller Validierungsfehler: Das generierte JSON konnte nicht in das finale Pydantic-Modell übertragen werden. Details: {e}")