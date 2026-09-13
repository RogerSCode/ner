import os
import json
# NEU: Spezifische Exceptions aus der OpenAI-Bibliothek importieren
from openai import OpenAI, APIConnectionError, APITimeoutError, APIStatusError
from pydantic import ValidationError
from models import ExtractionResult

local_client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    timeout=120.0
)

class PipelineError(Exception):
    pass

# Die neuen, erweiterten Kategorien
ALLOWED_CATEGORIES = {
    "Krankheit", "Medikament", "Wirkstaerke", "Form", 
    "Dosierung", "Verabreichungsweg", "Haeufigkeit", "Dauer"
}

def generate_with_retry_and_filter(messages: list, model_name: str, expects_gedankengang: bool = False, max_retries: int = 3) -> dict:
    allowed_keys = set(ALLOWED_CATEGORIES)
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
            
            parsed_dict = json.loads(raw_content)
            errors = []
            
            extra_keys = set(parsed_dict.keys()) - allowed_keys
            if extra_keys:
                errors.append(f"Unzulässige Schlüssel generiert: {list(extra_keys)}. Erlaubt sind NUR: {list(allowed_keys)}.")
            
            for key in ALLOWED_CATEGORIES:
                if key in parsed_dict:
                    if not isinstance(parsed_dict[key], list):
                        errors.append(f"Der Schlüssel '{key}' muss eine Liste sein.")
                    else:
                        for item in parsed_dict[key]:
                            if not isinstance(item, str):
                                errors.append(f"Elemente in '{key}' müssen einfache Strings sein.")

            if errors and attempt < max_retries - 1:
                raise ValueError(" | ".join(errors))

            filtered_dict = {k: v for k, v in parsed_dict.items() if k in allowed_keys}
            
            for key in ALLOWED_CATEGORIES:
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
                    filtered_dict[key] = []
            
            return filtered_dict
            
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                error_msg = f"Fehler in deiner Ausgabe: {e} Bitte korrigiere dies und gib AUSSCHLIESSLICH das validierte JSON-Format zurück."
                messages.append({"role": "assistant", "content": raw_content if raw_content else "{}"})
                messages.append({"role": "user", "content": error_msg})
            else:
                raise PipelineError(f"Modell scheiterte nach {max_retries} Versuchen an Formatierungsfehlern. Letzter Fehler: {e}")
        
        # NEU: Spezifische Fehlerbehandlung auch in der Haupt-Generierung
        except APIConnectionError:
            raise PipelineError("Verbindungsfehler: Der lokale Ollama-Server ist nicht erreichbar. Bitte stelle sicher, dass Docker/Ollama läuft.")
        except APITimeoutError:
            raise PipelineError(f"Zeitüberschreitung: Das Modell '{model_name}' hat nach 120 Sekunden nicht geantwortet. Ist der Rechner überlastet?")
        except APIStatusError as e:
            raise PipelineError(f"Ollama API-Fehler (Code {e.status_code}): {e.message}")
        except Exception as e:
            raise PipelineError(f"Unerwarteter Systemfehler bei der Generierung: {str(e)}")

def orchestrator_agent(text: str, model_name: str) -> str:
    system_prompt = """Du bist der Orchestrator-Agent einer medizinischen NER-Pipeline. 
    Analysiere den Text und wähle die Prompting-Strategie:
    - 'Zero-Shot' für kurze Standardtexte.
    - 'Few-Shot' für Texte mit ungewöhnlichen Formaten/Abkürzungen.
    - 'Chain-of-Thought' für komplexe diagnostische Zusammenhänge.
    Antworte AUSSCHLIESSLICH mit exakt einem dieser drei Wörter."""
    
    try:
        response = local_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
            temperature=0.0
        )
        raw_output = response.choices[0].message.content.strip()
        if "Chain-of-Thought" in raw_output: return "Chain-of-Thought"
        if "Few-Shot" in raw_output: return "Few-Shot"
        return "Zero-Shot"
        
    # NEU: Gezielte Reaktion auf Netz- und API-Fehler im Orchestrator (ersetzt generisches Exception)
    except APIConnectionError:
        raise PipelineError("Orchestrator-Fehler: Keine Verbindung zu Ollama. Läuft der Dienst im Hintergrund?")
    except APITimeoutError:
        raise PipelineError(f"Orchestrator-Fehler: Timeout beim Zugriff auf Modell '{model_name}'.")
    except APIStatusError as e:
        raise PipelineError(f"Orchestrator-Fehler: API meldet Status {e.status_code}. Ist das Modell korrekt geladen?")
    except Exception as e:
        raise PipelineError(f"Unerwarteter Fehler im Orchestrator: {str(e)}")

def extractor_agent(text: str, strategy: str, model_name: str) -> dict:
    system_prompt = f"""Du bist ein medizinischer NER-Agent. Extrahiere Entitäten in folgende Kategorien: 
    {', '.join(ALLOWED_CATEGORIES)}. Antworte AUSSCHLIESSLICH im gültigen JSON-Format."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if strategy == "Few-Shot":
        messages.append({"role": "user", "content": "Patient nimmt morgens 1x 500mg Ibuprofen Tablette."})
        messages.append({"role": "assistant", "content": '{"Krankheit": [], "Medikament": ["Ibuprofen"], "Wirkstaerke": ["500mg"], "Form": ["Tablette"], "Dosierung": ["1x"], "Verabreichungsweg": [], "Haeufigkeit": ["morgens"], "Dauer": []}'})
    elif strategy == "Chain-of-Thought":
        system_prompt += " Denke Schritt für Schritt. Schreibe deine Analyse zuerst in ein Feld 'gedankengang' (als Text), bevor du die Arrays befüllst."
        messages[0]["content"] = system_prompt
        
    messages.append({"role": "user", "content": text})
    return generate_with_retry_and_filter(messages=messages, model_name=model_name, expects_gedankengang=(strategy == "Chain-of-Thought"))

def critic_agent(text: str, initial_json: dict, model_name: str) -> dict:
    system_prompt = f"""Du bist ein strenger medizinischer Qualitäts-Agent. 
    Du erhältst einen Originaltext und ein extrahiertes JSON.
    Prüfe: Wurden medizinische Entitäten übersehen oder falsch zugeordnet?
    REGELN:
    1. Erfinde NIEMALS neue JSON-Schlüssel! 
    2. Nutze AUSSCHLIESSLICH diese Kategorien: {', '.join(ALLOWED_CATEGORIES)}.
    Gib AUSSCHLIESSLICH das korrigierte JSON-Objekt zurück."""
    
    clean_json_for_prompt = {k: v for k, v in initial_json.items() if k in ALLOWED_CATEGORIES}
    user_prompt = f"Originaltext: {text}\n\nZu prüfendes JSON: {json.dumps(clean_json_for_prompt, ensure_ascii=False)}"
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return generate_with_retry_and_filter(messages=messages, model_name=model_name, expects_gedankengang=False)

def run_agentic_pipeline(text: str, model_name: str = "llama3", forced_strategy: str = "Auto") -> ExtractionResult:
    # 1. Orchestrator-Phase 
    if forced_strategy and forced_strategy != "Auto":
        strategy = forced_strategy
    else:
        strategy = orchestrator_agent(text, model_name)
        
    # 2. Extractor-Phase
    initial_json = extractor_agent(text, strategy, model_name)
    
    # 3. Critic-Phase
    if forced_strategy == "Zero-Shot":
        refined_json = initial_json.copy()
    else:
        refined_json = critic_agent(text, initial_json, model_name)
    
    # 4. Aufräumen für die GUI
    raw_gedankengang = initial_json.pop("gedankengang", "Kein Gedankengang")
    refined_json.pop("gedankengang", None)
    
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
        raise PipelineError(f"Struktureller Validierungsfehler: {e}")