import os
import json
from openai import OpenAI
from models import ExtractionResult

# Der Client erhält jetzt ein striktes Timeout (120 Sekunden), um Hänger zu vermeiden.
local_client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
    timeout=120.0
)

class PipelineError(Exception):
    """Eine benutzerdefinierte Ausnahme für alle Fehler innerhalb der Agenten-Pipeline."""
    pass

def orchestrator_agent(text: str, model_name: str) -> str:
    """Agent 1: Analysiert Textkomplexität und empfiehlt autonom eine Prompt-Strategie."""
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
        
        # Sicherheits-Parsing
        if "Chain-of-Thought" in raw_output: return "Chain-of-Thought"
        if "Few-Shot" in raw_output: return "Few-Shot"
        return "Zero-Shot"
    
    except Exception as e:
        # Hier schlucken wir den Fehler nicht mehr, sondern leiten ihn weiter
        raise PipelineError(f"Der Orchestrator-Agent konnte nicht antworten. API-Fehler: {str(e)}")

def extractor_agent(text: str, strategy: str, model_name: str) -> dict:
    """Agent 2: Führt die eigentliche Extraktion basierend auf der Strategie aus."""
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
    
    try:
        response = local_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        raise PipelineError(f"Der Extractor-Agent ist während der Generierung fehlgeschlagen: {str(e)}")
    
    # Schutz vor nicht-valider JSON-Ausgabe des LLMs
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        snippet = raw_content[:150] + "..." if raw_content else "Leere Antwort"
        raise PipelineError(f"Der Extractor-Agent hat ungültiges JSON geliefert. (Generierter Start: {snippet}). Parser-Fehler: {str(e)}")

def critic_agent(text: str, initial_json: dict, model_name: str) -> dict:
    """Agent 3: Self-Refinement. Prüft das Ergebnis und korrigiert Fehler."""
    system_prompt = """Du bist ein strenger medizinischer Qualitäts-Agent. 
    Du erhältst einen Originaltext und ein extrahiertes JSON.
    Prüfe: Wurden medizinische Entitäten übersehen?
    
    REGELN:
    1. Erfinde NIEMALS neue JSON-Schlüssel (Keys)! 
    2. Nutze AUSSCHLIESSLICH die Kategorien 'Krankheit' und 'Medikament'.
    3. Ignoriere Alter, Geschlecht oder Dosierungen.
    
    Gib AUSSCHLIESSLICH das korrigierte JSON-Objekt zurück."""
    
    # Sichere das initial_json wieder in einen String für den Prompt
    user_prompt = f"Originaltext: {text}\n\nZu prüfendes JSON: {json.dumps(initial_json, ensure_ascii=False)}"
    
    try:
        response = local_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content
    except Exception as e:
        raise PipelineError(f"Der Critic-Agent ist während der Generierung fehlgeschlagen: {str(e)}")
    
    # Schutz vor nicht-valider JSON-Ausgabe des LLMs
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        snippet = raw_content[:150] + "..." if raw_content else "Leere Antwort"
        raise PipelineError(f"Der Critic-Agent hat ungültiges JSON geliefert. (Generierter Start: {snippet}). Parser-Fehler: {str(e)}")

def run_agentic_pipeline(text: str, model_name: str = "llama3") -> ExtractionResult:
    """Orchestriert den gesamten Workflow der drei Agenten."""
    strategy = orchestrator_agent(text, model_name)
    initial_json = extractor_agent(text, strategy, model_name)
    refined_json = critic_agent(text, initial_json, model_name)
    
    # Sicherstellen, dass "gedankengang" immer ein String ist
    raw_gedankengang = initial_json.get("gedankengang", "Kein Gedankengang (da kein Chain-of-Thought)")
    if not isinstance(raw_gedankengang, str):
        raw_gedankengang = json.dumps(raw_gedankengang, ensure_ascii=False)
    
    return ExtractionResult(
        initial_strategy=strategy,
        initial_json=initial_json,
        refined_json=refined_json,
        gedankengang=raw_gedankengang
    )