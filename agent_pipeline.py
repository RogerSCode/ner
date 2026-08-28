import os
import json
from openai import OpenAI
from models import ExtractionResult

# Der Client spricht mit dem lokalen Ollama-Docker-Container
local_client = OpenAI(
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama" 
)

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
        print(f"Orchestrator Fehler: {e}")
        return "Zero-Shot"

def extractor_agent(text: str, strategy: str, model_name: str) -> dict:
    """Agent 2: Führt die eigentliche Extraktion basierend auf der Strategie aus."""
    system_prompt = """Du bist ein medizinischer NER-Agent. Extrahiere Entitäten in die Kategorien 
    'Krankheit' und 'Medikament'. Antworte AUSSCHLIESSLICH im gültigen JSON-Format."""
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if strategy == "Few-Shot":
        messages.append({"role": "user", "content": "Patient hat Migräne, nimmt Ibuprofen."})
        messages.append({"role": "assistant", "content": '{"Krankheit": ["Migräne"], "Medikament": ["Ibuprofen"]}'})
    elif strategy == "Chain-of-Thought":
        # Erfordert vom Modell, seine Schritte zu erklären. Wir fordern explizit einen Text-String.
        system_prompt += " Denke Schritt für Schritt. Schreibe deine Analyse zuerst in ein Feld 'gedankengang' (als einfachen Text-String), bevor du die Arrays für 'Krankheit' und 'Medikament' befüllst."
        messages[0]["content"] = system_prompt
        
    messages.append({"role": "user", "content": text})
    
    response = local_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

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
    
    user_prompt = f"Originaltext: {text}\n\nZu prüfendes JSON: {json.dumps(initial_json)}"
    
    response = local_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def run_agentic_pipeline(text: str, model_name: str = "llama3") -> ExtractionResult:
    """Orchestriert den gesamten Workflow der drei Agenten."""
    strategy = orchestrator_agent(text, model_name)
    initial_json = extractor_agent(text, strategy, model_name)
    refined_json = critic_agent(text, initial_json, model_name)
    
    # 1. Den Wert aus dem JSON extrahieren
    raw_gedankengang = initial_json.get("gedankengang", "Kein Gedankengang (da kein Chain-of-Thought)")
    
    # 2. Falls das LLM eine Liste oder ein Dictionary statt eines Strings generiert hat, in String umwandeln
    if not isinstance(raw_gedankengang, str):
        raw_gedankengang = json.dumps(raw_gedankengang, ensure_ascii=False)
    
    return ExtractionResult(
        initial_strategy=strategy,
        initial_json=initial_json,
        refined_json=refined_json,
        gedankengang=raw_gedankengang
    )