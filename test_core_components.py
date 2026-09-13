"""
Umfassende Unit-Tests für die Kernkomponenten des NER-Systems.
Führe die Tests im Terminal mit dem Befehl aus: pytest test_core_components.py
"""

import pytest
import json
from pydantic import ValidationError
from unittest.mock import MagicMock, patch
from openai import APIConnectionError

# Importiere die zu testenden Module
from config import Settings
from models import NEREntities, ExtractionResult
from agent_pipeline import (
    orchestrator_agent, 
    generate_with_retry_and_filter, 
    PipelineError,
    ERLAUBTE_KATEGORIEN
)

# =====================================================================
# 1. Tests für config.py
# =====================================================================
def test_config_defaults():
    """Testet, ob die Konfiguration sichere Standardwerte lädt."""
    test_settings = Settings()
    assert test_settings.default_model == "llama3"
    assert test_settings.api_timeout == 120.0
    assert "localhost" in test_settings.ollama_base_url

# =====================================================================
# 2. Tests für models.py (Pydantic Validierung)
# =====================================================================
def test_ner_entities_valid():
    """Testet, ob valide Datenstrukturen korrekt instanziiert werden."""
    entitaeten = NEREntities(Medikament=["Ibuprofen"], Dosierung=["500mg"])
    assert entitaeten.Medikament == ["Ibuprofen"]
    assert entitaeten.Dosierung == ["500mg"]
    assert entitaeten.Krankheit == [] # Test auf Default-Factory

def test_ner_entities_forbidden_extra():
    """Testet, ob unerlaubte Kategorien von Pydantic strikt abgelehnt werden."""
    with pytest.raises(ValidationError) as excinfo:
        # 'Symptom' ist keine GERNERMED-Kategorie im Modell
        NEREntities(Symptom=["Kopfschmerz"])
    assert "Extra inputs are not permitted" in str(excinfo.value)

def test_extraction_result_valid():
    """Testet die korrekte Erstellung des finalen Extraktionsergebnisses."""
    initial = NEREntities(Medikament=["Aspirin"])
    refined = NEREntities(Medikament=["Aspirin"], Dosierung=["100mg"])
    
    ergebnis = ExtractionResult(
        initial_strategy="Zero-Shot",
        initial_json=initial,
        refined_json=refined,
        gedankengang="Ich habe die Dosierung ergänzt."
    )
    assert ergebnis.initial_strategy == "Zero-Shot"
    assert ergebnis.refined_json.Dosierung == ["100mg"]

# =====================================================================
# 3. Tests für agent_pipeline.py (inkl. Mocking)
# =====================================================================

@patch('agent_pipeline.local_client.chat.completions.create')
def test_orchestrator_agent_success(mock_create):
    """Testet, ob der Orchestrator die richtige Strategie aus der Antwort liest."""
    # Simuliere eine gültige LLM-Antwort
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Nach der Analyse empfehle ich: Chain-of-Thought."
    mock_create.return_value = mock_response

    strategie = orchestrator_agent("Komplexer Text...", "llama3")
    assert strategie == "Chain-of-Thought"

@patch('agent_pipeline.local_client.chat.completions.create')
def test_orchestrator_agent_fallback(mock_create):
    """Testet den Fallback auf 'Zero-Shot', falls das Modell Unsinn redet."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Ich bin mir nicht sicher, was ich tun soll."
    mock_create.return_value = mock_response

    strategie = orchestrator_agent("Text...", "llama3")
    assert strategie == "Zero-Shot"

@patch('agent_pipeline.local_client.chat.completions.create')
def test_generate_with_retry_and_filter_valid(mock_create):
    """Testet eine sofort erfolgreiche JSON-Extraktion ohne Filterbedarf."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"Medikament": ["Ibuprofen"]}'
    mock_create.return_value = mock_response

    ergebnis = generate_with_retry_and_filter([], "llama3")
    assert "Medikament" in ergebnis
    assert ergebnis["Medikament"] == ["Ibuprofen"]
    assert mock_create.call_count == 1

@patch('agent_pipeline.local_client.chat.completions.create')
def test_generate_with_retry_and_filter_fixes_string_to_list(mock_create):
    """Testet, ob die Filter-Logik Strings automatisch in Listen umwandelt."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"Medikament": "Ibuprofen"}' # Falscher Datentyp (String statt Liste)
    mock_create.return_value = mock_response

    ergebnis = generate_with_retry_and_filter([], "llama3")
    # Die Funktion sollte den String in eine Liste packen
    assert ergebnis["Medikament"] == ["Ibuprofen"]

@patch('agent_pipeline.local_client.chat.completions.create')
def test_generate_with_retry_and_filter_retries_on_invalid_json(mock_create):
    """Testet, ob der Retry-Mechanismus bei defektem JSON greift."""
    # 1. Versuch: Fehlerhaftes JSON (Klammer fehlt)
    mock_response_fail = MagicMock()
    mock_response_fail.choices[0].message.content = '{"Medikament": ["Ibuprofen"'
    
    # 2. Versuch: Korrektes JSON
    mock_response_success = MagicMock()
    mock_response_success.choices[0].message.content = '{"Medikament": ["Ibuprofen"]}'
    
    mock_create.side_effect = [json.JSONDecodeError("Fehler", "", 0), mock_response_success]

    ergebnis = generate_with_retry_and_filter([], "llama3")
    assert ergebnis["Medikament"] == ["Ibuprofen"]
    # Der Client muss exakt zweimal aufgerufen worden sein
    assert mock_create.call_count == 2

@patch('agent_pipeline.local_client.chat.completions.create')
def test_generate_with_retry_and_filter_removes_extra_keys(mock_create):
    """Testet, ob erfundene Kategorien (Halluzinationen) des LLMs sauber gefiltert werden."""
    mock_response = MagicMock()
    # 'ErfundeneKategorie' ist nicht in ERLAUBTE_KATEGORIEN
    mock_response.choices[0].message.content = '{"Medikament": ["Ibuprofen"], "ErfundeneKategorie": ["Test"]}'
    mock_create.return_value = mock_response

    ergebnis = generate_with_retry_and_filter([], "llama3")
    assert "Medikament" in ergebnis
    assert "ErfundeneKategorie" not in ergebnis

@patch('agent_pipeline.local_client.chat.completions.create')
def test_api_connection_error_raises_pipeline_error(mock_create):
    """Testet, ob Verbindungsabbrüche in saubere PipelineErrors übersetzt werden."""
    # Simuliere einen abgestürzten Ollama-Docker-Container
    mock_create.side_effect = APIConnectionError(request=MagicMock())
    
    with pytest.raises(PipelineError) as excinfo:
        orchestrator_agent("Text", "llama3")
    
    assert "Verbindungsfehler" in str(excinfo.value)