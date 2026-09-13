"""
Zentrale Konfigurationsverwaltung für das NER-System.
Nutzt Pydantic-Settings für die typsichere Validierung von Umgebungsvariablen.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """
    Datenmodell für die Systemkonfiguration.
    Liest automatisch aus System-Umgebungsvariablen oder einer .env-Datei.
    """
    
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1", 
        description="Die Basis-URL für die lokale Ollama-Instanz"
    )
    
    default_model: str = Field(
        default="llama3", 
        description="Das Standard-Sprachmodell, das beim Start ausgewählt wird"
    )
    
    api_timeout: float = Field(
        default=120.0, 
        description="Timeout für Modell-Antworten in Sekunden"
    )

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Globale Instanz, die einmalig beim Import validiert wird
settings = Settings()