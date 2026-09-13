from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Validiert sofort, ob die URL ein gültiges Format hat
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1", 
        description="Die Basis-URL für die lokale Ollama-Instanz"
    )
    
    default_model: str = Field(
        default="llama3", 
        description="Das Standard-LLM, das beim Start ausgewählt wird"
    )
    
    api_timeout: float = Field(
        default=120.0, 
        description="Timeout für Modell-Antworten in Sekunden"
    )

    # Liest automatisch aus einer .env Datei, falls vorhanden
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignoriert unwichtige ENV-Variablen
    )

# Globale Instanz, die einmalig beim Import validiert wird
settings = Settings()