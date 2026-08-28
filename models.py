from pydantic import BaseModel, Field
from typing import List

class NEREntities(BaseModel):
    # Erzwingt Listen und setzt leere Listen als Standard, falls ein Key fehlt
    Krankheit: List[str] = Field(default_factory=list)
    Medikament: List[str] = Field(default_factory=list)
    
    # Verbietet rigoros zusätzliche/erfundene Schlüssel
    model_config = {
        "extra": "forbid"
    }

class ExtractionResult(BaseModel):
    initial_strategy: str
    initial_json: NEREntities
    refined_json: NEREntities
    gedankengang: str = "" # Wichtig für Chain-of-Thought