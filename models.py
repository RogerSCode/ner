from pydantic import BaseModel, Field
from typing import List

class NEREntities(BaseModel):
    # Erweitert um alle GERNERMED-Kategorien + Krankheit
    Krankheit: List[str] = Field(default_factory=list)
    Medikament: List[str] = Field(default_factory=list) # in GERNERMED: Drug
    Wirkstaerke: List[str] = Field(default_factory=list) # in GERNERMED: Strength
    Form: List[str] = Field(default_factory=list) # in GERNERMED: Form (z.B. Tablette)
    Dosierung: List[str] = Field(default_factory=list) # in GERNERMED: Dosage
    Verabreichungsweg: List[str] = Field(default_factory=list) # in GERNERMED: Route (z.B. i.v.)
    Haeufigkeit: List[str] = Field(default_factory=list) # in GERNERMED: Frequency
    Dauer: List[str] = Field(default_factory=list) # in GERNERMED: Duration
    
    # Verbietet weiterhin rigoros erfundene Schlüssel
    model_config = {
        "extra": "forbid"
    }

class ExtractionResult(BaseModel):
    initial_strategy: str
    initial_json: NEREntities
    refined_json: NEREntities
    gedankengang: str = ""# Wichtig für Chain-of-Thought