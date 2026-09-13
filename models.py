"""
Definition der Datenmodelle für den Datenaustausch zwischen den KI-Agenten
und der Benutzeroberfläche mittels Pydantic.
"""

from pydantic import BaseModel, Field
from typing import List

class NEREntities(BaseModel):
    """
    Pydantic-Modell für die strukturierten medizinischen Entitäten.
    
    Spiegelt die Kategorien des GERNERMED-Korpus wider und erzwingt eine strikte
    Typisierung (ausschließlich Listen von Zeichenketten) sowie den Ausschluss
    unerlaubter Felder durch das System.
    """
    Krankheit: List[str] = Field(default_factory=list)
    Medikament: List[str] = Field(default_factory=list)
    Wirkstaerke: List[str] = Field(default_factory=list)
    Form: List[str] = Field(default_factory=list)
    Dosierung: List[str] = Field(default_factory=list)
    Verabreichungsweg: List[str] = Field(default_factory=list)
    Haeufigkeit: List[str] = Field(default_factory=list)
    Dauer: List[str] = Field(default_factory=list)
    
    model_config = {
        "extra": "forbid"
    }

class ExtractionResult(BaseModel):
    """
    Pydantic-Modell für das Gesamtergebnis eines Pipeline-Durchlaufs.
    
    Kapselt die gewählte Strategie, den initialen Entwurf, das verfeinerte
    Ergebnis (Self-Refinement) sowie den optionalen Denkprozess der KI.
    """
    initial_strategy: str
    initial_json: NEREntities
    refined_json: NEREntities
    gedankengang: str = ""