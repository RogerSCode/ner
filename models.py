from pydantic import BaseModel
from typing import Dict, List, Any

class ExtractionResult(BaseModel):
    initial_strategy: str
    initial_json: Dict[str, Any]
    refined_json: Dict[str, Any]
    gedankengang: str = "" # Wichtig für Chain-of-Thought