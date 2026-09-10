"""
Intent taxonomy loader and Pydantic validation schema for @AppleSupport.
"""
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


class IntentDefinition(BaseModel):
    id: str = Field(..., description="Unique machine-readable identifier for the intent")
    name: str = Field(..., description="Human-readable title of the intent")
    description: str = Field(..., description="Operational definition")
    inclusion_criteria: List[str] = Field(default_factory=list)
    exclusion_criteria: List[str] = Field(default_factory=list)
    typical_keywords: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)


class IntentTaxonomy(BaseModel):
    version: str
    brand: str
    intents: List[IntentDefinition]

    @property
    def intent_ids(self) -> List[str]:
        return [i.id for i in self.intents]

    def get_intent(self, intent_id: str) -> Optional[IntentDefinition]:
        for i in self.intents:
            if i.id == intent_id:
                return i
        return None


def load_intent_taxonomy(config_path: Path = Path("configs/intents.yaml")) -> IntentTaxonomy:
    """Load and validate the intent taxonomy YAML file."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Taxonomy config not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return IntentTaxonomy(**data)
