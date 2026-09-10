"""
Phase 8: Intent Classifier for @AppleSupport.
Classifies incoming customer messages into the 8-intent taxonomy using structured LLM prompts,
strict Pydantic validation, and robust fallback handling.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from src.intents.taxonomy import load_intent_taxonomy
from src.utils.llm_client import LLMClient


ALLOWED_INTENTS = {
    "SOFTWARE_UPDATE_OS",
    "BATTERY_POWER_CHARGING",
    "APPLE_ID_ACCOUNT_SECURITY",
    "AUDIO_CONNECTIVITY_BLUETOOTH",
    "HARDWARE_PHYSICAL_DAMAGE",
    "APP_STORE_BILLING_SUBSCRIPTIONS",
    "DEVICE_PERFORMANCE_CRASH",
    "OTHER_OR_UNCLEAR",
}


class IntentClassificationResult(BaseModel):
    intent: str = Field(..., description="One of the 8 allowed intent identifiers")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    rationale: str = Field(..., description="Concise justification grounded in inquiry keywords")

    @field_validator("intent")
    @classmethod
    def validate_allowed_intent(cls, v: str) -> str:
        clean = v.strip().upper()
        if clean not in ALLOWED_INTENTS:
            # Coerce unrecognized intents to fallback
            return "OTHER_OR_UNCLEAR"
        return clean


class IntentClassifier:
    """Classifies customer inquiries into AppleSupport operational intents."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.taxonomy = load_intent_taxonomy()

        # Build prompt instructions once
        intents_desc = []
        for item in self.taxonomy.intents:
            intents_desc.append(
                f"- {item.id} ({item.name}): {item.description} Keywords: {', '.join(item.typical_keywords[:5])}"
            )
        self.taxonomy_prompt_text = "\n".join(intents_desc)

    def classify(self, customer_message: str) -> IntentClassificationResult:
        """Classify customer message using structured LLM call with validation."""
        system_instruction = (
            "You are an expert customer support intent classifier for Apple Support (@AppleSupport).\n"
            "Your job is to classify the customer's incoming message into EXACTLY ONE of the following 8 allowed intents:\n\n"
            f"{self.taxonomy_prompt_text}\n\n"
            "Return a strictly valid JSON object with the following schema:\n"
            '{\n  "intent": "<ONE_OF_THE_8_INTENT_IDS>",\n  "confidence": <float_between_0.0_and_1.0>,\n  "rationale": "<brief_reason>"\n}'
        )

        user_prompt = f'Customer Inquiry:\n"{customer_message}"'

        try:
            data = self.llm.generate_json(prompt=user_prompt, system_instruction=system_instruction)
            return IntentClassificationResult(**data)
        except Exception as e:
            print(f"[WARN] LLM intent classification failed: {e}. Falling back to heuristic classifier.")
            from src.data.sample_golden_set import classify_inquiry_heuristics
            fallback_intent = classify_inquiry_heuristics(customer_message)
            return IntentClassificationResult(
                intent=fallback_intent,
                confidence=0.5,
                rationale="Fallback heuristic classification due to API interruption",
            )
