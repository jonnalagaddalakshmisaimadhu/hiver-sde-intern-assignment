"""
Phase 8: Evidence-Grounded Reply Generator for @AppleSupport.
Drafts customer support replies strictly grounded in retrieved historical evidence,
enforcing anti-hallucination guardrails and authentic Apple Support tone.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.retrieval.retriever import RetrievedEvidence
from src.utils.llm_client import LLMClient


class GeneratedReply(BaseModel):
    reply: str = Field(..., description="Drafted customer-facing reply in authentic Apple Support persona")
    grounded_in_evidence: bool = Field(True, description="Whether the response is fully supported by retrieved cases")
    requires_clarification: bool = Field(False, description="Whether additional customer diagnostic info is needed")


class ReplyGenerator:
    """Generates brand-consistent, evidence-grounded customer support replies."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def generate_reply(
        self,
        customer_message: str,
        predicted_intent: str,
        evidence: List[RetrievedEvidence],
        conversation_context: Optional[str] = None,
    ) -> GeneratedReply:
        """Draft a grounded customer support reply."""
        # Format evidence snippets
        evidence_text_blocks = []
        for i, ev in enumerate(evidence, 1):
            evidence_text_blocks.append(
                f"[Evidence {i}] (Similarity: {ev.similarity:.2f})\n"
                f"Historical Customer: \"{ev.customer_message}\"\n"
                f"Historical Apple Resolution: \"{ev.brand_response}\""
            )
        evidence_str = "\n\n".join(evidence_text_blocks) if evidence_text_blocks else "No relevant historical evidence found."

        system_instruction = (
            "You are an official customer support agent for Apple Support (@AppleSupport) on social media.\n\n"
            "STRICT OPERATIONAL GUIDELINES:\n"
            "1. Ground your answer STRICTLY in how Apple Support historically resolved similar issues shown in the retrieved evidence.\n"
            "2. DO NOT invent unsupported policies, repair guarantees, or free replacements.\n"
            "3. DO NOT claim actions were already performed on their device.\n"
            "4. Keep the reply friendly, professional, empathetic, and concise (under 280 characters if possible, standard Twitter length).\n"
            "5. If the issue is physical damage, guide them to schedule a Genius Bar / Apple Authorized Service Provider appointment.\n"
            "6. If the customer query lacks details, ask for device model and current iOS/macOS version.\n"
            "7. Never output internal reasoning or mention 'evidence' or 'prompts' in the reply.\n\n"
            "Output a valid JSON object with the following schema:\n"
            '{\n  "reply": "<your_customer_facing_tweet>",\n  "grounded_in_evidence": <true_or_false>,\n  "requires_clarification": <true_or_false>\n}'
        )

        user_prompt = (
            f"Incoming Customer Message:\n\"{customer_message}\"\n\n"
            f"Classified Intent: {predicted_intent}\n\n"
            f"Retrieved Historical Support Resolutions:\n{evidence_str}"
        )

        try:
            data = self.llm.generate_json(prompt=user_prompt, system_instruction=system_instruction)
            return GeneratedReply(**data)
        except Exception as e:
            print(f"[WARN] Reply generation failed: {e}. Using fallback evidence resolution.")
            if evidence:
                fallback_text = evidence[0].brand_response
            else:
                fallback_text = "We're here to help! Could you let us know which device model and iOS version you're using?"
            return GeneratedReply(
                reply=fallback_text,
                grounded_in_evidence=bool(evidence),
                requires_clarification=True,
            )
