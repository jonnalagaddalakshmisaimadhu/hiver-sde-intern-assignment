"""
Phase 8: Escalation Policy Engine for @AppleSupport.
Determines whether an inquiry can be safely AUTO_HANDLE or must ESCALATE to a human specialist,
providing an explicit, audit-ready rationale.
"""
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.retrieval.retriever import RetrievedEvidence


class EscalationDecision(BaseModel):
    decision: str = Field(..., description="AUTO_HANDLE or ESCALATE")
    reason: str = Field(..., description="Explicit, human-auditable justification for the decision")
    trigger_rule: Optional[str] = Field(None, description="Specific safety rule or trigger that fired")


# Safety triggers
LEGAL_KEYWORDS = [r"\blawyer\b", r"\bpolice\b", r"\bcourt\b", r"\bsue\b", r"\blawsuit\b", r"\blegal action\b"]
SECURITY_KEYWORDS = [r"\bhacked\b", r"\bstolen\b", r"\bunauthorized\b", r"\bfraud\b", r"\bscam\b", r"\bidentity theft\b"]
DISPUTE_KEYWORDS = [r"\bdispute\b", r"\bchargeback\b", r"\bstole my money\b", r"\bunauthorized charge\b"]
ABUSIVE_KEYWORDS = [r"\bfuck\b", r"\bshit\b", r"\bbastards\b", r"\bassholes\b", r"\bscammers\b"]


class EscalationEngine:
    """Multi-tiered escalation decision engine minimizing False Auto-Handle Risks."""

    def __init__(self, min_similarity_threshold: float = 0.38):
        self.min_similarity_threshold = min_similarity_threshold

    def evaluate(
        self,
        customer_message: str,
        predicted_intent: str,
        intent_confidence: float,
        evidence: List[RetrievedEvidence],
    ) -> EscalationDecision:
        """Evaluate inquiry against deterministic safety policies and evidence sufficiency."""
        text_lower = customer_message.lower()

        # 1. Critical Safety Trigger: Legal Threat
        for pattern in LEGAL_KEYWORDS:
            if re.search(pattern, text_lower):
                return EscalationDecision(
                    decision="ESCALATE",
                    reason="Customer explicitly mentions legal action, police, or lawsuits; requires human management review.",
                    trigger_rule="LEGAL_THREAT_TRIGGER",
                )

        # 2. Critical Safety Trigger: Security & Account Compromise
        for pattern in SECURITY_KEYWORDS:
            if re.search(pattern, text_lower):
                return EscalationDecision(
                    decision="ESCALATE",
                    reason="Inquiry involves potential account hijacking, stolen device, or unauthorized credentials.",
                    trigger_rule="SECURITY_COMPROMISE_TRIGGER",
                )

        # 3. Critical Safety Trigger: Abusive / Extreme Emotion
        for pattern in ABUSIVE_KEYWORDS:
            if re.search(pattern, text_lower):
                return EscalationDecision(
                    decision="ESCALATE",
                    reason="Severe customer agitation or abusive language detected; human agent empathy required.",
                    trigger_rule="CUSTOMER_AGITATION_TRIGGER",
                )

        # 4. Domain Trigger: Physical Hardware Damage (AI cannot repair physical glass or hardware)
        if predicted_intent == "HARDWARE_PHYSICAL_DAMAGE":
            return EscalationDecision(
                decision="ESCALATE",
                reason="Physical hardware or liquid damage cannot be serviced digitally; requires in-person Genius Bar assessment.",
                trigger_rule="PHYSICAL_HARDWARE_REPAIR_TRIGGER",
            )

        # 5. Financial Dispute Trigger
        if predicted_intent == "APP_STORE_BILLING_SUBSCRIPTIONS":
            for pattern in DISPUTE_KEYWORDS:
                if re.search(pattern, text_lower):
                    return EscalationDecision(
                        decision="ESCALATE",
                        reason="Disputed monetary transactions or unauthorized billing require human financial verification.",
                        trigger_rule="BILLING_DISPUTE_TRIGGER",
                    )

        # 6. Intent Ambiguity / Low Confidence Trigger
        if predicted_intent == "OTHER_OR_UNCLEAR" and len(customer_message.split()) < 5:
            return EscalationDecision(
                decision="ESCALATE",
                reason="Inquiry is too ambiguous or brief to diagnose technical symptoms safely.",
                trigger_rule="LOW_CONFIDENCE_AMBIGUOUS_TRIGGER",
            )

        if intent_confidence < 0.40:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Intent classification confidence ({intent_confidence:.2f}) is below operational threshold (0.40).",
                trigger_rule="LOW_INTENT_CONFIDENCE_TRIGGER",
            )

        # 7. Insufficient Historical Evidence Trigger
        if not evidence or evidence[0].similarity < self.min_similarity_threshold:
            top_sim = evidence[0].similarity if evidence else 0.0
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Insufficient historical resolution grounding (Top similarity: {top_sim:.2f} < {self.min_similarity_threshold}).",
                trigger_rule="INSUFFICIENT_EVIDENCE_TRIGGER",
            )

        # 8. All checks passed: Safe to AUTO_HANDLE
        return EscalationDecision(
            decision="AUTO_HANDLE",
            reason=f"Standard diagnostic resolution available in historical support evidence (Similarity: {evidence[0].similarity:.2f}).",
            trigger_rule="SAFE_TECHNICAL_RESOLUTION",
        )
