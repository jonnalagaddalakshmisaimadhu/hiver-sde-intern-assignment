"""
Phase 8: End-to-End Customer Support Agent for @AppleSupport.
Coordinates Intent Classification, Dense Semantic Retrieval, Evidence Grounding,
Reply Generation, and Escalation Decision making into a unified Pydantic contract.
"""
import sys
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.intents.classifier import IntentClassifier, IntentClassificationResult
from src.retrieval.retriever import HistoricalSupportRetriever, RetrievedEvidence
from src.generation.generator import ReplyGenerator, GeneratedReply
from src.escalation.policy import EscalationEngine, EscalationDecision
from src.utils.llm_client import LLMClient


class AgentResponse(BaseModel):
    intent: str = Field(..., description="Classified intent from the 8-intent taxonomy")
    reply: str = Field(..., description="Drafted customer-facing resolution")
    decision: str = Field(..., description="AUTO_HANDLE or ESCALATE")
    reason: str = Field(..., description="Explicit rationale for the decision")
    evidence: List[RetrievedEvidence] = Field(default_factory=list, description="Top-k retrieved historical support resolutions")
    intent_confidence: float = Field(..., ge=0.0, le=1.0)


class AppleSupportAgent:
    """Production-grade modular AI support agent for Apple Support."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[HistoricalSupportRetriever] = None,
        top_k: int = 3,
    ):
        self.llm_client = llm_client or LLMClient()
        self.classifier = IntentClassifier(llm_client=self.llm_client)
        self.retriever = retriever or HistoricalSupportRetriever()
        self.generator = ReplyGenerator(llm_client=self.llm_client)
        self.escalation_engine = EscalationEngine()
        self.top_k = top_k

    def process_inquiry(self, customer_message: str) -> AgentResponse:
        """Process incoming customer inquiry through the full pipeline."""
        # 1. Intent Classification
        clf_result = self.classifier.classify(customer_message)

        # 2. Historical Retrieval
        evidence = self.retriever.retrieve(customer_message, top_k=self.top_k)

        # 3. Escalation Decision Check
        esc_decision = self.escalation_engine.evaluate(
            customer_message=customer_message,
            predicted_intent=clf_result.intent,
            intent_confidence=clf_result.confidence,
            evidence=evidence,
        )

        # 4. Reply Generation (grounded in retrieved evidence)
        generated_reply = self.generator.generate_reply(
            customer_message=customer_message,
            predicted_intent=clf_result.intent,
            evidence=evidence,
        )

        # 5. Format Structured Contract
        return AgentResponse(
            intent=clf_result.intent,
            reply=generated_reply.reply,
            decision=esc_decision.decision,
            reason=esc_decision.reason,
            evidence=evidence,
            intent_confidence=clf_result.confidence,
        )


if __name__ == "__main__":
    import json
    agent = AppleSupportAgent()
    sample_queries = [
        "My iPhone X screen completely shattered when I dropped it on concrete. How do I get it fixed?",
        "Why is my battery dropping from 50% to 10% in half an hour since updating to iOS 11?",
        "Someone hacked my Apple ID and changed my recovery email! Please help!",
    ]

    for q in sample_queries:
        print("\n" + "=" * 60)
        print(f"Customer Inquiry: \"{q}\"")
        resp = agent.process_inquiry(q)
        print(f"  Predicted Intent: {resp.intent} (Confidence: {resp.intent_confidence:.2f})")
        print(f"  Decision: {resp.decision} - {resp.reason}")
        print(f"  Drafted Reply: \"{resp.reply}\"")
        print(f"  Evidence Retrieved: {len(resp.evidence)} cases (Top similarity: {resp.evidence[0].similarity if resp.evidence else 0.0})")
