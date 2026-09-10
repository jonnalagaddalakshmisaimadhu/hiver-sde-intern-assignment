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
    import sys
    from test_agent_live import format_agent_response, run_interactive

    agent = AppleSupportAgent()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        t0 = time.time()
        resp = agent.process_inquiry(query)
        format_agent_response(query, resp, (time.time() - t0) * 1000)
    else:
        run_interactive(agent)
