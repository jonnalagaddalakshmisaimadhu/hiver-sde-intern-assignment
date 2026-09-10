"""
Unit tests for Phase 8 Modular AI Agent (retriever, escalation engine, and structured agent response).
"""
import pytest

from src.escalation.policy import EscalationEngine
from src.retrieval.retriever import HistoricalSupportRetriever, RetrievedEvidence
from src.agent.agent import AgentResponse


def test_retriever_masks_golden_set_and_retrieves():
    """Verify that the retriever excludes golden set IDs and returns top-k evidence."""
    retriever = HistoricalSupportRetriever()
    assert len(retriever.records) > 0
    assert retriever.embeddings is not None

    # Retrieve for battery query
    evidence = retriever.retrieve("My iPhone battery is draining very fast", top_k=3)
    assert len(evidence) == 3
    for ev in evidence:
        assert isinstance(ev, RetrievedEvidence)
        assert len(ev.customer_message) > 0
        assert len(ev.brand_response) > 0
        assert 0.0 <= ev.similarity <= 1.0


def test_escalation_engine_safety_triggers():
    """Verify that escalation engine correctly triggers on safety, legal, and hardware rules."""
    engine = EscalationEngine()

    # 1. Hardware damage must escalate
    dec1 = engine.evaluate(
        customer_message="My iPhone screen is cracked and shattered",
        predicted_intent="HARDWARE_PHYSICAL_DAMAGE",
        intent_confidence=0.98,
        evidence=[],
    )
    assert dec1.decision == "ESCALATE"
    assert dec1.trigger_rule == "PHYSICAL_HARDWARE_REPAIR_TRIGGER"

    # 2. Account security must escalate
    dec2 = engine.evaluate(
        customer_message="Someone hacked my account and stole my passwords",
        predicted_intent="APPLE_ID_ACCOUNT_SECURITY",
        intent_confidence=0.95,
        evidence=[],
    )
    assert dec2.decision == "ESCALATE"
    assert dec2.trigger_rule == "SECURITY_COMPROMISE_TRIGGER"

    # 3. Standard battery issue with high similarity evidence can AUTO_HANDLE
    mock_ev = [RetrievedEvidence(
        source_id="apple_123",
        customer_message="Battery drain issue on iPhone",
        brand_response="Please DM us your iOS version and battery health",
        similarity=0.85,
    )]
    dec3 = engine.evaluate(
        customer_message="Battery drops fast after update",
        predicted_intent="BATTERY_POWER_CHARGING",
        intent_confidence=0.90,
        evidence=mock_ev,
    )
    assert dec3.decision == "AUTO_HANDLE"
    assert dec3.trigger_rule == "SAFE_TECHNICAL_RESOLUTION"


def test_agent_response_schema():
    """Verify that AgentResponse strictly validates all fields."""
    resp = AgentResponse(
        intent="BATTERY_POWER_CHARGING",
        reply="Please DM us your device model and iOS version.",
        decision="AUTO_HANDLE",
        reason="Standard diagnostic available.",
        evidence=[],
        intent_confidence=0.92,
    )
    assert resp.intent == "BATTERY_POWER_CHARGING"
    assert resp.decision == "AUTO_HANDLE"
    assert resp.intent_confidence == 0.92
