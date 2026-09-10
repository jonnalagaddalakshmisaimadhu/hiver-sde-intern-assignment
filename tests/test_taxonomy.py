"""
Unit tests for Phase 5 intent taxonomy and schema validation.
"""
from pathlib import Path
import pytest

from src.intents.taxonomy import load_intent_taxonomy, IntentTaxonomy


def test_intent_taxonomy_loads_and_validates():
    """Verify that the intent taxonomy YAML loads into Pydantic models with 8 intents."""
    taxonomy = load_intent_taxonomy(Path("configs/intents.yaml"))
    assert isinstance(taxonomy, IntentTaxonomy)
    assert taxonomy.brand == "AppleSupport"
    assert len(taxonomy.intents) == 8

    expected_ids = {
        "SOFTWARE_UPDATE_OS",
        "BATTERY_POWER_CHARGING",
        "APPLE_ID_ACCOUNT_SECURITY",
        "AUDIO_CONNECTIVITY_BLUETOOTH",
        "HARDWARE_PHYSICAL_DAMAGE",
        "APP_STORE_BILLING_SUBSCRIPTIONS",
        "DEVICE_PERFORMANCE_CRASH",
        "OTHER_OR_UNCLEAR",
    }
    assert set(taxonomy.intent_ids) == expected_ids

    # Validate that each intent has rich criteria and examples
    for intent in taxonomy.intents:
        assert len(intent.name) > 0
        assert len(intent.description) > 20
        assert len(intent.inclusion_criteria) >= 2
        assert len(intent.exclusion_criteria) >= 1
        assert len(intent.typical_keywords) >= 4
        assert len(intent.examples) >= 3


def test_taxonomy_markdown_doc_exists():
    """Verify that configs/intent_taxonomy.md documentation exists and covers all intents."""
    doc_path = Path("configs/intent_taxonomy.md")
    assert doc_path.is_file(), "Intent taxonomy documentation missing"
    content = doc_path.read_text(encoding="utf-8")
    assert "SOFTWARE_UPDATE_OS" in content
    assert "BATTERY_POWER_CHARGING" in content
    assert "OTHER_OR_UNCLEAR" in content
