"""
Unit tests for Phase 4 conversation reconstruction and data cleaning.
"""
import json
from pathlib import Path
import pytest


def test_reconstruction_summary_exists():
    """Verify that conversation_reconstruction_summary.json exists and has expected counts."""
    summary_path = Path("results/conversation_reconstruction_summary.json")
    assert summary_path.is_file(), "Reconstruction summary file missing"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["brand"] == "AppleSupport"
    assert summary["total_reconstructed_conversations"] > 100_000
    assert summary["sample_size"] == 5000


def test_sample_conversations_structure_and_types():
    """Verify that sample JSONL records adhere strictly to the conversational schema."""
    sample_path = Path("data/sample/applesupport_sample.jsonl")
    assert sample_path.is_file(), "Sample JSONL file missing"

    count = 0
    required_keys = [
        "conversation_id",
        "brand",
        "customer_id",
        "turn_count",
        "customer_inquiry",
        "brand_initial_response",
        "turns",
        "has_multi_turn_followup",
    ]

    with open(sample_path, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            record = json.loads(line)
            for k in required_keys:
                assert k in record, f"Record {count} missing required key: {k}"

            assert record["brand"] == "AppleSupport"
            assert len(record["customer_inquiry"]) > 0
            assert len(record["brand_initial_response"]) > 0
            assert len(record["turns"]) >= 2
            assert record["turns"][0]["speaker"] == "customer"
            assert record["turns"][1]["speaker"] == "brand"

    assert count == 5000, f"Expected 5,000 sampled conversations, got {count}"
