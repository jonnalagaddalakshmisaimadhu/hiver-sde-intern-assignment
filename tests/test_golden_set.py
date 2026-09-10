"""
Unit tests for Phase 6 Golden Evaluation Set integrity and audit.
"""
import json
from pathlib import Path
import pytest

from src.data.sample_golden_set import GoldenExample


def test_golden_set_files_exist():
    """Verify that JSONL, CSV, and labeling guide exist."""
    jsonl_path = Path("golden_set/golden_evaluation_set.jsonl")
    csv_path = Path("golden_set/golden_evaluation_set.csv")
    guide_path = Path("golden_set/labeling_guide.md")
    dist_path = Path("results/golden_set_distribution.json")

    assert jsonl_path.is_file(), "Golden set JSONL missing"
    assert csv_path.is_file(), "Golden set CSV missing"
    assert guide_path.is_file(), "Labeling guide missing"
    assert dist_path.is_file(), "Distribution summary missing"


def test_golden_set_size_and_schema():
    """Verify that the Golden Set contains exactly 200 validated records with 8 intents."""
    jsonl_path = Path("golden_set/golden_evaluation_set.jsonl")

    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            ex = GoldenExample(**data)
            examples.append(ex)

    assert len(examples) == 200, f"Expected 200 golden examples, got {len(examples)}"

    # Check intent diversity
    intents = {e.true_intent for e in examples}
    assert len(intents) == 8, f"Expected 8 intents, got {len(intents)}"

    # Verify each intent has 25 examples
    from collections import Counter
    counts = Counter(e.true_intent for e in examples)
    for intent, count in counts.items():
        assert count == 25, f"Intent {intent} has {count} examples (expected 25)"

    # Verify escalation distribution has both AUTO_HANDLE and ESCALATE
    escalations = {e.true_escalation_decision for e in examples}
    assert escalations == {"AUTO_HANDLE", "ESCALATE"}
