"""
Unit tests for Phase 10: LLM-as-a-Judge and Human Agreement Validation.
"""
from pathlib import Path
import pytest
import numpy as np

from evaluation.llm_judge import LLMJudge, load_rubric, JudgeDimensionScore
from evaluation.judge_agreement import compute_dimension_agreement


def test_rubric_loading():
    """Verify judge rubric loads and contains all 6 required dimensions."""
    rubric = load_rubric(Path("configs/judge_rubric.yaml"))
    assert "dimensions" in rubric
    dims = rubric["dimensions"]
    expected_dims = [
        "relevance",
        "groundedness",
        "helpfulness",
        "factual_integrity",
        "brand_consistency",
        "tone",
    ]
    for d in expected_dims:
        assert d in dims, f"Missing dimension: {d}"
        assert "description" in dims[d]
        assert "scale" in dims[d]
        assert len(dims[d]["scale"]) == 5


def test_judge_evaluation_schema():
    """Verify judge evaluation output conforms to Pydantic schema and Likert bounds."""
    judge = LLMJudge(rubric_path=Path("configs/judge_rubric.yaml"))
    result = judge.evaluate_reply(
        customer_inquiry="My iPhone 8 battery drains in 2 hours after updating to iOS 11.",
        drafted_reply="We want to help look into your battery performance. Could you DM us your iOS version and battery health percentage?",
        retrieved_evidence=[{
            "customer_message": "iPhone 8 battery draining fast",
            "brand_response": "We'd like to help. Send us a DM with your battery stats.",
            "similarity": 0.85
        }],
        intent="BATTERY_POWER_CHARGING",
    )
    assert isinstance(result, JudgeDimensionScore)
    for dim in ["relevance", "groundedness", "helpfulness", "factual_integrity", "brand_consistency", "tone"]:
        val = getattr(result, dim)
        assert 1 <= val <= 5, f"Score for {dim} out of bounds: {val}"
    assert len(result.critique) > 10


def test_agreement_metrics_calculation():
    """Verify exact, adjacent, and correlation agreement math."""
    h_scores = [3, 4, 5, 2, 4]
    j_scores = [3, 4, 4, 2, 5]

    res = compute_dimension_agreement(h_scores, j_scores)
    assert res["sample_size"] == 5
    assert res["exact_agreement"] == 0.6  # 3 of 5 match exactly (indices 0, 1, 3)
    assert res["adjacent_agreement"] == 1.0  # all diffs are <= 1
    assert -1.0 <= res["pearson_r"] <= 1.0
    assert -1.0 <= res["cohen_kappa_quadratic"] <= 1.0
    assert abs(res["judge_bias"]) < 1.0
