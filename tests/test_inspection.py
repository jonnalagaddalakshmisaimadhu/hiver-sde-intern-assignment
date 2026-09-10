"""
Tests for Phase 2 data inspection pipeline.
"""
import json
from pathlib import Path
import pytest


def test_inspection_report_exists_and_valid():
    """Verify that results/dataset_inspection_report.json exists and contains correct schema."""
    report_path = Path("results/dataset_inspection_report.json")
    assert report_path.is_file(), "Inspection report not found"
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Validate top-level keys
    required_keys = [
        "file_metadata",
        "data_quality",
        "tweet_types",
        "thread_linkage",
        "authors",
        "text_statistics",
        "timestamps",
    ]
    for key in required_keys:
        assert key in report, f"Missing key in inspection report: {key}"

    # Verify empirical row count matches twcs dataset (~2.8M rows)
    assert report["file_metadata"]["total_rows"] == 2811774
    assert report["file_metadata"]["total_columns"] == 7
    
    # Verify exact column names
    expected_columns = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    assert report["file_metadata"]["columns"] == expected_columns

    # Verify brand count and top brand
    assert report["authors"]["brand_handles_count"] == 108
    assert report["authors"]["top_25_brands_by_tweet_volume"][0]["brand"] == "AmazonHelp"
