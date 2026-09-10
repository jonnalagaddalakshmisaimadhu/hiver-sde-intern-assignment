"""
Unit tests for Phase 7 Baselines (Baseline 1: Majority Class, Baseline 2: TF-IDF + Logistic Regression).
"""
import csv
import json
from pathlib import Path
import pytest


def test_baseline_artifacts_exist():
    """Verify that metrics JSON and prediction CSV files exist for both baselines."""
    b1_metrics = Path("results/baseline1_metrics.json")
    b1_preds = Path("results/baseline1_predictions.csv")
    b2_metrics = Path("results/baseline2_metrics.json")
    b2_preds = Path("results/baseline2_predictions.csv")

    assert b1_metrics.is_file(), "Baseline 1 metrics JSON missing"
    assert b1_preds.is_file(), "Baseline 1 predictions CSV missing"
    assert b2_metrics.is_file(), "Baseline 2 metrics JSON missing"
    assert b2_preds.is_file(), "Baseline 2 predictions CSV missing"


def test_baseline_metrics_validity():
    """Verify metric calculations, sample sizes, and comparative behavior."""
    with open("results/baseline1_metrics.json", "r", encoding="utf-8") as f:
        m1 = json.load(f)
    with open("results/baseline2_metrics.json", "r", encoding="utf-8") as f:
        m2 = json.load(f)

    assert m1["total_evaluated"] == 200
    assert m2["total_evaluated"] == 200

    # Baseline 1 on 8 balanced classes has exactly 1/8 accuracy
    assert m1["accuracy"] == pytest.approx(0.125, rel=1e-2)
    assert m1["macro_f1"] < 0.05

    # Baseline 2 must outperform trivial baseline
    assert m2["accuracy"] > m1["accuracy"]
    assert m2["macro_f1"] > 0.60

    # Verify prediction counts in CSVs
    for pred_file in ["results/baseline1_predictions.csv", "results/baseline2_predictions.csv"]:
        with open(pred_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 200, f"Expected 200 rows in {pred_file}, got {len(rows)}"
