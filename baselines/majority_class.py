"""
Baseline 1: Trivial Majority-Class Intent Predictor.
Identifies the most frequent intent in the training split (strictly excluding Golden Set conversations),
predicts that majority intent for all evaluation examples, and outputs standard evaluation metrics.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

import pandas as pd

from evaluation.metrics import compute_intent_metrics
from src.data.sample_golden_set import classify_inquiry_heuristics


def run_majority_baseline(
    train_corpus_path: Path = Path("data/sample/applesupport_sample.jsonl"),
    golden_set_path: Path = Path("golden_set/golden_evaluation_set.jsonl"),
    output_metrics_path: Path = Path("results/baseline1_metrics.json"),
    output_predictions_path: Path = Path("results/baseline1_predictions.csv"),
) -> Dict[str, Any]:
    """Train and evaluate Baseline 1 (Majority Class)."""
    train_corpus_path = Path(train_corpus_path)
    golden_set_path = Path(golden_set_path)
    output_metrics_path = Path(output_metrics_path)
    output_predictions_path = Path(output_predictions_path)

    # 1. Load Golden Set to establish evaluation targets and anti-contamination mask
    print(f"[INFO] Loading Golden Evaluation Set from: {golden_set_path}")
    golden_records = []
    golden_conv_ids: Set[str] = set()

    with open(golden_set_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            golden_records.append(rec)
            golden_conv_ids.add(rec["conversation_id"])

    print(f"[INFO] Loaded {len(golden_records)} golden examples ({len(golden_conv_ids)} masked IDs).")

    # 2. Ingest Training Corpus with strict anti-contamination exclusion
    print(f"[INFO] Ingesting training corpus from {train_corpus_path} (excluding masked IDs)...")
    train_intent_counts = Counter()
    total_train = 0
    excluded_count = 0

    with open(train_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["conversation_id"] in golden_conv_ids:
                excluded_count += 1
                continue
            intent = classify_inquiry_heuristics(rec["customer_inquiry"])
            train_intent_counts[intent] += 1
            total_train += 1

    print(f"[INFO] Training corpus size: {total_train:,} (Excluded {excluded_count} Golden Set instances).")
    majority_intent, majority_count = train_intent_counts.most_common(1)[0]
    majority_pct = round((majority_count / total_train) * 100, 2)
    print(f"[INFO] Determined Majority Intent: '{majority_intent}' ({majority_count:,}/{total_train:,} = {majority_pct}%)")

    # 3. Predict majority intent on all Golden Set examples
    y_true = [rec["true_intent"] for rec in golden_records]
    y_pred = [majority_intent] * len(golden_records)

    # 4. Compute metrics
    labels = sorted(list(set(y_true)))
    metrics = compute_intent_metrics(y_true, y_pred, labels=labels)
    metrics["model_name"] = "Baseline 1: Majority Class"
    metrics["majority_intent"] = majority_intent
    metrics["training_distribution"] = dict(train_intent_counts)

    # 5. Save outputs
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Write predictions CSV
    with open(output_predictions_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "customer_inquiry", "true_intent", "predicted_intent"])
        writer.writeheader()
        for rec, pred in zip(golden_records, y_pred):
            writer.writerow({
                "example_id": rec["example_id"],
                "customer_inquiry": rec["customer_inquiry"],
                "true_intent": rec["true_intent"],
                "predicted_intent": pred,
            })

    print(f"[SUCCESS] Baseline 1 metrics saved to {output_metrics_path}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline 1 (Majority Class).")
    parser.add_argument("--train-corpus", type=str, default="data/sample/applesupport_sample.jsonl")
    parser.add_argument("--golden-set", type=str, default="golden_set/golden_evaluation_set.jsonl")
    parser.add_argument("--output-metrics", type=str, default="results/baseline1_metrics.json")
    parser.add_argument("--output-preds", type=str, default="results/baseline1_predictions.csv")
    args = parser.parse_args()

    run_majority_baseline(
        train_corpus_path=Path(args.train_corpus),
        golden_set_path=Path(args.golden_set),
        output_metrics_path=Path(args.output_metrics),
        output_predictions_path=Path(args.output_preds),
    )
