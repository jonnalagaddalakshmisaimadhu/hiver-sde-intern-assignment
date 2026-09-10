"""
Baseline 2: Conventional ML Intent Classifier (TF-IDF + Logistic Regression).
Extracts sublinear n-gram TF-IDF representations from customer inquiries and trains
a balanced multinomial Logistic Regression classifier with fixed random seed (42).
Evaluated on the exact 200-example Golden Evaluation Set with zero contamination.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from evaluation.metrics import compute_intent_metrics
from src.data.sample_golden_set import classify_inquiry_heuristics


def run_tfidf_baseline(
    train_corpus_path: Path = Path("data/sample/applesupport_sample.jsonl"),
    golden_set_path: Path = Path("golden_set/golden_evaluation_set.jsonl"),
    output_metrics_path: Path = Path("results/baseline2_metrics.json"),
    output_predictions_path: Path = Path("results/baseline2_predictions.csv"),
    model_save_path: Path = Path("models/baseline2_tfidf_logreg.joblib"),
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Train and evaluate Baseline 2 (TF-IDF + Logistic Regression)."""
    train_corpus_path = Path(train_corpus_path)
    golden_set_path = Path(golden_set_path)
    output_metrics_path = Path(output_metrics_path)
    output_predictions_path = Path(output_predictions_path)
    model_save_path = Path(model_save_path)

    # 1. Load Golden Set to isolate evaluation targets and mask IDs
    print(f"[INFO] Loading Golden Evaluation Set from: {golden_set_path}")
    golden_records = []
    golden_conv_ids: Set[str] = set()

    with open(golden_set_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            golden_records.append(rec)
            golden_conv_ids.add(rec["conversation_id"])

    print(f"[INFO] Loaded {len(golden_records)} golden examples ({len(golden_conv_ids)} masked IDs).")

    # 2. Ingest Training Split with strict anti-contamination filtering
    print(f"[INFO] Ingesting training corpus from {train_corpus_path}...")
    X_train_text: List[str] = []
    y_train_intent: List[str] = []
    excluded_count = 0

    with open(train_corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["conversation_id"] in golden_conv_ids:
                excluded_count += 1
                continue
            inquiry = rec["customer_inquiry"]
            intent = classify_inquiry_heuristics(inquiry)
            X_train_text.append(inquiry)
            y_train_intent.append(intent)

    print(f"[INFO] Training set size: {len(X_train_text):,} (Excluded {excluded_count} Golden Set instances).")

    # 3. Fit TF-IDF Vectorizer
    print("[INFO] Vectorizing customer inquiries with TF-IDF (unigrams + bigrams, sublinear TF)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True,
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)

    # 4. Fit Balanced Logistic Regression
    print(f"[INFO] Fitting Logistic Regression (class_weight='balanced', seed={random_seed})...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=random_seed,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_vec, y_train_intent)

    # 5. Predict on Golden Evaluation Set
    X_gold_text = [rec["customer_inquiry"] for rec in golden_records]
    y_gold_true = [rec["true_intent"] for rec in golden_records]

    X_gold_vec = vectorizer.transform(X_gold_text)
    y_gold_pred = clf.predict(X_gold_vec).tolist()
    y_gold_prob = clf.predict_proba(X_gold_vec)

    # 6. Compute metrics
    labels = sorted(list(clf.classes_))
    metrics = compute_intent_metrics(y_gold_true, y_gold_pred, labels=labels)
    metrics["model_name"] = "Baseline 2: TF-IDF + Logistic Regression"
    metrics["vocabulary_size"] = len(vectorizer.vocabulary_)
    metrics["hyperparameters"] = {
        "ngram_range": [1, 2],
        "max_features": 10000,
        "class_weight": "balanced",
        "random_seed": random_seed,
    }

    # 7. Save outputs
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions CSV
    output_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_predictions_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "customer_inquiry", "true_intent", "predicted_intent", "confidence"])
        writer.writeheader()
        for i, (rec, pred) in enumerate(zip(golden_records, y_gold_pred)):
            max_conf = float(np.max(y_gold_prob[i]))
            writer.writerow({
                "example_id": rec["example_id"],
                "customer_inquiry": rec["customer_inquiry"],
                "true_intent": rec["true_intent"],
                "predicted_intent": pred,
                "confidence": round(max_conf, 4),
            })

    # Save trained pipeline model
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "classifier": clf}, model_save_path)

    print(f"[SUCCESS] Baseline 2 metrics saved to {output_metrics_path}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Baseline 2 (TF-IDF + Logistic Regression).")
    parser.add_argument("--train-corpus", type=str, default="data/sample/applesupport_sample.jsonl")
    parser.add_argument("--golden-set", type=str, default="golden_set/golden_evaluation_set.jsonl")
    parser.add_argument("--output-metrics", type=str, default="results/baseline2_metrics.json")
    parser.add_argument("--output-preds", type=str, default="results/baseline2_predictions.csv")
    parser.add_argument("--model-save", type=str, default="models/baseline2_tfidf_logreg.joblib")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_tfidf_baseline(
        train_corpus_path=Path(args.train_corpus),
        golden_set_path=Path(args.golden_set),
        output_metrics_path=Path(args.output_metrics),
        output_predictions_path=Path(args.output_preds),
        model_save_path=Path(args.model_save),
        random_seed=args.seed,
    )
