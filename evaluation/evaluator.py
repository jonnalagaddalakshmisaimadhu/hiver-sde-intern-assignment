"""
Phase 9: Automated Evaluation Harness for @AppleSupport AI Support Agent.
Executes the agent across all 200 Golden Evaluation Set examples, computes intent
classification metrics (Macro-F1, per-intent P/R/F1, confusion matrix), evaluates
escalation quality (Precision, Recall, F1, False Auto-Handle Rate), and generates
comparative benchmark tables against Baseline 1 and Baseline 2.
"""
import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from evaluation.metrics import compute_intent_metrics, compute_escalation_metrics
from src.agent.agent import AppleSupportAgent, AgentResponse


def run_agent_evaluation(
    golden_set_path: Path = Path("golden_set/golden_evaluation_set.jsonl"),
    output_predictions_csv: Path = Path("results/agent_predictions.csv"),
    output_predictions_jsonl: Path = Path("results/agent_golden_predictions.jsonl"),
    output_intent_metrics: Path = Path("results/agent_intent_metrics.json"),
    output_escalation_metrics: Path = Path("results/agent_escalation_metrics.json"),
    output_comparison_path: Path = Path("results/system_comparison.json"),
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation suite on Golden Set."""
    golden_set_path = Path(golden_set_path)
    output_predictions_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading Golden Evaluation Set from {golden_set_path}...")
    golden_examples = []
    with open(golden_set_path, "r", encoding="utf-8") as f:
        for line in f:
            golden_examples.append(json.loads(line))

    print(f"[INFO] Evaluating agent on {len(golden_examples)} Golden Set examples...")

    agent_responses: List[Dict[str, Any]] = []

    # Check if predictions already cached to avoid duplicate API spend
    if use_cache and output_predictions_jsonl.is_file():
        cached_lines = []
        with open(output_predictions_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                cached_lines.append(json.loads(line))
        if len(cached_lines) == len(golden_examples):
            print(f"[INFO] Restoring {len(cached_lines)} agent predictions from cache {output_predictions_jsonl}...")
            agent_responses = cached_lines

    if not agent_responses:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        agent = AppleSupportAgent()
        start_time = time.time()

        def evaluate_single(ex: Dict[str, Any]) -> Dict[str, Any]:
            resp = agent.process_inquiry(ex["customer_inquiry"])
            return {
                "example_id": ex["example_id"],
                "conversation_id": ex["conversation_id"],
                "customer_inquiry": ex["customer_inquiry"],
                "true_intent": ex["true_intent"],
                "predicted_intent": resp.intent,
                "true_escalation_decision": ex["true_escalation_decision"],
                "predicted_escalation_decision": resp.decision,
                "escalation_rationale": resp.reason,
                "drafted_reply": resp.reply,
                "ground_truth_brand_reply": ex["ground_truth_brand_reply"],
                "top_similarity": resp.evidence[0].similarity if resp.evidence else 0.0,
                "retrieved_evidence": [e.model_dump() for e in resp.evidence],
                "intent_confidence": resp.intent_confidence,
            }

        print(f"[INFO] Running concurrent evaluation across 5 worker threads...")
        evaluated_dict: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_id = {executor.submit(evaluate_single, ex): ex["example_id"] for ex in golden_examples}
            with tqdm(total=len(golden_examples), desc="Evaluating Agent") as pbar:
                for future in as_completed(future_to_id):
                    ex_id = future_to_id[future]
                    try:
                        res = future.result()
                        evaluated_dict[ex_id] = res
                    except Exception as exc:
                        print(f"[ERROR] Example {ex_id} failed: {exc}")
                    pbar.update(1)

        # Restore canonical order
        for ex in golden_examples:
            if ex["example_id"] in evaluated_dict:
                agent_responses.append(evaluated_dict[ex["example_id"]])

        elapsed = time.time() - start_time
        print(f"[INFO] Evaluation inference completed in {elapsed:.1f}s ({elapsed/len(golden_examples):.2f}s/example).")

        # Cache predictions to JSONL
        with open(output_predictions_jsonl, "w", encoding="utf-8") as f:
            for rec in agent_responses:
                f.write(json.dumps(rec) + "\n")

    # Export predictions CSV
    fieldnames = [
        "example_id",
        "customer_inquiry",
        "true_intent",
        "predicted_intent",
        "true_escalation_decision",
        "predicted_escalation_decision",
        "escalation_rationale",
        "drafted_reply",
        "top_similarity",
        "intent_confidence",
    ]
    with open(output_predictions_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in agent_responses:
            writer.writerow(r)

    # 1. Compute Intent Metrics
    y_intent_true = [r["true_intent"] for r in agent_responses]
    y_intent_pred = [r["predicted_intent"] for r in agent_responses]
    labels = sorted(list(set(y_intent_true)))
    intent_metrics = compute_intent_metrics(y_intent_true, y_intent_pred, labels=labels)
    intent_metrics["model_name"] = "AppleSupport AI Agent"

    with open(output_intent_metrics, "w", encoding="utf-8") as f:
        json.dump(intent_metrics, f, indent=2)

    # 2. Compute Escalation Metrics
    y_esc_true = [r["true_escalation_decision"] for r in agent_responses]
    y_esc_pred = [r["predicted_escalation_decision"] for r in agent_responses]
    escalation_metrics = compute_escalation_metrics(y_esc_true, y_esc_pred)
    escalation_metrics["model_name"] = "AppleSupport AI Agent"

    with open(output_escalation_metrics, "w", encoding="utf-8") as f:
        json.dump(escalation_metrics, f, indent=2)

    # 3. Load Baseline Metrics for Side-by-Side Comparison
    b1_path = Path("results/baseline1_metrics.json")
    b2_path = Path("results/baseline2_metrics.json")
    b1_metrics = json.loads(b1_path.read_text(encoding="utf-8")) if b1_path.is_file() else {}
    b2_metrics = json.loads(b2_path.read_text(encoding="utf-8")) if b2_path.is_file() else {}

    comparison = {
        "dataset": "AppleSupport Golden Evaluation Set (N=200)",
        "intent_classification": {
            "Baseline 1 (Majority Class)": {
                "accuracy": b1_metrics.get("accuracy", 0.0),
                "macro_f1": b1_metrics.get("macro_f1", 0.0),
                "macro_precision": b1_metrics.get("macro_precision", 0.0),
                "macro_recall": b1_metrics.get("macro_recall", 0.0),
            },
            "Baseline 2 (TF-IDF + LogReg)": {
                "accuracy": b2_metrics.get("accuracy", 0.0),
                "macro_f1": b2_metrics.get("macro_f1", 0.0),
                "macro_precision": b2_metrics.get("macro_precision", 0.0),
                "macro_recall": b2_metrics.get("macro_recall", 0.0),
            },
            "AppleSupport AI Agent": {
                "accuracy": intent_metrics["accuracy"],
                "macro_f1": intent_metrics["macro_f1"],
                "macro_precision": intent_metrics["macro_precision"],
                "macro_recall": intent_metrics["macro_recall"],
            },
        },
        "escalation_performance": {
            "AppleSupport AI Agent": {
                "accuracy": escalation_metrics["accuracy"],
                "precision": escalation_metrics["escalation_precision"],
                "recall": escalation_metrics["escalation_recall"],
                "f1_score": escalation_metrics["escalation_f1"],
                "false_auto_handle_rate": escalation_metrics["false_auto_handle_rate"],
                "confusion_matrix": escalation_metrics["confusion_matrix"],
            }
        }
    }

    with open(output_comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION HARNESS SUMMARY (PHASE 9)")
    print("=" * 60)
    print(f"Total Evaluated: {len(agent_responses)}")
    print("\nIntent Classification Performance:")
    print(f"  Baseline 1 (Majority): Accuracy = {b1_metrics.get('accuracy', 0.0):.4f} | Macro-F1 = {b1_metrics.get('macro_f1', 0.0):.4f}")
    print(f"  Baseline 2 (TF-IDF):   Accuracy = {b2_metrics.get('accuracy', 0.0):.4f} | Macro-F1 = {b2_metrics.get('macro_f1', 0.0):.4f}")
    print(f"  AI Support Agent:      Accuracy = {intent_metrics['accuracy']:.4f} | Macro-F1 = {intent_metrics['macro_f1']:.4f}")
    print("\nEscalation Performance (AI Agent):")
    print(f"  Escalation Precision: {escalation_metrics['escalation_precision']:.4f}")
    print(f"  Escalation Recall:    {escalation_metrics['escalation_recall']:.4f}")
    print(f"  Escalation F1:        {escalation_metrics['escalation_f1']:.4f}")
    print(f"  False Auto-Handle Rate (FAHR): {escalation_metrics['false_auto_handle_rate']:.4f}")
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full evaluation harness.")
    parser.add_argument("--golden-set", type=str, default="golden_set/golden_evaluation_set.jsonl")
    parser.add_argument("--no-cache", action="store_true", help="Do not use cached predictions")
    args = parser.parse_args()

    run_agent_evaluation(
        golden_set_path=Path(args.golden_set),
        use_cache=not args.no_cache,
    )
