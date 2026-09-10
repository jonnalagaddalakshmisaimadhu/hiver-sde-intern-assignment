"""
Standardized evaluation metrics calculator for Intent Classification and Escalation.
Calculates Accuracy, Macro-F1, Weighted-F1, per-intent Precision/Recall/F1, and confusion matrix.
"""
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_intent_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for intent prediction."""
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_p = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Per-class report
    clf_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_intent = {}
    for label in labels:
        if label in clf_report:
            per_intent[label] = {
                "precision": round(float(clf_report[label]["precision"]), 4),
                "recall": round(float(clf_report[label]["recall"]), 4),
                "f1_score": round(float(clf_report[label]["f1-score"]), 4),
                "support": int(clf_report[label]["support"]),
            }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_evaluated": len(y_true),
        "labels": labels,
        "confusion_matrix": cm,
        "per_intent_metrics": per_intent,
    }


def compute_escalation_metrics(
    y_true: List[str],
    y_pred: List[str],
    escalate_label: str = "ESCALATE",
    auto_handle_label: str = "AUTO_HANDLE",
) -> Dict[str, Any]:
    """Calculate escalation metrics focusing on safety and False Auto-Handle Rate."""
    # Binary mapping: 1 for ESCALATE, 0 for AUTO_HANDLE
    y_true_bin = [1 if y == escalate_label else 0 for y in y_true]
    y_pred_bin = [1 if y == escalate_label else 0 for y in y_pred]

    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    # Confusion matrix
    # [ [True Auto-Handle, False Escalate],
    #   [False Auto-Handle, True Escalate] ]
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist()
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    # False Auto-Handle Rate (FAHR): FN / (FN + TP)
    # The proportion of cases needing human escalation that were mistakenly auto-handled
    fahr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": round(acc, 4),
        "escalation_precision": round(prec, 4),
        "escalation_recall": round(rec, 4),
        "escalation_f1": round(f1, 4),
        "false_auto_handle_rate": round(fahr, 4),
        "true_positives_escalate": int(tp),
        "false_negatives_auto_handled": int(fn),
        "true_negatives_auto_handle": int(tn),
        "false_positives_escalated": int(fp),
        "confusion_matrix": cm,
    }
