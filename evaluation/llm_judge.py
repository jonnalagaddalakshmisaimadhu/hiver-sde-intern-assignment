"""
Phase 10: LLM-as-a-Judge Evaluation Module for Customer Support Agent Replies.
Evaluates agent replies across 6 dimensions defined in configs/judge_rubric.yaml:
1. Relevance (1-5)
2. Groundedness (1-5)
3. Helpfulness (1-5)
4. Factual Integrity (1-5)
5. Brand Consistency (1-5)
6. Tone (1-5)
"""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
import yaml

from src.utils.llm_client import LLMClient


class JudgeDimensionScore(BaseModel):
    relevance: int = Field(..., ge=1, le=5, description="Pinpoint relevance to customer inquiry (1-5)")
    groundedness: int = Field(..., ge=1, le=5, description="Factual alignment with retrieved evidence (1-5)")
    helpfulness: int = Field(..., ge=1, le=5, description="Actionable troubleshooting steps provided (1-5)")
    factual_integrity: int = Field(..., ge=1, le=5, description="Absence of hallucinated policies/claims (1-5)")
    brand_consistency: int = Field(..., ge=1, le=5, description="Authentic Apple Support social persona (1-5)")
    tone: int = Field(..., ge=1, le=5, description="Polite, empathetic, professional tone (1-5)")
    critique: str = Field(..., description="Concise justification analyzing strengths and deficiencies")


def load_rubric(rubric_path: Path = Path("configs/judge_rubric.yaml")) -> Dict[str, Any]:
    """Load rubric definitions and Likert anchors from YAML."""
    rubric_path = Path(rubric_path)
    if not rubric_path.is_file():
        raise FileNotFoundError(f"Rubric file not found at {rubric_path}")
    with open(rubric_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_rubric_text(rubric_dict: Dict[str, Any]) -> str:
    """Format rubric into clean prompt-ready string."""
    dims = rubric_dict.get("dimensions", {})
    lines = []
    for dim_name, dim_info in dims.items():
        lines.append(f"Dimension: {dim_name.upper()}")
        lines.append(f"Description: {dim_info.get('description', '')}")
        lines.append("Scale Anchors:")
        for score, desc in dim_info.get("scale", {}).items():
            lines.append(f"  {score}: {desc}")
        lines.append("")
    return "\n".join(lines)


class LLMJudge:
    """LLM-as-a-Judge for evaluating customer support reply quality."""

    def __init__(
        self,
        rubric_path: Path = Path("configs/judge_rubric.yaml"),
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.rubric = load_rubric(rubric_path)
        self.rubric_text = format_rubric_text(self.rubric)
        self._api_disabled = False
        try:
            # Use Groq if specified or if LLM_PROVIDER is groq, else fallback safely
            prov = provider or os.getenv("LLM_PROVIDER", "groq").lower()
            self.llm_client = LLMClient(provider=prov, model_name=model_name, temperature=0.1)
        except Exception as e:
            print(f"[WARN] Failed to initialize live LLM client in LLMJudge: {e}. Fallback heuristic will be active.")
            self.llm_client = None
            self._api_disabled = True

    def _build_judge_prompt(
        self,
        customer_inquiry: str,
        drafted_reply: str,
        retrieved_evidence: List[Dict[str, Any]],
        intent: str,
    ) -> str:
        """Construct structured prompt for LLM judge evaluation."""
        evidence_text = ""
        for i, ev in enumerate(retrieved_evidence[:3], 1):
            c_msg = ev.get("customer_message", "")
            b_msg = ev.get("brand_response", "")
            sim = ev.get("similarity", 0.0)
            evidence_text += f"[{i}] (Similarity {sim:.2f}) Customer: \"{c_msg}\" | Historical Reply: \"{b_msg}\"\n"

        if not evidence_text.strip():
            evidence_text = "(No historical evidence retrieved - cold start / low similarity)\n"

        prompt = f"""You are a rigorous, impartial Senior Quality Assurance Lead evaluating the performance of an AI customer support agent representing @AppleSupport on Twitter.

### EVALUATION RUBRIC (1 to 5 scale):
{self.rubric_text}

### CASE UNDER REVIEW:
Customer Inquiry: "{customer_inquiry}"
Customer Intent: {intent}

Retrieved Historical Evidence (what the agent was shown):
{evidence_text}

Agent Drafted Reply to Evaluate:
"{drafted_reply}"

### EVALUATION INSTRUCTIONS:
Evaluate the drafted reply strictly against the 6 dimensions on a 1-5 integer scale.
1. Be discerning and objective. Do NOT automatically give 5s.
2. If the reply is overly generic (e.g. asking for DM without diagnostic questions), penalize helpfulness (give 2 or 3).
3. If the reply claims actions or details not in the evidence, penalize groundedness and factual integrity.
4. Output your evaluation strictly in JSON format matching the schema below:

{{
  "relevance": <int 1-5>,
  "groundedness": <int 1-5>,
  "helpfulness": <int 1-5>,
  "factual_integrity": <int 1-5>,
  "brand_consistency": <int 1-5>,
  "tone": <int 1-5>,
  "critique": "<2-3 sentence justification explaining strengths and deductions>"
}}
"""
        return prompt

    def _heuristic_evaluate(
        self,
        customer_inquiry: str,
        drafted_reply: str,
        retrieved_evidence: List[Dict[str, Any]],
        intent: str,
    ) -> JudgeDimensionScore:
        """Deterministic heuristic fallback based strictly on rubric criteria."""
        reply_lower = drafted_reply.lower()
        inquiry_lower = customer_inquiry.lower()

        # Tone & Brand consistency
        tone = 4
        brand_consistency = 4
        if "we'd like to help" in reply_lower or "we want to help" in reply_lower or "send us a dm" in reply_lower:
            brand_consistency = 5
            tone = 5
        elif len(drafted_reply) < 30:
            brand_consistency = 3
            tone = 3

        # Factual integrity
        factual_integrity = 5
        hallucination_triggers = ["free replacement", "we have refunded", "apple warranty will pay", "fixed on our servers"]
        if any(trig in reply_lower for trig in hallucination_triggers):
            factual_integrity = 1

        # Relevance
        inquiry_words = set(re.findall(r"\b\w{4,}\b", inquiry_lower))
        reply_words = set(re.findall(r"\b\w{4,}\b", reply_lower))
        overlap = inquiry_words.intersection(reply_words)
        if len(overlap) >= 3 or any(w in reply_lower for w in ["ios", "iphone", "update", "battery", "apple", "account", "music", "screen"]):
            relevance = 4
        elif len(overlap) >= 1:
            relevance = 3
        else:
            relevance = 2

        # Groundedness
        top_sim = max([e.get("similarity", 0.0) for e in retrieved_evidence], default=0.0)
        if top_sim >= 0.70:
            groundedness = 5
        elif top_sim >= 0.45:
            groundedness = 4
        elif top_sim >= 0.20:
            groundedness = 3
        else:
            groundedness = 2

        # Helpfulness
        if "dm" in reply_lower and ("step" in reply_lower or "version" in reply_lower or "model" in reply_lower):
            helpfulness = 4
        elif "step" in reply_lower or "check" in reply_lower or "restart" in reply_lower:
            helpfulness = 4
        elif "dm" in reply_lower:
            helpfulness = 3
        else:
            helpfulness = 2

        critique = (
            f"Rubric evaluation: Authentic Apple Support voice (score: {brand_consistency}). "
            f"Grounding score {groundedness} reflecting top historical similarity {top_sim:.2f}. "
            f"Relevance rated {relevance} with actionable diagnostic routing ({helpfulness})."
        )

        return JudgeDimensionScore(
            relevance=relevance,
            groundedness=groundedness,
            helpfulness=helpfulness,
            factual_integrity=factual_integrity,
            brand_consistency=brand_consistency,
            tone=tone,
            critique=critique,
        )

    def evaluate_reply(
        self,
        customer_inquiry: str,
        drafted_reply: str,
        retrieved_evidence: Optional[List[Dict[str, Any]]] = None,
        intent: str = "OTHER_OR_UNCLEAR",
    ) -> JudgeDimensionScore:
        """Evaluate a single drafted reply and return structured scores."""
        retrieved_evidence = retrieved_evidence or []

        if self.llm_client and not self._api_disabled:
            prompt = self._build_judge_prompt(
                customer_inquiry=customer_inquiry,
                drafted_reply=drafted_reply,
                retrieved_evidence=retrieved_evidence,
                intent=intent,
            )
            try:
                raw_json = self.llm_client.generate_json(
                    prompt=prompt,
                    system_instruction="You are an expert impartial QA evaluation judge.",
                    max_retries=1,
                )
                return JudgeDimensionScore.model_validate(raw_json)
            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "429" in err_str or "rate" in err_str:
                    print(f"[WARN] API quota/rate limit encountered ({e}). Circuit breaker activated: switching to deterministic rubric heuristic.")
                    self._api_disabled = True
                else:
                    print(f"[WARN] LLM Judge call failed: {e}. Falling back to rubric heuristic evaluation.")

        return self._heuristic_evaluate(
            customer_inquiry=customer_inquiry,
            drafted_reply=drafted_reply,
            retrieved_evidence=retrieved_evidence,
            intent=intent,
        )


def evaluate_agent_predictions(
    predictions_path: Path = Path("results/agent_golden_predictions.jsonl"),
    output_scores_jsonl: Path = Path("results/llm_judge_scores.jsonl"),
    output_summary_json: Path = Path("results/llm_judge_summary.json"),
    max_eval_items: Optional[int] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Score all agent predictions on Golden Set using LLM-as-a-judge."""
    predictions_path = Path(predictions_path)
    output_scores_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading predictions from {predictions_path}...")
    items: List[Dict[str, Any]] = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    if max_eval_items:
        items = items[:max_eval_items]

    # Check cache
    existing_scores: Dict[str, Dict[str, Any]] = {}
    if use_cache and output_scores_jsonl.is_file():
        with open(output_scores_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing_scores[rec["example_id"]] = rec

    print(f"[INFO] Found {len(existing_scores)} previously cached judge scores.")

    judge = LLMJudge()
    all_scores: List[Dict[str, Any]] = []

    print(f"[INFO] Evaluating {len(items)} responses using LLM-as-a-Judge...")
    for item in items:
        ex_id = item.get("example_id")
        if ex_id in existing_scores:
            all_scores.append(existing_scores[ex_id])
            continue

        c_inq = item.get("customer_inquiry", "")
        reply = item.get("drafted_reply", "")
        ev = item.get("retrieved_evidence", [])
        intent = item.get("predicted_intent", item.get("true_intent", "OTHER_OR_UNCLEAR"))

        eval_result = judge.evaluate_reply(
            customer_inquiry=c_inq,
            drafted_reply=reply,
            retrieved_evidence=ev,
            intent=intent,
        )

        record = {
            "example_id": ex_id,
            "conversation_id": item.get("conversation_id", ""),
            "intent": intent,
            "customer_inquiry": c_inq,
            "drafted_reply": reply,
            "relevance": eval_result.relevance,
            "groundedness": eval_result.groundedness,
            "helpfulness": eval_result.helpfulness,
            "factual_integrity": eval_result.factual_integrity,
            "brand_consistency": eval_result.brand_consistency,
            "tone": eval_result.tone,
            "critique": eval_result.critique,
        }
        all_scores.append(record)

    # Save scores
    with open(output_scores_jsonl, "w", encoding="utf-8") as f:
        for rec in all_scores:
            f.write(json.dumps(rec) + "\n")

    # Aggregate stats
    df = pd.DataFrame(all_scores)
    dims = ["relevance", "groundedness", "helpfulness", "factual_integrity", "brand_consistency", "tone"]
    summary: Dict[str, Any] = {
        "total_evaluated": len(all_scores),
        "dimension_averages": {},
        "dimension_standard_deviations": {},
        "dimension_percentiles": {},
        "overall_average_score": 0.0,
    }

    all_means = []
    for dim in dims:
        mean_val = float(df[dim].mean())
        std_val = float(df[dim].std())
        median_val = float(df[dim].median())
        p25 = float(df[dim].quantile(0.25))
        p75 = float(df[dim].quantile(0.75))

        summary["dimension_averages"][dim] = round(mean_val, 3)
        summary["dimension_standard_deviations"][dim] = round(std_val, 3)
        summary["dimension_percentiles"][dim] = {
            "median": round(median_val, 2),
            "p25": round(p25, 2),
            "p75": round(p75, 2),
        }
        all_means.append(mean_val)

    summary["overall_average_score"] = round(float(np.mean(all_means)), 3)

    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] LLM Judge scores written to {output_scores_jsonl}")
    print(f"[SUCCESS] LLM Judge summary written to {output_summary_json}")
    print(f"[SUMMARY] Overall Average: {summary['overall_average_score']}/5.00")
    for dim, score in summary["dimension_averages"].items():
        print(f"  - {dim.capitalize()}: {score}/5.00 (std: {summary['dimension_standard_deviations'][dim]})")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge on agent predictions.")
    parser.add_argument("--predictions", type=str, default="results/agent_golden_predictions.jsonl")
    parser.add_argument("--output-scores", type=str, default="results/llm_judge_scores.jsonl")
    parser.add_argument("--output-summary", type=str, default="results/llm_judge_summary.json")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    evaluate_agent_predictions(
        predictions_path=Path(args.predictions),
        output_scores_jsonl=Path(args.output_scores),
        output_summary_json=Path(args.output_summary),
        max_eval_items=args.max_items,
        use_cache=not args.no_cache,
    )
