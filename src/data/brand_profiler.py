"""
Brand profiling and comparative selection pipeline for Phase 3.
Analyzes top candidate brands across conversation completeness, multi-turn depth,
response informativeness (troubleshooting vs. generic canned DM redirect),
and Golden Set annotation feasibility.
"""
import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd


CANDIDATE_BRANDS = [
    "AppleSupport",
    "AmazonHelp",
    "SpotifyCares",
    "Uber_Support",
    "Delta",
]

DM_REDIRECT_PATTERNS = [
    r"\bdm\b",
    r"direct message",
    r"send us a message",
    r"reach out via dm",
    r"pm us",
    r"private message",
]


def profile_candidate_brands(
    csv_path: Path = Path("data/raw/twcs.csv"),
    candidate_brands: List[str] = CANDIDATE_BRANDS,
    output_path: Path = Path("results/brand_profiling_report.json"),
    max_rows_per_brand: int = 150_000,
) -> Dict[str, Any]:
    """Profile candidate brands across conversational and qualitative dimensions."""
    print(f"[INFO] Profiling candidate brands from: {csv_path}")
    print(f"[INFO] Candidates: {', '.join(candidate_brands)}")

    # We read twcs.csv and filter for tweets authored by or directed to candidate brands
    brand_outbound: Dict[str, List[Dict[str, Any]]] = {b: [] for b in candidate_brands}
    
    # Track candidate brand tweets
    candidate_set = set(candidate_brands)
    
    chunk_size = 250_000
    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
        # Filter for brand outbound tweets
        brand_mask = chunk["author_id"].isin(candidate_set)
        if brand_mask.any():
            matched = chunk[brand_mask]
            for _, row in matched.iterrows():
                brand = row["author_id"]
                if len(brand_outbound[brand]) < max_rows_per_brand:
                    brand_outbound[brand].append({
                        "tweet_id": row["tweet_id"],
                        "created_at": row["created_at"],
                        "text": str(row["text"]),
                        "in_response_to_tweet_id": row["in_response_to_tweet_id"],
                        "response_tweet_id": row["response_tweet_id"],
                    })

        all_filled = all(len(brand_outbound[b]) >= max_rows_per_brand for b in candidate_brands)
        if all_filled:
            print(f"[INFO] Reached sampling threshold ({max_rows_per_brand:,} per brand) early at chunk {chunk_idx + 1}.")
            break

    print("[INFO] Outbound tweets collected. Analyzing conversational depth and response characteristics...")

    results = {}

    for brand in candidate_brands:
        tweets = brand_outbound[brand]
        total_brand_tweets = len(tweets)
        if total_brand_tweets == 0:
            continue

        df = pd.DataFrame(tweets)

        # 1. Replies to customer tweets (has in_response_to_tweet_id)
        replies_to_customer = df["in_response_to_tweet_id"].notna().sum()
        reply_ratio = round((replies_to_customer / total_brand_tweets) * 100, 2)

        # 2. Multi-turn responses (has response_tweet_id indicating continued discussion)
        has_followup = df["response_tweet_id"].notna().sum()
        followup_ratio = round((has_followup / total_brand_tweets) * 100, 2)

        # 3. Informative Resolution vs. Generic Canned DM Redirect
        texts = df["text"].fillna("").str.lower()
        dm_regex = re.compile("|".join(DM_REDIRECT_PATTERNS), re.IGNORECASE)
        dm_redirect_count = texts.str.contains(dm_regex).sum()
        dm_redirect_pct = round((dm_redirect_count / total_brand_tweets) * 100, 2)
        informative_resolution_pct = round(100.0 - dm_redirect_pct, 2)

        # 4. Text length metrics
        char_lens = texts.str.len()
        mean_char_len = round(float(char_lens.mean()), 1)

        # 5. Non-English detection heuristic (e.g. non-ascii character ratio)
        non_ascii_ratios = [len(re.findall(r"[^\x00-\x7F]", t)) / max(1, len(t)) for t in texts]
        substantially_non_ascii = sum(r > 0.15 for r in non_ascii_ratios)
        non_english_pct = round((substantially_non_ascii / total_brand_tweets) * 100, 2)

        # 6. Technical troubleshooting signal (presence of action verbs / technical terms)
        tech_keywords = r"\b(update|settings|restart|device|version|ios|app|account|password|reset|link|steps|try|error|install)\b"
        troubleshoot_count = texts.str.contains(tech_keywords, regex=True).sum()
        troubleshoot_pct = round((troubleshoot_count / total_brand_tweets) * 100, 2)

        results[brand] = {
            "total_outbound_sampled": total_brand_tweets,
            "replies_to_customer_count": int(replies_to_customer),
            "reply_ratio_pct": reply_ratio,
            "multi_turn_followup_count": int(has_followup),
            "multi_turn_followup_pct": followup_ratio,
            "canned_dm_redirect_pct": dm_redirect_pct,
            "informative_content_pct": informative_resolution_pct,
            "troubleshooting_keyword_pct": troubleshoot_pct,
            "non_english_heuristic_pct": non_english_pct,
            "mean_response_char_length": mean_char_len,
        }

    # Comparative evaluation & selection recommendation
    report = {
        "candidate_brand_profiles": results,
        "selection_criteria": {
            "conversational_volume": "Sufficient volume for clean train/retrieval split and 150-250 Golden Set.",
            "informative_content": "Lower percentage of pure canned DM redirects; higher percentage of self-contained technical troubleshooting.",
            "intent_diversity": "Rich variety of diagnostic intents rather than single-issue churn.",
            "cleanliness": "Predominantly English and consistent support persona.",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Brand profiling report saved to: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile candidate brands for selection.")
    parser.add_argument("--csv-path", type=str, default="data/raw/twcs.csv", help="Path to raw twcs.csv")
    parser.add_argument("--output", type=str, default="results/brand_profiling_report.json", help="Report path")
    args = parser.parse_args()

    report = profile_candidate_brands(Path(args.csv_path), output_path=Path(args.output))
    print("\n" + "=" * 70)
    print("BRAND PROFILING SUMMARY (PHASE 3)")
    print("=" * 70)
    for brand, metrics in report["candidate_brand_profiles"].items():
        print(f"\nBrand: @{brand}")
        print(f"  Sampled Outbound Tweets: {metrics['total_outbound_sampled']:,}")
        print(f"  Replies to Inbound Inquiries: {metrics['reply_ratio_pct']}%")
        print(f"  Multi-turn Followup Ratio: {metrics['multi_turn_followup_pct']}%")
        print(f"  Troubleshooting Keyword Density: {metrics['troubleshooting_keyword_pct']}%")
        print(f"  Canned DM Redirect Ratio: {metrics['canned_dm_redirect_pct']}%")
        print(f"  Informative Content Ratio: {metrics['informative_content_pct']}%")
        print(f"  Non-English Ratio: {metrics['non_english_heuristic_pct']}%")
