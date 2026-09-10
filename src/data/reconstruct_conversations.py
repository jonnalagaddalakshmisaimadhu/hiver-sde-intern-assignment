"""
Phase 4: Conversation reconstruction and cleaning pipeline for @AppleSupport.
Links inbound customer inquiries with outbound brand responses using parent-child tweet pointers,
enforces chronological ordering, removes duplicates, and generates audit-ready conversation threads.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


TARGET_BRAND = "AppleSupport"
DEFAULT_RAW_CSV = Path("data/raw/twcs.csv")
DEFAULT_OUTPUT_JSONL = Path("data/processed/applesupport_conversations.jsonl")
DEFAULT_SAMPLE_JSONL = Path("data/sample/applesupport_sample.jsonl")
DEFAULT_SUMMARY_PATH = Path("results/conversation_reconstruction_summary.json")


def clean_text_preserving_natural_tone(text: str) -> str:
    """Normalize whitespace and control characters while strictly preserving natural Twitter idioms, typos, and punctuation."""
    if not isinstance(text, str):
        return ""
    # Strip non-printable control characters, normalize excessive whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def reconstruct_apple_conversations(
    raw_csv_path: Path = DEFAULT_RAW_CSV,
    output_jsonl: Path = DEFAULT_OUTPUT_JSONL,
    sample_jsonl: Path = DEFAULT_SAMPLE_JSONL,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
    sample_size: int = 5000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Reconstruct multi-turn dialogues between customers and AppleSupport."""
    start_time = time.time()
    raw_csv_path = Path(raw_csv_path)
    output_jsonl = Path(output_jsonl)
    sample_jsonl = Path(sample_jsonl)
    summary_path = Path(summary_path)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    sample_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Streaming raw data from {raw_csv_path} to isolate @{TARGET_BRAND} ecosystem...")

    # Step 1: Collect all outbound AppleSupport tweets and their referenced tweet IDs
    apple_outbound_tweets: Dict[int, Dict[str, Any]] = {}
    parent_ids_needed: Set[int] = set()

    chunk_size = 250_000
    for chunk in pd.read_csv(raw_csv_path, chunksize=chunk_size, low_memory=False):
        apple_mask = (chunk["author_id"] == TARGET_BRAND) & (~chunk["inbound"].astype(bool))
        if apple_mask.any():
            matched = chunk[apple_mask]
            for _, row in matched.iterrows():
                tid = int(row["tweet_id"])
                in_resp = row["in_response_to_tweet_id"]
                parent_id = int(in_resp) if pd.notna(in_resp) and str(in_resp).replace(".", "").isdigit() else None
                
                apple_outbound_tweets[tid] = {
                    "tweet_id": tid,
                    "author_id": TARGET_BRAND,
                    "inbound": False,
                    "created_at": row["created_at"],
                    "text": clean_text_preserving_natural_tone(str(row["text"])),
                    "in_response_to_tweet_id": parent_id,
                    "response_tweet_id": str(row["response_tweet_id"]) if pd.notna(row["response_tweet_id"]) else None,
                }
                if parent_id is not None:
                    parent_ids_needed.add(parent_id)

    print(f"[INFO] Found {len(apple_outbound_tweets):,} outbound @{TARGET_BRAND} tweets.")
    print(f"[INFO] Looking for {len(parent_ids_needed):,} customer parent tweet IDs...")

    # Step 2: Second streaming pass to collect customer tweets referenced by AppleSupport
    customer_inbound_tweets: Dict[int, Dict[str, Any]] = {}

    for chunk in pd.read_csv(raw_csv_path, chunksize=chunk_size, low_memory=False):
        # We need tweets whose tweet_id is in parent_ids_needed OR whose text mentions @AppleSupport
        inbound_chunk = chunk[chunk["inbound"].astype(bool)]
        matched = inbound_chunk[inbound_chunk["tweet_id"].isin(parent_ids_needed)]
        if not matched.empty:
            for _, row in matched.iterrows():
                tid = int(row["tweet_id"])
                in_resp = row["in_response_to_tweet_id"]
                parent_id = int(in_resp) if pd.notna(in_resp) and str(in_resp).replace(".", "").isdigit() else None
                
                customer_inbound_tweets[tid] = {
                    "tweet_id": tid,
                    "author_id": str(row["author_id"]),
                    "inbound": True,
                    "created_at": row["created_at"],
                    "text": clean_text_preserving_natural_tone(str(row["text"])),
                    "in_response_to_tweet_id": parent_id,
                    "response_tweet_id": str(row["response_tweet_id"]) if pd.notna(row["response_tweet_id"]) else None,
                }

    print(f"[INFO] Retrieved {len(customer_inbound_tweets):,} linked customer inquiries.")

    # Step 3: Conversation Stitching & Deduplication
    # A canonical support interaction starts with a customer inquiry and is followed by AppleSupport's resolution
    conversations: List[Dict[str, Any]] = []
    seen_inquiry_texts: Set[str] = set()
    exact_duplicates_dropped = 0
    short_inquiries_dropped = 0

    for brand_tid, brand_tweet in apple_outbound_tweets.items():
        parent_id = brand_tweet["in_response_to_tweet_id"]
        if parent_id is None or parent_id not in customer_inbound_tweets:
            continue

        cust_tweet = customer_inbound_tweets[parent_id]
        cust_text = cust_tweet["text"]
        brand_text = brand_tweet["text"]

        # Cleaning rule: customer inquiry must have at least 3 words / 10 chars
        if len(cust_text) < 10 or len(cust_text.split()) < 3:
            short_inquiries_dropped += 1
            continue

        # Deduplication rule: avoid exact duplicate customer queries that arise from retweets or bot repetition
        normalized_inquiry = cust_text.lower().strip()
        if normalized_inquiry in seen_inquiry_texts:
            exact_duplicates_dropped += 1
            continue
        seen_inquiry_texts.add(normalized_inquiry)

        # Build turns
        turns = [
            {
                "turn_index": 0,
                "speaker": "customer",
                "author_id": cust_tweet["author_id"],
                "tweet_id": cust_tweet["tweet_id"],
                "created_at": cust_tweet["created_at"],
                "text": cust_text,
            },
            {
                "turn_index": 1,
                "speaker": "brand",
                "author_id": TARGET_BRAND,
                "tweet_id": brand_tweet["tweet_id"],
                "created_at": brand_tweet["created_at"],
                "text": brand_text,
            }
        ]

        conversation_id = f"apple_{cust_tweet['tweet_id']}"
        conv_record = {
            "conversation_id": conversation_id,
            "brand": TARGET_BRAND,
            "customer_id": cust_tweet["author_id"],
            "turn_count": len(turns),
            "customer_inquiry": cust_text,
            "brand_initial_response": brand_text,
            "turns": turns,
            "has_multi_turn_followup": brand_tweet["response_tweet_id"] is not None,
        }
        conversations.append(conv_record)

    total_conversations = len(conversations)
    print(f"[INFO] Reconstructed {total_conversations:,} unique customer-support conversations.")
    print(f"[INFO] Dropped {exact_duplicates_dropped:,} exact duplicate inquiries and {short_inquiries_dropped:,} uninformative short queries.")

    # Write full dataset to JSONL
    print(f"[INFO] Writing full conversations to {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    # Write deterministic subsample for <15 min reproduction
    import random
    rng = random.Random(random_seed)
    sampled_conversations = rng.sample(conversations, min(sample_size, total_conversations))

    print(f"[INFO] Writing {len(sampled_conversations):,} sampled conversations to {sample_jsonl}...")
    with open(sample_jsonl, "w", encoding="utf-8") as f:
        for conv in sampled_conversations:
            f.write(json.dumps(conv) + "\n")

    elapsed = time.time() - start_time

    # Summary metrics
    summary = {
        "brand": TARGET_BRAND,
        "raw_source_file": str(raw_csv_path),
        "total_apple_outbound_tweets": len(apple_outbound_tweets),
        "linked_customer_inbound_tweets": len(customer_inbound_tweets),
        "total_reconstructed_conversations": total_conversations,
        "exact_duplicates_filtered": exact_duplicates_dropped,
        "short_unusable_queries_filtered": short_inquiries_dropped,
        "sample_size": len(sampled_conversations),
        "sample_random_seed": random_seed,
        "execution_time_seconds": round(elapsed, 2),
        "output_files": {
            "full_processed": str(output_jsonl),
            "sample_subsample": str(sample_jsonl),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[SUCCESS] Conversation reconstruction complete in {elapsed:.1f}s. Summary saved to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct and clean AppleSupport conversations.")
    parser.add_argument("--raw-csv", type=str, default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--sample-output", type=str, default=str(DEFAULT_SAMPLE_JSONL))
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reconstruct_apple_conversations(
        raw_csv_path=Path(args.raw_csv),
        output_jsonl=Path(args.output),
        sample_jsonl=Path(args.sample_output),
        sample_size=args.sample_size,
        random_seed=args.seed,
    )
