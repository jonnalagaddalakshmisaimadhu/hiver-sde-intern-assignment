"""
Comprehensive data inspection pipeline for the raw Twitter Customer Support dataset (twcs.csv).
Analyzes schema, row counts, nullability, duplicates, thread linkage, brand distributions,
and textual characteristics without assuming any schema a priori.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def inspect_dataset(
    csv_path: Path,
    output_report_path: Path = Path("results/dataset_inspection_report.json"),
    chunk_size: int = 250_000,
) -> Dict[str, Any]:
    """Inspect twcs.csv using memory-efficient chunked streaming."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    file_size_bytes = csv_path.stat().st_size
    print(f"[INFO] Inspecting dataset: {csv_path} ({file_size_bytes:,} bytes)")

    # 1. Preview first 5 rows to capture exact headers and raw representation
    preview_df = pd.read_csv(csv_path, nrows=5)
    columns = list(preview_df.columns)
    print(f"[INFO] Detected Columns ({len(columns)}): {columns}")

    # Initialize accumulators
    total_rows = 0
    null_counts = {col: 0 for col in columns}
    dtypes = {col: str(preview_df[col].dtype) for col in columns}
    
    inbound_counts = {"True": 0, "False": 0, "Other": 0}
    author_counts: Dict[str, int] = {}
    
    # Thread linkage metrics
    has_response_tweet_id = 0
    has_in_response_to_id = 0
    starts_thread_count = 0  # in_response_to_tweet_id is NaN
    
    # Text length stats accumulators
    text_char_lengths = []
    text_word_lengths = []
    empty_text_count = 0
    
    # Timestamps
    min_date = None
    max_date = None
    
    start_time = time.time()
    print(f"[INFO] Processing dataset in chunks of {chunk_size:,} rows...")

    for chunk_idx, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False)):
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        
        # Missing values
        for col in columns:
            null_counts[col] += int(chunk[col].isna().sum())

        # Inbound distribution
        if "inbound" in chunk.columns:
            inbound_s = chunk["inbound"].astype(str).str.strip().str.capitalize()
            vc = inbound_s.value_counts().to_dict()
            inbound_counts["True"] += vc.get("True", 0)
            inbound_counts["False"] += vc.get("False", 0)
            for k, v in vc.items():
                if k not in ("True", "False"):
                    inbound_counts["Other"] += v

        # Author distribution (focusing on brand authors vs customer IDs)
        if "author_id" in chunk.columns:
            author_vc = chunk["author_id"].value_counts()
            for author, count in author_vc.items():
                author_str = str(author)
                author_counts[author_str] = author_counts.get(author_str, 0) + int(count)

        # Thread linkage
        if "response_tweet_id" in chunk.columns:
            has_response_tweet_id += int(chunk["response_tweet_id"].notna().sum())
        if "in_response_to_tweet_id" in chunk.columns:
            has_in_resp = chunk["in_response_to_tweet_id"].notna()
            has_in_response_to_id += int(has_in_resp.sum())
            starts_thread_count += int((~has_in_resp).sum())

        # Text metrics (sample a fraction for length distribution to preserve memory)
        if "text" in chunk.columns:
            empty_text_count += int(chunk["text"].isna().sum() + (chunk["text"].astype(str).str.strip() == "").sum())
            sampled_text = chunk["text"].dropna().sample(frac=0.05, random_state=42)
            char_lens = sampled_text.astype(str).str.len().tolist()
            word_lens = sampled_text.astype(str).str.split().str.len().tolist()
            text_char_lengths.extend(char_lens)
            text_word_lengths.extend(word_lens)

        # Timestamps
        if "created_at" in chunk.columns:
            chunk_min_date = chunk["created_at"].dropna().min()
            chunk_max_date = chunk["created_at"].dropna().max()
            if min_date is None or (chunk_min_date and chunk_min_date < min_date):
                min_date = str(chunk_min_date)
            if max_date is None or (chunk_max_date and chunk_max_date > max_date):
                max_date = str(chunk_max_date)

        print(f"  Processed chunk {chunk_idx + 1} ({total_rows:,} rows total)...")

    elapsed = time.time() - start_time
    print(f"[INFO] Streaming complete in {elapsed:.1f}s. Total rows: {total_rows:,}")

    # Distinguish brand authors from customer authors:
    # In TWCS, brands have named handles (e.g., 'AppleSupport', 'AmazonHelp', 'SpotifyCares')
    # whereas customers are anonymized numbers or anonymized strings without alphabetical brand names.
    brand_candidates = {}
    customer_author_count = 0
    for author, count in author_counts.items():
        # Heuristic inspection: non-numeric handles are brand accounts
        if not author.isdigit():
            brand_candidates[author] = count
        else:
            customer_author_count += 1

    sorted_brands = sorted(brand_candidates.items(), key=lambda x: x[1], reverse=True)

    char_arr = np.array(text_char_lengths)
    word_arr = np.array(text_word_lengths)

    report = {
        "file_metadata": {
            "file_path": str(csv_path),
            "file_size_bytes": file_size_bytes,
            "total_rows": total_rows,
            "total_columns": len(columns),
            "columns": columns,
            "dtypes": dtypes,
        },
        "data_quality": {
            "null_counts": null_counts,
            "null_percentages": {col: round((null_counts[col] / total_rows) * 100, 3) for col in columns},
            "empty_or_whitespace_text_count": empty_text_count,
        },
        "tweet_types": {
            "inbound_count": inbound_counts["True"],
            "inbound_percentage": round((inbound_counts["True"] / total_rows) * 100, 2),
            "outbound_count": inbound_counts["False"],
            "outbound_percentage": round((inbound_counts["False"] / total_rows) * 100, 2),
            "other_count": inbound_counts["Other"],
        },
        "thread_linkage": {
            "has_response_tweet_id_count": has_response_tweet_id,
            "has_response_tweet_id_pct": round((has_response_tweet_id / total_rows) * 100, 2),
            "has_in_response_to_id_count": has_in_response_to_id,
            "has_in_response_to_id_pct": round((has_in_response_to_id / total_rows) * 100, 2),
            "thread_initiating_tweets_count": starts_thread_count,
            "thread_initiating_tweets_pct": round((starts_thread_count / total_rows) * 100, 2),
        },
        "authors": {
            "total_unique_authors": len(author_counts),
            "customer_authors_count": customer_author_count,
            "brand_handles_count": len(brand_candidates),
            "top_25_brands_by_tweet_volume": [
                {"brand": brand, "tweet_count": count} for brand, count in sorted_brands[:25]
            ],
        },
        "text_statistics": {
            "char_length": {
                "mean": round(float(np.mean(char_arr)), 1),
                "median": round(float(np.median(char_arr)), 1),
                "p95": round(float(np.percentile(char_arr, 95)), 1),
                "min": int(np.min(char_arr)),
                "max": int(np.max(char_arr)),
            },
            "word_length": {
                "mean": round(float(np.mean(word_arr)), 1),
                "median": round(float(np.median(word_arr)), 1),
                "p95": round(float(np.percentile(word_arr, 95)), 1),
                "min": int(np.min(word_arr)),
                "max": int(np.max(word_arr)),
            },
        },
        "timestamps": {
            "min_created_at": min_date,
            "max_created_at": max_date,
        },
    }

    # Save to JSON
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[SUCCESS] Inspection report saved to {output_report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Twitter Customer Support dataset.")
    parser.add_argument("--csv-path", type=str, default="data/raw/twcs.csv", help="Path to raw twcs.csv")
    parser.add_argument("--output", type=str, default="results/dataset_inspection_report.json", help="Report output path")
    args = parser.parse_args()

    report = inspect_dataset(Path(args.csv_path), Path(args.output))
    print("\n" + "=" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print(f"Total Rows: {report['file_metadata']['total_rows']:,}")
    print(f"Columns: {', '.join(report['file_metadata']['columns'])}")
    print(f"Inbound (Customer): {report['tweet_types']['inbound_count']:,} ({report['tweet_types']['inbound_percentage']}%)")
    print(f"Outbound (Brand): {report['tweet_types']['outbound_count']:,} ({report['tweet_types']['outbound_percentage']}%)")
    print(f"Thread Starters: {report['thread_linkage']['thread_initiating_tweets_count']:,} ({report['thread_linkage']['thread_initiating_tweets_pct']}%)")
    print(f"Unique Brands Identified: {report['authors']['brand_handles_count']}")
    print("\nTop 10 Brands by Tweet Count:")
    for b in report['authors']['top_25_brands_by_tweet_volume'][:10]:
        print(f"  - {b['brand']}: {b['tweet_count']:,} tweets")
