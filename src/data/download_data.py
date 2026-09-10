"""
Dataset download utility for Twitter Customer Support dataset (twcs.csv).
Downloads the raw authentic dataset from Hugging Face mirror to data/raw/twcs.csv.
"""
import argparse
from pathlib import Path
import sys
import urllib.request
import time


DATASET_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
DEFAULT_OUTPUT_PATH = Path("data/raw/twcs.csv")


def download_twcs(output_path: Path = DEFAULT_OUTPUT_PATH, url: str = DATASET_URL) -> Path:
    """Download the twcs.csv dataset with progress reporting."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 500_000_000:
        print(f"[INFO] Dataset already exists at {output_path} ({output_path.stat().st_size:,} bytes). Skipping download.")
        return output_path

    print(f"[INFO] Initiating download from: {url}")
    print(f"[INFO] Target destination: {output_path}")

    start_time = time.time()
    last_print = start_time
    total_bytes = 0

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) HiverAssignment/1.0"}
        )
        with urllib.request.urlopen(req) as response, open(output_path, "wb") as out_file:
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length else None
            
            chunk_size = 1024 * 1024  # 1 MB chunks
            downloaded = 0

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                
                now = time.time()
                if now - last_print >= 5 or (total_size and downloaded >= total_size):
                    elapsed = max(0.1, now - start_time)
                    speed_mb = (downloaded / (1024 * 1024)) / elapsed
                    if total_size:
                        pct = (downloaded / total_size) * 100
                        print(f"  Progress: {downloaded / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({pct:.1f}%) | Speed: {speed_mb:.2f} MB/s")
                    else:
                        print(f"  Progress: {downloaded / (1024 * 1024):.1f} MB downloaded | Speed: {speed_mb:.2f} MB/s")
                    last_print = now

        elapsed = time.time() - start_time
        print(f"[SUCCESS] Download completed in {elapsed:.1f} seconds. File size: {output_path.stat().st_size:,} bytes.")
        return output_path

    except Exception as e:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        print(f"[ERROR] Failed to download dataset: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Twitter Customer Support dataset.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Path to save twcs.csv")
    parser.add_argument("--url", type=str, default=DATASET_URL, help="URL to download twcs.csv from")
    args = parser.parse_args()

    download_twcs(Path(args.output), args.url)
