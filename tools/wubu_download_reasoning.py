#!/usr/bin/env python3
"""Download the best reasoning-trace + interaction datasets for WuBu training.

Target ~10GB of QUALITY data (2026-08-09):
  1. OpenThoughts-114k       3.55GB  DeepSeek-R1 verified reasoning traces
  2. WildChat-1M             3.36GB  real user interactions with GPT-4
  3. OpenR1-Math-220k        3.0GB   (shard) math reasoning traces
  4. Magpie-Pro-300K-Filtered 0.56GB high-quality alignment
  5. OpenThoughts-114k-math  0.98GB  verified math CoT
= ~10.4GB

Downloads land in /home/wubu/models/corpus/{reasoning,interactions}/.
"""
import os, sys, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from huggingface_hub import snapshot_download

BASE = "/home/wubu/models/corpus"
TASKS = [
    # (repo, target dir, allow_patterns)
    ("open-thoughts/OpenThoughts-114k",   f"{BASE}/reasoning/openthoughts-114k",
     ["*.parquet", "*.json", "*.jsonl", "*.md", "*.txt"]),
    ("allenai/WildChat-1M",               f"{BASE}/interactions/wildchat-1m",
     ["*.parquet", "*.json", "*.jsonl", "*.md"]),
    ("open-r1/OpenR1-Math-220k",          f"{BASE}/reasoning/openr1-math-220k",
     ["*train-00000*.parquet", "*train-00001*.parquet", "*train-00002*.parquet",
      "*.md"]),   # shard ~3GB
    ("Magpie-Align/Magpie-Pro-300K-Filtered", f"{BASE}/interactions/magpie-pro-300k",
     ["*.parquet", "*.json", "*.jsonl", "*.md"]),
    ("open-r1/OpenThoughts-114k-math",    f"{BASE}/reasoning/openthoughts-114k-math",
     ["*.parquet", "*.json", "*.jsonl", "*.md"]),
]

def main():
    for repo, dst, patterns in TASKS:
        print(f"[{time.strftime('%H:%M:%S')}] downloading {repo} -> {dst}", flush=True)
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=dst,
                allow_patterns=patterns,
                token=os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN"),
            )
            print(f"[{time.strftime('%H:%M:%S')}] DONE {repo}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] FAIL {repo}: {e}", flush=True)

if __name__ == "__main__":
    main()
