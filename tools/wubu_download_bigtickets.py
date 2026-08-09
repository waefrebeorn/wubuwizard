#!/usr/bin/env python3
"""Wave 3 — THE BIG TICKETS (25GB target):
  1. open-thoughts/OpenThoughts3-1.2M   (QwQ-32B teacher, SOTA recipe)  ~10GB shard
  2. Jackrong/GLM-5.1-Reasoning-1M-Cleaned (2026-04)                    ~8GB shard
Runs AFTER wave 2 (frontier smalls) — sequential, resume-safe.
"""
import os, time, sys
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from huggingface_hub import hf_hub_download, list_repo_files

BASE = "/home/wubu/models/corpus"
TOK = os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN")

def shard_download(repo, dst, budget_gb):
    os.makedirs(dst, exist_ok=True)
    files = sorted(f for f in list_repo_files(repo, repo_type="dataset", token=TOK)
                   if f.endswith((".parquet", ".jsonl")))
    got = 0.0
    n = 0
    for f in files:
        if got >= budget_gb: break
        # hf_hub_download preserves repo subdirs (e.g. data/xxx.parquet)
        out = os.path.join(dst, f)
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            got += os.path.getsize(out) / 1e9
            continue
        try:
            hf_hub_download(repo_id=repo, repo_type="dataset", filename=f,
                            local_dir=dst, token=TOK)
            if os.path.exists(out):
                got += os.path.getsize(out) / 1e9
            n += 1
            if n % 5 == 0:
                print(f"  {n} shards, {got:.2f} GB", flush=True)
        except Exception as e:
            print(f"  shard fail {f}: {e}", flush=True)
    print(f"  DONE {repo}: {n} shards, {got:.2f} GB", flush=True)

def main():
    shard_download("open-thoughts/OpenThoughts3-1.2M",
                   f"{BASE}/reasoning/openthoughts3-1.2m", 10.0)
    shard_download("Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
                   f"{BASE}/reasoning/glm-5.1-reasoning-1m", 8.0)

if __name__ == "__main__":
    main()
