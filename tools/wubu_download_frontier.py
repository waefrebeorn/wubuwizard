#!/usr/bin/env python3
"""Wave 2 — THE 2026 FRONTIER (user directive: nothing before 2025 unless
extreme quality; 2026 data is the priority).

Kept from wave 1 (already downloaded):
  open-thoughts/OpenThoughts-114k  (2025-08, verified R1 traces = the
  OpenThinker SOTA basis — extreme quality exception, 3.4GB on disk)

2026 frontier (all filtered/high-value):
  Fable-5-Max-Reasoning-250x       0.04G  2026-07-31
  Fable-5-Distill-462x             0.15G  2026-06-15
  GLM-5.2-Conversation             0.49G  2026-07-16
  GLM-5.2-coding-traces            0.04G  2026-07-25
  GPT-5.6-Sol-Luna-Terra           0.30G  2026-07-29
  DeepSeek-v4-Pro-Agent            0.28G  2026-05-22
  DeepSeek-V4-Distill-8000x        0.14G  2026-04-24
  GLM-5.1-Reasoning-1M (SHARD)      ~4G   2026-04-19

DROPPED (2024, not proven extreme quality): WildChat-1M, Magpie-Pro.
DROPPED (2025-02, redundant with our math tier): OpenR1-Math-220k.
"""
import os, time
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from huggingface_hub import snapshot_download, hf_hub_download

BASE = "/home/wubu/models/corpus"
TASKS = [
    ("MoreThought/Fable-5-Max-Reasoning-Filtered-250x", f"{BASE}/reasoning/fable-5-max-250x"),
    ("HelioAI/Fable-5-Distill-Reasoning-462x", f"{BASE}/reasoning/fable-5-distill-462x"),
    ("ianncity/GLM-5.2-Conversation", f"{BASE}/interactions/glm-5.2-conversation"),
    ("greghavens/glm-5.2-coding-and-debugging-traces", f"{BASE}/reasoning/glm-5.2-coding-traces"),
    ("Crownelius/GPT-5.6-Sol-Luna-Terra-Traces", f"{BASE}/reasoning/gpt-5.6-sol-luna-terra"),
    ("TeichAI/DeepSeek-v4-Pro-Agent", f"{BASE}/interactions/deepseek-v4-pro-agent"),
    ("Jackrong/DeepSeek-V4-Distill-8000x", f"{BASE}/reasoning/deepseek-v4-distill-8000x"),
]
PATTERNS = ["*.parquet", "*.json", "*.jsonl", "*.md", "*.csv"]

def main():
    tok = os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN")
    for repo, dst in TASKS:
        print(f"[{time.strftime('%H:%M:%S')}] downloading {repo}", flush=True)
        try:
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                              allow_patterns=PATTERNS, token=tok)
            print(f"[{time.strftime('%H:%M:%S')}] DONE {repo}", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] FAIL {repo}: {e}", flush=True)

    # GLM-5.1-Reasoning-1M-Cleaned: take a ~4GB shard of the parquet
    repo = "Jackrong/GLM-5.1-Reasoning-1M-Cleaned"
    dst = f"{BASE}/reasoning/glm-5.1-reasoning-1m"
    os.makedirs(dst, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] downloading {repo} (4GB shard)", flush=True)
    try:
        from huggingface_hub import list_repo_files
        files = [f for f in list_repo_files(repo, repo_type="dataset", token=tok)
                 if f.endswith(".parquet")]
        files.sort()
        budget = 4.0  # GB
        got = 0.0
        for f in files:
            if got >= budget: break
            # download to the local dir with a stable name
            out = os.path.join(dst, os.path.basename(f))
            if os.path.exists(out) and os.path.getsize(out) > 1000:
                got += os.path.getsize(out) / 1e9
                continue
            hf_hub_download(repo_id=repo, repo_type="dataset", filename=f,
                            local_dir=dst, token=tok)
            got += os.path.getsize(out) / 1e9
            print(f"  {f}: {got:.2f} GB so far", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] DONE {repo} shard ({got:.2f} GB)", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] FAIL {repo}: {e}", flush=True)

if __name__ == "__main__":
    main()
