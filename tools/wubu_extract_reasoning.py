#!/usr/bin/env python3
"""
wubu_extract_reasoning.py -- extract QUALITY text from the reasoning +
interaction datasets into corpus/text/ for the C11 tokenizer.

Schema-aware per dataset (the "best ways to wire it in"):
  - OpenThoughts-114k:  problem + deepseek_reasoning + deepseek_solution
                        -> keep the FULL trace (reasoning chain + answer)
  - OpenThoughts-114k-math: same schema (problem/reasoning/solution)
  - OpenR1-Math-220k:    problem + solution (deepseek-r1 style)
  - WildChat-1M:         multi-turn conversation (real user interactions)
  - Magpie-Pro-300K:     instruction + response (single-turn alignment)

Output: one doc per line-ish paragraph into <name>.txt; the C11
tokenizer (wubu_tokenc) converts to .tok. Deterministic shuffle.
"""
import os, sys, json, glob, random

import pyarrow.parquet as pq

TXT = "/home/wubu/models/corpus/text"
os.makedirs(TXT, exist_ok=True)
random.seed(48)

SPECIAL = {
    "<|im_start|>": "\n[USER]\n", "<|im_end|>": "\n", "<|assistant|>": "\n[ASSISTANT]\n",
    "<|endoftext|>": "\n", "<|start_of_turn|>": "\n[USER]\n", "<|end_of_turn|>": "\n",
    "<|begin_of_text|>": "", "<|user|>": "\n[USER]\n", "<|assistant|>": "\n[ASSISTANT]\n",
}

def clean(s):
    if s is None: return ""
    if not isinstance(s, str): s = str(s)
    for k, v in SPECIAL.items():
        s = s.replace(k, v)
    return s.strip()

def extract_open_thoughts(path, out, n_out):
    """problem + deepseek_reasoning + deepseek_solution (full CoT)."""
    t = pq.read_table(path)
    col_r = "deepseek_reasoning" if "deepseek_reasoning" in t.column_names else None
    col_s = "deepseek_solution" if "deepseek_solution" in t.column_names else None
    col_p = "problem" if "problem" in t.column_names else t.column_names[0]
    for i in range(t.num_rows):
        p = clean(t.column(col_p)[i].as_py())
        if len(p) < 8: continue
        out.write("[PROBLEM]\n" + p + "\n")
        if col_r:
            r = clean(t.column(col_r)[i].as_py())
            if len(r) > 8:
                out.write("[REASONING]\n" + r + "\n")
        if col_s:
            s = clean(t.column(col_s)[i].as_py())
            if len(s) > 4:
                out.write("[ANSWER]\n" + s + "\n")
        out.write("\n")
        n_out[0] += 1
        if n_out[0] % 5000 == 0:
            print(f"    {n_out[0]} docs", flush=True)

def extract_conversation(path, out, n_out):
    """WildChat-style multi-turn: messages[] or turns[]."""
    t = pq.read_table(path)
    col = None
    for c in ("messages", "conversation", "turns", "chat"):
        if c in t.column_names: col = c; break
    if col is None: col = t.column_names[0]
    for i in range(t.num_rows):
        v = t.column(col)[i].as_py()
        if not v: continue
        doc = []
        for m in v:
            if not isinstance(m, dict): continue
            role = str(m.get("role", "")).lower()
            content = clean(m.get("content", m.get("value", "")))
            if not content: continue
            who = "[USER]\n" if "user" in role else "[ASSISTANT]\n"
            doc.append(who + content)
        if len(doc) < 2: continue
        out.write("\n".join(doc) + "\n\n")
        n_out[0] += 1
        if n_out[0] % 10000 == 0:
            print(f"    {n_out[0]} docs", flush=True)

def extract_pair(path, out, n_out, q_col="problem", a_col="solution"):
    """problem/solution pair (OpenR1-Math, Magpie instruction/response)."""
    t = pq.read_table(path)
    cols = t.column_names
    q = q_col if q_col in cols else ("instruction" if "instruction" in cols else cols[0])
    a = a_col if a_col in cols else ("response" if "response" in cols else None)
    for i in range(t.num_rows):
        p = clean(t.column(q)[i].as_py())
        if len(p) < 8: continue
        out.write("[PROBLEM]\n" + p + "\n")
        if a and a in cols:
            s = clean(t.column(a)[i].as_py())
            if len(s) > 4:
                out.write("[ANSWER]\n" + s + "\n")
        out.write("\n")
        n_out[0] += 1
        if n_out[0] % 10000 == 0:
            print(f"    {n_out[0]} docs", flush=True)

def extract_row_json(path, out, n_out):
    """Crownelius-verified-library format: row_json holds the conversation."""
    import json as _json
    t = pq.read_table(path)
    col = "row_json" if "row_json" in t.column_names else t.column_names[0]
    for i in range(t.num_rows):
        v = t.column(col)[i].as_py()
        if not v: continue
        try:
            obj = _json.loads(v) if isinstance(v, str) else v
        except Exception:
            continue
        doc = []
        msgs = obj.get("messages") or obj.get("conversations") or []
        for m in msgs:
            if not isinstance(m, dict): continue
            role = str(m.get("role", m.get("from", ""))).lower()
            content = clean(m.get("content", m.get("value", "")))
            if not content: continue
            who = "[USER]\n" if ("user" in role or role == "human") else "[ASSISTANT]\n"
            doc.append(who + content)
        if len(doc) < 1: continue
        out.write("\n".join(doc) + "\n\n")
        n_out[0] += 1
        if n_out[0] % 10000 == 0:
            print(f"    {n_out[0]} docs", flush=True)

def extract_jsonl(path, out, n_out, q_col="query", a_cols=None):
    """Fable-5-style jsonl: query + reasoning/answer fields."""
    import json as _json
    a_cols = a_cols or ["reasoning", "response", "answer", "completion", "output"]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = _json.loads(line)
            except Exception:
                continue
            q = clean(obj.get(q_col, ""))
            if len(q) < 8: continue
            out.write("[PROBLEM]\n" + q + "\n")
            for ac in a_cols:
                a = clean(obj.get(ac, ""))
                if len(a) > 4:
                    out.write("[ANSWER]\n" + a + "\n")
            out.write("\n")
            n_out[0] += 1
            if n_out[0] % 5000 == 0:
                print(f"    {n_out[0]} docs", flush=True)

def main():
    # (glob, extractor, outname) — recursive: HF datasets land in
    # data/ and metadata/ subdirs
    jobs = [
        ("/home/wubu/models/corpus/reasoning/openthoughts-114k/**/*.parquet",
         extract_open_thoughts, "openthoughts-114k"),
        ("/home/wubu/models/corpus/reasoning/openthoughts-114k-math/**/*.parquet",
         extract_open_thoughts, "openthoughts-114k-math"),
        ("/home/wubu/models/corpus/reasoning/openr1-math-220k/**/*.parquet",
         lambda p, o, n: extract_pair(p, o, n), "openr1-math-220k"),
        ("/home/wubu/models/corpus/interactions/wildchat-1m/**/*.parquet",
         extract_conversation, "wildchat-1m"),
        ("/home/wubu/models/corpus/interactions/magpie-pro-300k/**/*.parquet",
         lambda p, o, n: extract_pair(p, o, n, "instruction", "response"),
         "magpie-pro-300k"),
        # 2026 frontier
        ("/home/wubu/models/corpus/reasoning/fable-5-max-250x/**/*",
         extract_jsonl, "fable-5-max-250x"),
        ("/home/wubu/models/corpus/reasoning/fable-5-distill-462x/**/*",
         extract_jsonl, "fable-5-distill-462x"),
        ("/home/wubu/models/corpus/reasoning/glm-5.2-coding-traces/**/*.parquet",
         extract_conversation, "glm-5.2-coding"),
        ("/home/wubu/models/corpus/reasoning/gpt-5.6-sol-luna-terra/**/*.parquet",
         extract_row_json, "gpt-5.6-sol-luna-terra"),
        ("/home/wubu/models/corpus/reasoning/deepseek-v4-distill-8000x/**/*.parquet",
         extract_open_thoughts, "deepseek-v4-distill"),
        ("/home/wubu/models/corpus/reasoning/glm-5.1-reasoning-1m/**/*.parquet",
         extract_conversation, "glm-5.1-reasoning"),
        ("/home/wubu/models/corpus/reasoning/openthoughts3-1.2m/**/*.parquet",
         extract_open_thoughts, "openthoughts3-1.2m"),
        ("/home/wubu/models/corpus/interactions/glm-5.2-conversation/**/*.parquet",
         extract_conversation, "glm-5.2-conversation"),
        ("/home/wubu/models/corpus/interactions/deepseek-v4-pro-agent/**/*.jsonl",
         extract_jsonl, "deepseek-v4-pro-agent"),
    ]
    for pat, fn, name in jobs:
        files = sorted(glob.glob(pat))
        if not files:
            print(f"  skip {name}: no files at {pat}", flush=True)
            continue
        dst = os.path.join(TXT, f"{name}.txt")
        n_out = [0]
        print(f"extracting {name}: {len(files)} shards -> {dst}", flush=True)
        with open(dst, "w") as out:
            for f in files:
                try:
                    fn(f, out, n_out)
                except Exception as e:
                    print(f"    shard fail {os.path.basename(f)}: {e}", flush=True)
        print(f"  DONE {name}: {n_out[0]} docs, {os.path.getsize(dst)/1e6:.1f} MB", flush=True)

if __name__ == "__main__":
    main()
