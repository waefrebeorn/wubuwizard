#!/usr/bin/env python3
"""
wubu_tokenize_reasoning.py -- tokenize the extracted reasoning/interaction
text into .tok (uint16 token ids) with the C11 BPE, then pack into one
stream per source for the trainer.

Usage:
    python3 tools/wubu_tokenize_reasoning.py

Pipeline: text/openthoughts-114k.txt -> tokens/openthoughts-114k.tok
The C tokenizer (wubu_tokenc) does the encode; this script just drives
it and reports stats.
"""
import os, subprocess, sys, glob

TOKENC = os.path.join(os.path.dirname(__file__), "..", "wubu_tokenc")
TXT = "/home/wubu/models/corpus/text"
TOK = "/home/wubu/models/corpus/tokens"
os.makedirs(TOK, exist_ok=True)

SOURCES = [
    "openthoughts-114k",
    "openthoughts-114k-math",
    "openr1-math-220k",
    "wildchat-1m",
    "magpie-pro-300k",
]

def main():
    tokenc = os.path.realpath(TOKENC)
    tok_js = "/home/wubu/models/wubu/tokenizer.json"
    if not os.path.exists(tokenc):
        print(f"missing {tokenc} — build: make wubu_tokenc")
        return 1
    if not os.path.exists(tok_js):
        print(f"missing tokenizer {tok_js}")
        return 1
    for name in SOURCES:
        src = os.path.join(TXT, f"{name}.txt")
        dst = os.path.join(TOK, f"{name}.tok")
        if not os.path.exists(src):
            print(f"  skip {name}: no {src}")
            continue
        print(f"tokenizing {name} ...", flush=True)
        r = subprocess.run([tokenc, tok_js, src, dst], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FAIL {name}: {r.stderr[-300:]}")
            continue
        nbytes = os.path.getsize(dst)
        ntok = nbytes // 2
        print(f"  DONE {name}: {ntok:,} tokens ({nbytes/1e6:.1f} MB)", flush=True)

if __name__ == "__main__":
    sys.exit(main())
