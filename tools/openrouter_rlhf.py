#!/usr/bin/env python3
"""
openrouter_rlhf.py -- the OpenRouter free-tier RLHF oracle for WuBu.

The user: "here are open router free tier keys to rlhf off of". This
client turns the 6 free keys into the RLHF reward loop:

  1. WuBu drafts an answer (the seed's own generation)
  2. An OpenRouter free model scores the draft (0-100) + critique
  3. The score is the reward; the critique is the SFT target

The R1/Prover pattern: no human labels, a frontier oracle grades the
seed. The 6 keys are used round-robin to spread the free quota.

Free models verified on openrouter.ai (2026-08-03): gemma-4-31b-it,
nemotron-3-super-120b, nemotron-3-ultra-550b (all :free).
"""
import os
import sys
import json
import urllib.request

BASE = "https://openrouter.ai/api/v1"


def _load_keys():
    """Load the 6 OpenRouter keys from the VAULT (secrets/hf.env).
    The user's doctrine: the keys stay in the vault, the tools call the
    keys THROUGH the vault — never hardcoded in the tool."""
    keys = []
    env_path = os.path.expanduser(
        "~/.hermes/profiles/mind-palace/secrets/hf.env")
    if os.path.exists(env_path):
        for line in open(env_path):
            m = __import__("re").match(r'(?:export )?OPENROUTER_KEY_(\d+)="([^"]*)"',
                                       line.strip())
            if m:
                keys.append((int(m.group(1)), m.group(2)))
    # also honor the environment directly (source hf.env before running)
    for i in range(1, 7):
        v = os.environ.get(f"OPENROUTER_KEY_{i}", "")
        if v and v not in [k for _, k in keys]:
            keys.append((i, v))
    keys.sort()
    return [k for _, k in keys]


KEYS = _load_keys()

# verified free models (2026-08-03)
FREE_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "inclusionai/ling-3.0-flash:free",
]
_round_robin = 0


def next_key():
    global _round_robin
    k = KEYS[_round_robin % len(KEYS)]
    _round_robin += 1
    return k


def call_chat(model, messages, max_tokens=512, temperature=0.7):
    """A chat completion through OpenRouter (free tier, key round-robin)."""
    key = next_key()
    body = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.load(r)
        return data["choices"][0]["message"]["content"], None
    except Exception as e:
        return None, str(e)[:160]


def score_draft(draft, prompt, model=None):
    """RLHF oracle: score WuBu's draft 0-100 against the prompt.
    Returns (score, full_out, err)."""
    model = model or FREE_MODELS[0]
    sys_msg = ("You are the RLHF verification oracle for an AGI seed "
               "named WuBu. Score the draft 0-100 for correctness, "
               "coherence, and completeness against the prompt. Reply "
               "with exactly: SCORE:<n> then a one-line critique.")
    out, err = call_chat(model, [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"PROMPT: {prompt}\nDRAFT: {draft}"},
    ], max_tokens=300, temperature=0)
    if err:
        return None, None, err
    score = None
    for line in (out or "").splitlines():
        if "SCORE:" in line.upper():
            try:
                score = int("".join(ch for ch in line.split(":", 1)[1] if ch.isdigit()))
            except Exception:
                pass
    return score, out, None


def rlhf_improve(draft, prompt, model=None):
    """One RLHF iteration: score + critique + an improved draft.
    The critique becomes the SFT target; the score is the reward."""
    model = model or FREE_MODELS[0]
    score, out, err = score_draft(draft, prompt, model)
    if err:
        return None, None, None, err
    improved, err2 = call_chat(model, [
        {"role": "system", "content": "You improve drafts. Given the "
         "prompt, the draft, and the critique, write the corrected "
         "draft. Output ONLY the corrected draft."},
        {"role": "user", "content": f"PROMPT: {prompt}\nDRAFT: {draft}\n"
                                    f"CRITIQUE: {out}"},
    ], max_tokens=400, temperature=0.3)
    return score, out, improved, err2


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        req = urllib.request.Request(f"{BASE}/models",
            headers={"Authorization": f"Bearer {next_key()}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        ids = [d["id"] for d in data.get("data", [])]
        free = [i for i in ids if ":free" in i]
        print(f"OpenRouter: {len(ids)} models, {len(free)} free")
        for f in free[:15]:
            print("  ", f)
    else:
        score, out, improved, err = rlhf_improve(
            "A hive is a chunked linked list with skipfield tombstones.",
            "Explain the hive data structure.")
        if err:
            print("err:", err)
        else:
            print(f"score: {score}\ncritique: {out}\nimproved: {improved}")
