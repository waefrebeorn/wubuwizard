#!/usr/bin/env python3
"""
qwen35_parity_fixture.py — capture the llama.cpp reference logits for
the Qwen3.5 logit-parity harness (the AN28 parity step).

Usage: ./tools/qwen35_parity_fixture.py "The capital of France is" \
       --n_predict 3 --n_probs 5 --port 8090 > /tmp/ref.json

The wizard's full-stack forward (once the hybrid loader runs) must
match the TOP-1 token per position (the MiniCPM5-style verification);
the TOP-5 logprobs are the richer oracle. The reference is
deterministic at temperature 0.
"""
import argparse, json, sys, urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--n_predict", type=int, default=3)
    ap.add_argument("--n_probs", type=int, default=5)
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    body = json.dumps({
        "prompt": args.prompt,
        "n_predict": args.n_predict,
        "temperature": 0,
        "n_probs": args.n_probs,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)

    choice = resp["choices"][0]
    per_token = []
    for t in choice.get("logprobs", {}).get("content", []):
        per_token.append({
            "position": len(per_token) + 1,
            "token": t.get("token"),
            "top5": [{"token": p.get("token"),
                      "logprob": round(p.get("logprob", 0), 4)}
                     for p in t.get("top_logprobs", [])[:5]],
        })

    fixture = {
        "model": resp.get("model"),
        "prompt": args.prompt,
        "temperature": 0,
        "reference_generation": choice.get("text"),
        "top5_per_token": per_token,
    }
    print(json.dumps(fixture, indent=2))

if __name__ == "__main__":
    main()
