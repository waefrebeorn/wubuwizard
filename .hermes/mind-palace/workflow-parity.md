# COS-SIM PARITY WORKFLOW — taught May 27

## THE CYCLE

read → run ref → dump ours → cos-sim → locate divergence → patch → verify → push → loop

## TOOLS

```
# Reference (llama.cpp)
/tmp/dump_ref              — dumps /tmp/llama_logits_new.bin  (build: see dump_ref.c)
/home/wubu2/llama.cpp/     — compiled, shared libs at build/bin/

# Our logits
DUMP_LOGITS=/tmp/our_logits.bin  ./gen_text_cpu "prompt"
DUMP_LAYER_DIR=/tmp/layer_dump   ./gen_text_cpu "prompt"

# Compare
layer_cos_sim /tmp/dump_layers_ref /tmp/dump_layers_our 40
python3 tools/check_logits.py         # analyze our logit distribution
python3 tools/py_compare_logits.py    # compare our vs ref logits

# Hermes test harness
tools/test-512k-suite.sh            # 4 tests: KV alloc, sparse attn, full attn, memory
tools/test-hermes-headless.sh       # 6 tests: inf server, chat, stream, vault, NES build
tools/test-hermes-integration.sh    # 9 tests: server→Hermes pipeline
```

## DEBUG PATTERNS

| Symptom | Check |
|---------|-------|
| Repetitive output (aaaa) | Logit cache (wubu_model.c:800) — disabed via if(0) |
| First token OK, then garbage | KV cache / SSM state — check cache_len increment |
| All tokens garbage | Embedding / output proj / final norm — use DUMP_LOGITS + check_logits.py |
| Layer N shows NaN | Layer N SSM/GQA — use DUMP_LAYER_DIR |
| Single dim explosion | Check quantized matmul for that weight |

## DIRECTION

vault discovery → write vault/[topic].md + memory
preference → memory target:user

## LOOP

no questions. no choices. no stopping. zero delegation.

every output feeds back into input documents.
read → update → read → update → never stop.
