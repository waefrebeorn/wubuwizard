#!/usr/bin/env python3
"""gguf_detok.py -- decode token ids with a GGUF's own tokenizer.ggml.tokens list.

The engine has no detokeniser (only gguf_tokenizer_token_count), so generated ids have to be read
somewhere. This walks the KV section, pulls the token list, and maps the ids given on the command line.
Gemma-family vocabularies mark a leading space with '\u0120' and use '<0xNN>' escape tokens, both of which
are expanded here so the output reads as text.
"""
import struct, sys

def load_tokens(path):
    f = open(path, "rb")
    f.read(4); struct.unpack("<I", f.read(4))[0]
    nt, nkv = struct.unpack("<QQ", f.read(16))
    def rs():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    def rv(t):
        if t in (0, 1): return struct.unpack("<B", f.read(1))[0]
        if t in (2, 3): return struct.unpack("<H", f.read(2))[0]
        if t in (4, 5): return struct.unpack("<I", f.read(4))[0]
        if t == 6: return struct.unpack("<f", f.read(4))[0]
        if t == 7: return struct.unpack("<B", f.read(1))[0]
        if t == 8: return rs()
        if t in (10, 11): return struct.unpack("<Q", f.read(8))[0]
        if t == 12: return struct.unpack("<d", f.read(8))[0]
        if t == 9:
            at, an = struct.unpack("<IQ", f.read(12))
            return [rv(at) for _ in range(an)]
        raise ValueError(f"kv type {t}")
    toks = None
    for _ in range(nkv):
        k = rs(); t = struct.unpack("<I", f.read(4))[0]
        v = rv(t)
        if k == "tokenizer.ggml.tokens" and isinstance(v, list):
            toks = v
    f.close()
    return toks

def render(s):
    s = s.replace("\u0120", " ")
    if len(s) == 6 and s.startswith("<0x") and s.endswith(">"):
        try: return bytes([int(s[3:5], 16)]).decode("latin-1")
        except ValueError: pass
    return s

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: gguf_detok.py <model.gguf> <id> [id ...]"); raise SystemExit(2)
    toks = load_tokens(sys.argv[1])
    ids = [int(x) for x in sys.argv[2:]]
    print("".join(render(toks[i]) for i in ids if 0 <= i < len(toks)))
    print("ids:", ids)
