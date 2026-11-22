#!/usr/bin/env python3
"""Generate tests/fixtures/tok_meta.gguf — a minimal GGUF v3 carrying the
tokenizer metadata fields EG-02 must expose (pre, token_type, add_bos,
chat_template) alongside the fields the extractor already read.

No weights: n_tensors = 0. This is a metadata-only fixture for the gate.
"""
import struct
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tok_meta.gguf"

u32 = lambda v: struct.pack("<I", v)
i32 = lambda v: struct.pack("<i", v)
u64 = lambda v: struct.pack("<Q", v)


def kv_str(name: str, value: str) -> bytes:
    b = value.encode()
    return u64(len(name)) + name.encode() + i32(8) + u64(len(b)) + b


def kv_u32(name: str, value: int) -> bytes:
    return u64(len(name)) + name.encode() + i32(4) + u32(value)


def kv_bool(name: str, value: bool) -> bytes:
    return u64(len(name)) + name.encode() + i32(7) + bytes([1 if value else 0])


def kv_str_array(name: str, values) -> bytes:
    body = i32(8) + u64(len(values))
    for v in values:
        b = v.encode()
        body += u64(len(b)) + b
    return u64(len(name)) + name.encode() + i32(9) + body


def kv_i32_array(name: str, values) -> bytes:
    body = i32(5) + u64(len(values)) + b"".join(i32(v) for v in values)
    return u64(len(name)) + name.encode() + i32(9) + body


kvs = b"".join([
    kv_str("general.architecture", "llama"),
    kv_str_array("tokenizer.ggml.tokens", ["<unk>", "hello", "world"]),
    kv_str_array("tokenizer.ggml.merges", ["hello world"]),
    kv_str("tokenizer.ggml.pre", "llama3"),
    kv_i32_array("tokenizer.ggml.token_type", [1, 2, 3]),
    kv_bool("tokenizer.ggml.add_bos_token", True),
    kv_str("tokenizer.chat_template", "{% for t in texts %}{{ t }}{% endfor %}"),
    kv_u32("tokenizer.ggml.bos_token_id", 1),
    kv_u32("tokenizer.ggml.eos_token_id", 2),
])

blob = b"GGUF" + u32(3) + u64(0) + u64(9) + kvs
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_bytes(blob)
print(f"wrote {OUT} ({len(blob)} bytes, 9 KVs)")
