"""Generate tiny metadata-only GGUF fixtures for the architecture-registry gate.

Usage: python tools/make_arch_fixtures.py <output-dir>
"""
import os
import struct
import sys


def kv_str(key, value):
    return (struct.pack("<Q", len(key)) + key.encode() +
            struct.pack("<I", 8) + struct.pack("<Q", len(value)) + value.encode())


def kv_i32(key, value):
    return (struct.pack("<Q", len(key)) + key.encode() +
            struct.pack("<I", 4) + struct.pack("<i", value))


def kv_f32(key, value):
    return (struct.pack("<Q", len(key)) + key.encode() +
            struct.pack("<I", 6) + struct.pack("<f", value))


def make(path, arch, prefix=None):
    prefix = prefix or arch
    pairs = [
        kv_str("general.architecture", arch),
        kv_i32(prefix + ".block_count", 32),
        kv_i32(prefix + ".embedding_length", 4096),
        kv_i32(prefix + ".feed_forward_length", 11008),
        kv_i32(prefix + ".attention.head_count", 32),
        kv_i32(prefix + ".attention.head_count_kv", 8),
        kv_i32(prefix + ".rope.dimension_count", 128),
        kv_f32(prefix + ".rope.freq_base", 10000.0),
        kv_i32("vocab_size", 32000),
    ]
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs)) + b"".join(pairs)
    with open(path, "wb") as f:
        f.write(data)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"
    os.makedirs(out, exist_ok=True)
    for name, arch in (("llama", "llama"), ("qwen3moe", "qwen3moe"),
                       ("futurex", "futurex")):
        make(os.path.join(out, name + ".gguf"), arch)
    print("wrote fixtures to", out)


if __name__ == "__main__":
    main()
