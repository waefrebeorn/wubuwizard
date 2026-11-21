"""Generate a tiny dense-LLaMA GGUF fixture for the generic-runner gate.

Usage: python tools/make_dense_llama_fixture.py <output.gguf>
All tensors are F32 so the fixture stays small and dequantization is a no-op.
"""
import math
import os
import struct
import sys

D = 64
FF = 128
H = 4
KVH = 2
HEAD = D // H
VOCAB = 32
LAYERS = 2


def kv_str(key, value):
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 8) + \
        struct.pack("<Q", len(value)) + value.encode()


def kv_i32(key, value):
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 4) + \
        struct.pack("<i", value)


def kv_f32(key, value):
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<I", 6) + \
        struct.pack("<f", value)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/dense_llama.gguf"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    pairs = [
        kv_str("general.architecture", "llama"),
        kv_i32("llama.block_count", LAYERS),
        kv_i32("llama.embedding_length", D),
        kv_i32("llama.feed_forward_length", FF),
        kv_i32("llama.attention.head_count", H),
        kv_i32("llama.attention.head_count_kv", KVH),
        kv_i32("llama.rope.dimension_count", HEAD),
        kv_f32("llama.rope.freq_base", 10000.0),
        kv_i32("vocab_size", VOCAB),
    ]

    specs = [("token_embd.weight", (D, VOCAB)),
             ("output.weight", (D, VOCAB)),
             ("output_norm.weight", (D,))]
    for layer in range(LAYERS):
        p = "blk.%d." % layer
        specs += [(p + "attn_norm.weight", (D,)),
                  (p + "post_attention_norm.weight", (D,)),
                  (p + "attn_q_norm.weight", (HEAD,)),
                  (p + "attn_k_norm.weight", (HEAD,)),
                  (p + "attn_q.weight", (D, H * HEAD * 2)),
                  (p + "attn_k.weight", (D, KVH * HEAD)),
                  (p + "attn_v.weight", (D, KVH * HEAD)),
                  (p + "attn_output.weight", (H * HEAD * 2, D)),
                  (p + "ffn_gate.weight", (D, FF)),
                  (p + "ffn_up.weight", (D, FF)),
                  (p + "ffn_down.weight", (FF, D))]

    header = b"GGUF" + struct.pack("<IQQ", 3, len(specs), len(pairs)) + b"".join(pairs)
    info = bytearray()
    for name, dims in specs:
        info += struct.pack("<Q", len(name)) + name.encode()
        info += struct.pack("<I", len(dims))
        for d in dims:
            info += struct.pack("<q", d)
        info += struct.pack("<I", 0)  # GGML_TYPE_F32

    prefix = header + info
    blob_off = (len(prefix) + 31) & ~31
    pad = b"\0" * (blob_off - len(prefix))

    # Assign data offsets relative to blob start and build the data buffer.
    data = bytearray()
    offsets = []
    for name, dims in specs:
        offsets.append(len(data))
        n = 1
        for d in dims:
            n *= d
        # Small deterministic values keep activations finite.
        for i in range(n):
            data += struct.pack("<f", math.sin((i + 1) * 0.017) * 0.05)

    infos = bytearray()
    for (name, dims), off in zip(specs, offsets):
        infos += struct.pack("<Q", len(name)) + name.encode()
        infos += struct.pack("<I", len(dims))
        for d in dims:
            infos += struct.pack("<q", d)
        infos += struct.pack("<I", 0)
        infos += struct.pack("<Q", off)

    with open(out, "wb") as f:
        f.write(header + infos + pad + data)
    print("wrote", out, len(specs), "tensors,", len(data), "bytes of F32 data")


if __name__ == "__main__":
    main()
