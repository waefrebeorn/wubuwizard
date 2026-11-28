"""Rewrite a GGUF so its DECLARED block count matches the blocks it actually contains.

Why: Inkling-2-Mini-UD-IQ2_XXS.gguf declares inkling2.block_count = 48 while every per-layer array carries 48 entries and
the file stores tensors only for blk.0..blk.23. Upstream llama.cpp trusts the declaration, walks to blk.24 and aborts
("check_tensor_dims: tensor 'blk.24.attn_q.weight' not found"); patching only the scalar then trips the array-length check
("inkling2.attention.head_count has wrong array length; expected 24, got 48"). This tool truncates block_count and every
per-layer array of that length to the real block count, re-serialises the metadata, and copies the tensor data blob
verbatim - tensor offsets are relative to the data blob, and the tensor list is unchanged, so no offset arithmetic is
needed beyond re-padding the header to the file's alignment.

Run:  python tools/gguf_reblock.py <src.gguf> <dst.gguf> [n_blocks]
"""
import struct
import sys
import shutil

SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
          6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    f = open(src, "rb")
    magic = f.read(4)
    assert magic == b"GGUF", magic
    ver = struct.unpack("<I", f.read(4))[0]
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    kv = []          # (key, type, value)  value: bytes for scalars/strings, (etype, [raw elem bytes]) for arrays
    for _ in range(n_kv):
        key = read_str(f).decode("utf-8")
        t = struct.unpack("<I", f.read(4))[0]
        if t == 8:
            kv.append((key, t, read_str(f)))
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            cnt = struct.unpack("<Q", f.read(8))[0]
            if et == 8:
                elems = [read_str(f) for _ in range(cnt)]
            else:
                fmt, sz = SCALAR[et]
                elems = [f.read(sz) for _ in range(cnt)]
            kv.append((key, t, (et, elems)))
        else:
            fmt, sz = SCALAR[t]
            kv.append((key, t, f.read(sz)))

    # tensor infos are kept byte-identical: same names, dims, types and data-blob offsets
    info_start = f.tell()
    infos = []
    for _ in range(n_tensors):
        name = read_str(f)
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        ty = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, dims, ty, off))
    old_data_start = f.tell()

    # alignment from metadata (default 32)
    align = 32
    for key, t, val in kv:
        if key == "general.alignment" and t == 4:
            align = struct.unpack("<I", val)[0]

    real_blocks = 0
    for name, _, _, _ in infos:
        if name.startswith(b"blk."):
            real_blocks = max(real_blocks, int(name.split(b".")[1]) + 1)
    want = int(sys.argv[3]) if len(sys.argv) > 3 else real_blocks
    arch_override = sys.argv[4].encode() if len(sys.argv) > 4 else None
    print(f"declared vs real blocks: {[v for k, t, v in kv if k.endswith('block_count')]}  real={real_blocks}  target={want}")
    assert want <= real_blocks, "cannot declare more blocks than the file contains"

    # ---- rewrite metadata: block_count -> want, per-layer arrays of the old declared length -> want
    old_decl = None
    for key, t, val in kv:
        if key.endswith("block_count") and t == 4:
            old_decl = struct.unpack("<I", val)[0]
    assert old_decl is not None, "no block_count key found"
    out = []
    changed = []
    for key, t, val in kv:
        if arch_override and key.startswith("inkling2."):
            newkey = "inkling2_loop." + key[len("inkling2."):]
            out.append((newkey, t, val))
            changed.append((key, key, newkey))
            continue
        if key == "general.architecture" and t == 8 and arch_override:
            out.append((key, t, arch_override))
            changed.append((key, val, arch_override))
        elif key.endswith("block_count") and t == 4:
            out.append((key, t, struct.pack("<I", want)))
            changed.append((key, old_decl, want))
        elif t == 9 and val[0] != 8 and len(val[1]) == old_decl:
            out.append((key, t, (val[0], val[1][:want])))
            changed.append((key, old_decl, want))
        else:
            out.append((key, t, val))
    for c in changed:
        print(f"  {c[0]}: {c[1]} -> {c[2]}")

    # ---- serialise
    with open(dst, "wb") as g:
        g.write(b"GGUF")
        g.write(struct.pack("<I", ver))
        g.write(struct.pack("<Q", n_tensors))
        g.write(struct.pack("<Q", len(out)))
        for key, t, val in out:
            kb = key.encode("utf-8")
            g.write(struct.pack("<Q", len(kb)))
            g.write(kb)
            g.write(struct.pack("<I", t))
            if t == 8:
                g.write(struct.pack("<Q", len(val)))
                g.write(val)
            elif t == 9:
                et, elems = val
                g.write(struct.pack("<I", et))
                g.write(struct.pack("<Q", len(elems)))
                for e in elems:
                    if et == 8:
                        g.write(struct.pack("<Q", len(e)))
                        g.write(e)
                    else:
                        g.write(e)
            else:
                g.write(val)

        for name, dims, ty, off in infos:
            g.write(struct.pack("<Q", len(name)))
            g.write(name)
            g.write(struct.pack("<I", len(dims)))
            for d in dims:
                g.write(struct.pack("<Q", d))
            g.write(struct.pack("<I", ty))
            g.write(struct.pack("<Q", off))

        new_data_start = g.tell()
        pad = (-new_data_start) % align
        g.write(b"\x00" * pad)
        new_data_start = g.tell()

        f.seek(old_data_start)
        shutil.copyfileobj(f, g, 64 << 20)

    print(f"old data_start={old_data_start}  new data_start={new_data_start}  "
          f"delta={new_data_start - old_data_start}  (tensor offsets are blob-relative, unchanged)")
    f.close()


if __name__ == "__main__":
    main()
