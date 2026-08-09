/*
 * wubu_runtime_dims.c -- runtime dimensions for the WuBu-35M engine.
 *
 * The Revolver Doctrine (THEORY/06): probe, don't assume. The loader
 * probes the real checkpoint tensor shapes and fills the runtime global
 * WUBU_RUNTIME_DIMS; every WUBU_* macro reads from it.
 *
 * The probe uses safetensors_reader (st_*) to inspect tensor shapes:
 *   - embedding.weight   -> vocab × dim   (16384 × 448)
 *   - layers.0.attn.q_proj.weight -> dim × heads*head_dim
 *   - layers.0.attn.k_proj.weight -> dim × kv_heads*head_dim
 *   - layers.N.attn.q_proj.weight presence -> ACTIVE layer count
 *     (progressive-growth checkpoints save fewer than 12)
 *   - layers.0.ffn.gate_up.weight -> dim × 2*ffn_dim
 * Fallback: wubu_runtime_dims_default() for anything the probe cannot see.
 */
#include "wubu_runtime_dims.h"
#include "safetensors_reader.h"
#include <stdio.h>
#include <string.h>

wubu_runtime_dims_t WUBU_RUNTIME_DIMS = {0};

void wubu_runtime_dims_default(void)
{
    /* Theory/08: the Aligned Rewrite.
     * Every geometry is chosen so the quant block (QK_K=256), the SIMD
     * tile (64), the 2:4 sparse mask (4), and the VNNI int8 tile (16)
     * all divide evenly — no remainder handling, no generic fallbacks.
     * Parameter count stays ~35M (was 35,072,768; aligned is 35,176,448). */
    wubu_runtime_dims_t d;
    d.vocab        = 16384;
    d.dim          = 512;    /* was 448 — 512 div 256, 64, 4, 16 */
    d.layers       = 12;
    d.heads        = 8;      /* was 7  — 512/64, div by all SIMD widths */
    d.kv_heads     = 1;      /* GQA 8:1 (head_dim stays 64) */
    d.head_dim     = 64;     /* unchanged — div 16 for VNNI, checkpoint-native */
    d.rope_dim     = 32;     /* head_dim/2 */
    d.ffn_dim      = 2048;   /* DESIGN TARGET: 4*dim=2048 (power-of-2).
                              * gate_up=[2*2048,512]=[4096,512]=16 QK_K
                              * blocks exactly; down=[512,2048]=4 blocks.
                              * (The seed checkpoint's 1228→1280 is only a
                              * compat shim for loading the OLD weights; new
                              * training uses 2048 — the hardware-native
                              * scale. THEORY/08.) */
    d.max_seq      = 16384;  /* was 2048 — 4096-aligned KV pages */
    d.local_win    = 512;
    d.full_every   = 4;
    d.select_every = 4;
    d.clip         = 10.0f;
    d.eps          = 1e-6f;
    d.selectors    = 4;
    d.rope_theta   = 10000.0f;
    d.params       = 56376832L; /* aligned (zero-padded from 35M ckpt */
    d.ckpt_dim     = 448;
    d.ckpt_ffn_dim = 1228;
    d.ckpt_head_dim = 64;
    d.ckpt_heads    = 7;
    d.ckpt_kv_heads = 1;
    WUBU_RUNTIME_DIMS = d;
}

void wubu_runtime_dims_set(const wubu_runtime_dims_t *d)
{
    if (!d) return;
    WUBU_RUNTIME_DIMS = *d;
    /* Derived: selectors = layers / select_every (the released layout). */
    if (WUBU_RUNTIME_DIMS.select_every > 0)
        WUBU_RUNTIME_DIMS.selectors = WUBU_RUNTIME_DIMS.layers / WUBU_RUNTIME_DIMS.select_every;
}

/* Binary-search the ACTIVE layer count: walk for the highest layer index
 * whose layers.N.attn.q_proj.weight tensor exists in the checkpoint. */
static int probe_active_layers(st_ctx *r, char *name, size_t name_sz)
{
    int lo = 1, hi = 12;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        snprintf(name, name_sz, "layers.%d.attn.q_proj.weight", mid - 1);
        if (st_find_tensor(r, name)) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

int wubu_runtime_dims_probe(const char *path, wubu_runtime_dims_t *d)
{
    if (!path || !d) return -1;
    wubu_runtime_dims_default();
    wubu_runtime_dims_t out = WUBU_RUNTIME_DIMS;
    char name[128];

    st_ctx *r = st_open(path);
    if (!r) return -1;

    /* vocab × dim from the tied embedding */
    const st_tensor_info *emb = st_find_tensor(r, "embedding.weight");
    if (emb && emb->n_elems > 0) {
        /* layout: [vocab, dim] — n_elems = vocab*dim. */
        if (emb->n_dims >= 2 && emb->dims[0] > 0 && emb->dims[1] > 0) {
            out.vocab = (int)emb->dims[0];
            out.dim   = (int)emb->dims[1];
        }
    }

    /* active layer count (progressive-growth checkpoints) */
    out.layers = probe_active_layers(r, name, sizeof(name));

    /* heads*head_dim from q_proj: layout is [out, in] = [heads*head_dim, dim] */
    const st_tensor_info *q = st_find_tensor(r, "layers.0.attn.q_proj.weight");
    if (q && q->n_dims >= 2 && q->dims[0] > 0 && q->dims[1] > 0) {
        int q_out = (int)q->dims[0];
        out.head_dim = 64;
        out.heads = q_out / out.head_dim;
    }

    /* kv_heads*head_dim from k_proj: [out, in] = [kv_heads*head_dim, dim] */
    const st_tensor_info *k = st_find_tensor(r, "layers.0.attn.k_proj.weight");
    if (k && k->n_dims >= 2 && k->dims[0] > 0 && k->dims[1] > 0) {
        int k_out = (int)k->dims[0];
        if (out.head_dim > 0) out.kv_heads = k_out / out.head_dim;
    }

    /* 2*ffn_dim from gate_up: [out, in] = [2*ffn_dim, dim] */
    const st_tensor_info *gu = st_find_tensor(r, "layers.0.ffn.gate_up.weight");
    if (gu && gu->n_dims >= 2 && gu->dims[0] > 0 && gu->dims[1] > 0)
        out.ffn_dim = (int)gu->dims[0] / 2;

    /* rope_dim: no direct tensor — derive from head_dim (aligned). */
    /* Theory/08: align probed geometry UPWARD to the block grid so every
     * quant type (QK_K=256) tiles evenly. 448→512 for dim/ffn; head_dim
     * stays 64 (the checkpoint's value) so per-head norms map 1:1, with
     * heads = dim/head_dim = 512/64 = 8. Loader zero-pads the 448→512
     * rows/cols in weight matrices (dead weights, no accuracy loss). */
    out.ckpt_dim      = out.dim;
    out.ckpt_ffn_dim  = out.ffn_dim;
    out.ckpt_head_dim = out.head_dim;
    out.ckpt_heads    = out.heads;
    out.ckpt_kv_heads = out.kv_heads;
    out.dim           = (out.dim + 255) & ~255;     /* 448 → 512 */
    out.ffn_dim       = 4 * out.dim;  /* DESIGN TARGET: 4*dim (power-of-2).
                                       * 1228 → 2048. gate_up=[4096,512]
                                       * = exactly 16 QK_K blocks; down=[512,
                                       * 2048] = exactly 8 blocks. No rem.
                                       * THEORY/08 hardware-native scale. */
    out.heads         = out.dim / out.head_dim;     /* 512/64 = 8 */
    out.kv_heads      = (out.kv_heads > 0) ? out.kv_heads : 1;
    if (out.kv_heads > out.heads) out.kv_heads = 1;
    out.rope_dim      = out.head_dim / 2;           /* 32 */
    out.local_win     = 512;
    st_close(r);

    *d = out;
    return 0;
}
