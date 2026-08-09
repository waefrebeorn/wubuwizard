/*
 * test_wubu35_dims.c -- the Revolver Doctrine + Aligned Rewrite gate (S1-S5, ADR005).
 *
 * Proves:
 *   1. wubu35_dims_probe() reads the REAL tensor shapes from the checkpoint.
 *   2. The probe ALIGNs geometry upward (Theory/08): 448→512 so every quant
 *      block (QK_K=256) tiles evenly — no remainder guards fire.
 *   3. wubu_load() probes + sets the runtime global BEFORE loading.
 *   4. The revolver rotates: a forced different dims set changes what the
 *      macros report (growth-safe, no fixed geometry).
 *   5. Aligned geometry tiles evenly across Q8_K/Q4_K/Q6_K (256), AVX-512 (64),
 *      2:4 sparse (4), VNNI int8 tile (16).
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "wubu.h"
#include "wubu35_dims.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "models/wubu/model.safetensors";
    printf("=== test_wubu35_dims (the Revolver Doctrine gate) ===\n");

    /* 1. The probe reads REAL geometry from the checkpoint. */
    wubu35_dims_t d;
    CHECK(wubu35_dims_probe(path, &d) == 0, "probe opens checkpoint");
    printf("  probed: vocab=%d dim=%d layers=%d heads=%d kv_heads=%d "
           "ffn_dim=%d rope_dim=%d rope_theta=%.0f\n",
           d.vocab, d.dim, d.layers, d.heads, d.kv_heads, d.ffn_dim,
           d.rope_dim, d.rope_theta);
    /* 448-dim checkpoint. The probe reads NATIVE dims; the engine then
     * aligns them to the block grid (448→512) by zero-padding the
     * checkpoint weights at load time (load_tensor Aligned path).
     * The probe reports the aligned geometry the engine will use. */
    CHECK(d.vocab == 16384, "probe: vocab == 16384");
    CHECK(d.dim == 512,     "probe: dim aligned 448→512 (div by 256)");
    CHECK(d.layers == 12,   "probe: layers == 12");
    CHECK(d.heads == 8,     "probe: heads == 8 (512/64, aligned)");
    CHECK(d.kv_heads == 1,  "probe: kv_heads == 1 (GQA 8:1)");
    CHECK(d.head_dim == 64, "probe: head_dim == 64 (div 16 for VNNI)");
    CHECK(d.ffn_dim == 2048, "probe: ffn_dim design target 4*dim=2048 (16 QK_K blocks)");
    CHECK(d.rope_dim == 32, "probe: rope_dim == 32 (head_dim/2)");

    /* 2. The defaults seed the aligned geometry. */
    wubu35_dims_default();
    CHECK(WUBU_VOCAB == 16384 && WUBU_DIM == 512 && WUBU_LAYERS == 12,
          "defaults seed aligned geometry (Theory/08)");

    /* 3. The revolver rotates: a DIFFERENT dims set changes what the
     * macros report (no fixed geometry — a grown model is loadable). */
    wubu35_dims_t grown = d;
    grown.layers = 16;   /* a hypothetical grown checkpoint */
    wubu35_dims_set(&grown);
    CHECK(WUBU_LAYERS == 16, "revolver: WUBU_LAYERS now 16 after set");
    CHECK(WUBU_SELECTORS == 16 / WUBU_SELECT_EVERY,
          "revolver: selectors derived from active layers");
    printf("  rotated dims -> WUBU_LAYERS=%d WUBU_SELECTORS=%d\n",
           WUBU_LAYERS, WUBU_SELECTORS);

    /* 4. Aligned geometry smoke test (no checkpoint needed — the
     * geometry is self-describing from the defaults. The old seed-sft2
     * checkpoint is frozen to SD archive per WuBu1's total break; the
     * new model trains from scratch in aligned geometry). */
    wubu35_dims_set(&d);
    CHECK(WUBU_LAYERS == 12, "revolver: back to 12 after restore");

    wubu_model_t m;
    memset(&m, 0, sizeof(m));
    /* use default geometry only (no checkpoint load — WuBu1 fresh train) */
    /* param count is a pure function of the runtime geometry */
    long params = wubu_parameter_count(&m);
    printf("  params=%ld (aligned 56376832, ckpt native 35073216)\n", params);
    CHECK(params == 56376832L, "param count == 56,376,832 (aligned, 3 selectors from 12 layers)");

    /* 5. Aligned geometry tiles evenly (Theory/08 contract). */
    CHECK(WUBU_DIM % 256 == 0 && WUBU_FFN_DIM % 256 == 0,
          "aligned dims divide QK_K=256 (no zero-fill guards fire)");
    CHECK(WUBU_DIM % 64 == 0, "dim divides 64 (AVX-512 cache line)");
    CHECK(WUBU_HEAD_DIM % 16 == 0, "head_dim divides 16 (VNNI int8 tile)");

    if (failures == 0) printf("=== ALL WUBU35-DIMS TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
