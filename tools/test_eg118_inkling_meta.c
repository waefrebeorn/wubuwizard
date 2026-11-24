/* test_eg118_inkling_meta.c -- EG-118 gate: Inkling-2 GGUF probe + tensor map.
 *
 * Skips when no Inkling GGUF is on disk; otherwise asserts every value the probe
 * returns against the file's own metadata and validates the full tensor map.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; } } while (0)

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : getenv("INKLING_GGUF");
    if (!path) path = "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";

    FILE *probe = fopen(path, "rb");
    if (!probe) { printf("test_eg118: SKIP (no %s)\n", path); return 0; }
    fclose(probe);

    gguf_ctx *ctx = gguf_open(path);
    CHECK(ctx != NULL);
    if (!ctx) return 1;

    wubu_inkling2_cfg_t cfg;
    const int rc = wubu_inkling2_probe(ctx, &cfg);
    CHECK(rc == WUBU_INKLING2_OK);
    if (rc == WUBU_INKLING2_OK) {
        printf("EG-118 inkling2: L=%d d=%d ff=%d experts=%d(top-%d) loops=%d core=[%d,%d) rank=%d\n",
               cfg.n_layer, cfg.d_model, cfg.d_ff, cfg.n_expert, cfg.n_expert_used,
               cfg.n_loops, cfg.loop_core_start, cfg.loop_core_end, cfg.loop_gate_rank);
        printf("EG-118 inkling2: heads=%d kv_heads=%d head_dim=%d swa=%d conv=%d res_block=%d\n",
               cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.swa_window, cfg.conv_lcache,
               cfg.attn_res_block);
        printf("EG-118 inkling2: vocab=%d/%d rope_dim=%d situ=%.1f/%.1f softcap=%.1f\n",
               cfg.vocab_size, cfg.vocab_size_unpadded, cfg.rope_dim, cfg.situ_beta,
               cfg.situ_linear_beta, cfg.logit_softcap);

        /* Values measured from this file (docs/research/inkling2-gguf-inventory.md). */
        CHECK(cfg.n_layer == 48);
        CHECK(cfg.d_model == 2304);
        CHECK(cfg.d_ff == 1152);
        CHECK(cfg.n_expert == 128);
        CHECK(cfg.n_expert_used == 8);
        CHECK(cfg.n_loops == 3);
        CHECK(cfg.loop_core_start == 6 && cfg.loop_core_end == 18);
        CHECK(cfg.loop_gate_rank == 64);
        CHECK(cfg.head_dim == 128);
        CHECK(cfg.n_head == 24);
        CHECK(cfg.n_kv_head == 8);
        CHECK(cfg.swa_window == 1024);
        CHECK(cfg.conv_lcache == 4);
        CHECK(cfg.attn_res_block == 6);
        CHECK(cfg.vocab_size == 201024);
        CHECK(cfg.vocab_size_unpadded == 200058);
        CHECK(cfg.rope_dim == 0);
        CHECK(cfg.logit_softcap == 30.0f);

        /* 48 logical layers are 24 stored blocks (core 6..17 runs 3 times). */
        CHECK(cfg.n_layer_phys == 24);
        CHECK(wubu_inkling2_phys_block(&cfg, 0) == 0);
        CHECK(wubu_inkling2_phys_block(&cfg, 5) == 5);
        CHECK(wubu_inkling2_phys_block(&cfg, 6) == 6);
        CHECK(wubu_inkling2_phys_block(&cfg, 17) == 17);
        CHECK(wubu_inkling2_phys_block(&cfg, 18) == 6);    /* 2nd pass starts at the core */
        CHECK(wubu_inkling2_phys_block(&cfg, 30) == 6);    /* 3rd pass */
        CHECK(wubu_inkling2_phys_block(&cfg, 41) == 17);
        CHECK(wubu_inkling2_phys_block(&cfg, 42) == 18);   /* back layers */
        CHECK(wubu_inkling2_phys_block(&cfg, 47) == 23);
        CHECK(wubu_inkling2_phys_block(&cfg, 48) == -1);

        const int problems = wubu_inkling2_validate(ctx, &cfg);
        printf("EG-118 inkling2: tensor map problems=%d\n", problems);
        CHECK(problems == 0);
    }

    /* A gemma file must be rejected, not silently accepted. */
    {
        gguf_ctx *g = gguf_open("C:/Users/eman5/models/gemma-12b/gemma-4-12b-it-UD-IQ2_M.gguf");
        if (g) {
            wubu_inkling2_cfg_t tmp;
            CHECK(wubu_inkling2_probe(g, &tmp) == WUBU_INKLING2_ERR_ARCH);
            gguf_close(g);
        }
    }

    gguf_close(ctx);
    if (failures) { fprintf(stderr, "test_eg118_inkling_meta: FAIL (%d)\n", failures); return 1; }
    printf("test_eg118_inkling_meta: PASS\n");
    return 0;
}
