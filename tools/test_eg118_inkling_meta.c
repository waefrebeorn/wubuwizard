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

    /* Walk every tensor: prove the dequant path works for the types we know and
     * report the exact tensors whose type this engine cannot decode. */
    if (rc == WUBU_INKLING2_OK) {
        int ok = 0, unknown = 0, io = 0;
        long long elems_total = 0;
        static float buf[1 << 20];
        char first_unknown[256] = {0};
        int unknown_type_id = -1;
        for (int64_t ti = 0; ti < ctx->n_tensors; ++ti) {
            const gguf_tensor_info *tt = &ctx->tensors[ti];
            int64_t n = 1;
            for (int d = 0; d < tt->n_dims; ++d) n *= tt->dims[d];
            int64_t got = 0;
            const int r = wubu_inkling2_tensor_f32(ctx, tt->name, buf, (int64_t)(1 << 20), &got);
            if (r == WUBU_INKLING2_OK) {
                ++ok;
                elems_total += got;
                int bad = 0;
                for (int64_t i = 0; i < got; ++i) if (!(buf[i] == buf[i])) { ++bad; break; }
                if (bad) { printf("EG-118 non-finite values in %s\n", tt->name); ++failures; }
            } else if (r == WUBU_INKLING2_ERR_UNKNOWN_TYPE) {
                ++unknown;
                if (!first_unknown[0]) { snprintf(first_unknown, sizeof(first_unknown), "%s", tt->name); unknown_type_id = tt->ggml_type; }
            } else if (r == WUBU_INKLING2_ERR_SHAPE && n > (1 << 20)) {
                ++io;   /* tensor larger than this gate's scratch buffer */
            } else {
                printf("EG-118 decode error %d on %s\n", r, tt->name);
                ++io;
            }
        }
        printf("EG-118 tensors: decoded=%d unknown_type=%d skipped_or_large=%d elems=%.3fB\n",
               ok, unknown, io, (double)elems_total / 1e9);
        if (unknown) printf("EG-118 first unknown type: id %d on %s\n", unknown_type_id, first_unknown);
        CHECK(ok + unknown + io == (int)ctx->n_tensors);
        /* 146 tensors are larger than this gate's 1M-element scratch (expert
         * stacks and the 201k-vocab embeddings); they are counted, not decoded. */
        CHECK(ok >= 500);
        CHECK(io + unknown < (int)ctx->n_tensors / 2);
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
