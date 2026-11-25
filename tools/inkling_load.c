/* inkling_load -- EG-118 step 2 probe: resolve the runtime tensor map and print the census. */
#include <stdio.h>
#include <stdlib.h>
#include "gguf_reader.h"
#include "wubu_inkling2.h"

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : getenv("INKLING_GGUF");
    if (!path) path = "C:/Users/eman5/models/inkling-2-mini/Inkling-2-Mini-UD-IQ2_XXS.gguf";

    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "open failed: %s\n", path); return 1; }

    wubu_inkling2_cfg_t cfg;
    int rc = wubu_inkling2_probe(ctx, &cfg);
    if (rc != WUBU_INKLING2_OK) { fprintf(stderr, "probe rc=%d\n", rc); return 2; }

    static wubu_inkling2_model_t m;   /* ~100 KB of pointers, keep it off the stack */
    rc = wubu_inkling2_load(ctx, &cfg, &m);
    wubu_inkling2_model_print(&m, &cfg);
    printf("load rc = %d (%s)\n", rc, rc == WUBU_INKLING2_OK ? "ok" : "problem");

    /* The loop message queries are the one global whose shape the forward pass depends on. */
    if (m.msg_query.info) {
        printf("loop_res_query dims:");
        for (int d = 0; d < m.msg_query.info->n_dims; ++d) printf(" %lld", (long long)m.msg_query.info->dims[d]);
        printf("  type %d\n", m.msg_query.info->ggml_type);
    }
    if (m.final_res_q.info) {
        printf("loop_kv_res_query dims:");
        for (int d = 0; d < m.final_res_q.info->n_dims; ++d) printf(" %lld", (long long)m.final_res_q.info->dims[d]);
        printf("\n");
    }
    if (m.tok_embd.info) {
        printf("token_embd dims:");
        for (int d = 0; d < m.tok_embd.info->n_dims; ++d) printf(" %lld", (long long)m.tok_embd.info->dims[d]);
        printf("\n");
    }
    gguf_close(ctx);
    return rc == WUBU_INKLING2_OK ? 0 : 3;
}
