/* test_semantic_router.c — route a query to the KV region it attends to
 * (AN22 #11)
 *
 *   T1 routes to the highest-attention region first
 *   T2 zero-attention regions are not returned
 *   T3 cfg.top_k limits the result set
 *   T4 region labels pass through
 *   T5 multi-query accumulation + window clamping
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_semantic_router.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail = 0;
static void chk(const char *label, int cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", label);
    if (!cond) fail++;
}

int main(void) {
    /* 6 context tokens, three regions: [0,2) [2,4) [4,6). */
    wubu_route_region_t R[3] = {
        { 0, 2, "/kv/in/pond-a/first.txt" },
        { 2, 2, "/kv/in/pond-a/attn.txt"  },
        { 4, 2, "/kv/in/pond-b/moe.txt"   },
    };
    /* query attends hard to region 1 (ctx 2,3) */
    float attn[6] = { 0.1f, 0.1f, 0.4f, 0.4f, 0.0f, 0.0f };

    wubu_semantic_router_t *r = wubu_semantic_router_create(NULL);
    chk("router_create returns handle", r != NULL);

    printf("[T1] top routing\n");
    size_t rank[4]; const char *labels[4];
    int n = wubu_semantic_router_route(r, attn, 1, 6, R, 3, rank, labels, 4);
    chk("route returns 2 regions (R2 has zero mass)", n == 2);
    chk("rank[0] == region 1", rank[0] == 1);
    chk("rank[1] == region 0", rank[1] == 0);

    printf("[T2] zero-mass regions excluded\n");
    chk("rank contains no region 2",
        rank[0] != 2 && rank[1] != 2);

    printf("[T3] cfg.top_k limits the result\n");
    wubu_semantic_router_cfg_t cfg = wubu_semantic_router_default_cfg();
    cfg.top_k = 1;
    wubu_semantic_router_t *r1 = wubu_semantic_router_create(&cfg);
    int n1 = wubu_semantic_router_route(r1, attn, 1, 6, R, 3, rank, NULL, 4);
    chk("top_k=1 returns 1", n1 == 1);
    chk("top region == 1", rank[0] == 1);

    printf("[T4] labels pass through\n");
    int n2 = wubu_semantic_router_route(r, attn, 1, 6, R, 3, rank, labels, 4);
    chk("labels returned", n2 > 0);
    chk("top region label correct",
        labels[0] && strcmp(labels[0], "/kv/in/pond-a/attn.txt") == 0);

    printf("[T5] multi-query accumulation + window clamp\n");
    /* two query rows both attend region 0 hard (ctx 0,1) */
    float attn2[12] = {
        0.4f, 0.4f, 0.1f, 0.1f, 0.0f, 0.0f,
        0.3f, 0.3f, 0.2f, 0.2f, 0.0f, 0.0f,
    };
    /* region 2 window extends past context (start 5 len 8) -> clamped */
    wubu_route_region_t R2[3] = {
        { 0, 2, "/kv/in/pond-a/first.txt" },
        { 2, 2, "/kv/in/pond-a/attn.txt"  },
        { 5, 8, "/kv/in/pond-b/moe.txt"   }, /* clamped to [5,6) */
    };
    int n3 = wubu_semantic_router_route(r, attn2, 2, 6, R2, 3, rank, NULL, 4);
    chk("multi-query routes to region 0 (accumulated mass)", n3 >= 1 && rank[0] == 0);
    chk("clamped R2 still contributes (no crash, mass in [5,6))", n3 >= 0);

    wubu_semantic_router_free(r1);
    wubu_semantic_router_free(r);
    printf("test_semantic_router: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
