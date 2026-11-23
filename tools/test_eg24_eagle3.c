/* test_eg24_eagle3.c -- EG-24 gate: tri-layer fusion + direct prediction. */
#include <stdio.h>
#include <math.h>

#include "wubu_eagle3.h"
#include "wubu_tree_verify.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float h0[4] = { 1.0f, 0.1f, 0.0f, 0.0f };
    float h1[4] = { 0.9f, 0.2f, 0.0f, 0.0f };
    float h2[4] = { 0.4f, 1.2f, 0.0f, 0.0f };
    float emb[16] = {
        1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1
    };
    float fused[4], logits[4], single[4];
    int top, conf_ok;
    float conf;
    wubu_tree_node_t tree[3] = {
        { -1, 0, 1.0f, 0 },
        {  0, 1, 0.0f, 1 },
        {  1, 2, 0.0f, 2 }
    };
    float u[2] = { 0.01f, 0.01f };
    int out[4], n;

    CHECK(wubu_eagle3_fuse(h0, h1, h2, 4, fused) == 0);
    CHECK(wubu_eagle3_predict(fused, emb, 4, 4, logits) == 0);
    top = wubu_eagle3_top1(logits, 4);
    conf = wubu_eagle3_confidence(logits, 4);
    printf("EG-24 fused_top1=%d confidence=%.4f fused=[%.2f %.2f]\n",
           top, conf, fused[0], fused[1]);
    CHECK(top == 1);
    conf_ok = conf > 0.05f && conf < 0.5f;
    CHECK(conf_ok);

    /* A single layer would draft the wrong token. */
    CHECK(wubu_eagle3_predict(h0, emb, 4, 4, single) == 0);
    CHECK(wubu_eagle3_top1(single, 4) == 0);

    /* Feed the fused confidence into the EG-21 tree verifier. */
    tree[1].accept_p = conf;
    tree[2].accept_p = conf;
    n = wubu_tree_verify(tree, 3, u, 2, out, 4);
    printf("EG-24 tree_accepted=%d/2 tokens=[%d %d]\n", n, out[0], out[1]);
    CHECK(n == 2 && out[0] == 1 && out[1] == 2);

    CHECK(wubu_eagle3_fuse(NULL, h1, h2, 4, fused) == -1);
    CHECK(wubu_eagle3_predict(fused, emb, 0, 4, logits) == -1);
    CHECK(wubu_eagle3_top1(NULL, 4) == -1);
    CHECK(wubu_eagle3_confidence(logits, 1) == 0.0f);

    if (failures) {
        fprintf(stderr, "test_eg24_eagle3: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg24_eagle3: PASS\n");
    return 0;
}
