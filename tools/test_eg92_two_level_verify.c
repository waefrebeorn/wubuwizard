/* test_eg92_two_level_verify.c -- EG-92 gate: quantized pre-verify + exact commit. */
#include <stdio.h>

#include "wubu_two_level_verify.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_tlv_t *p = wubu_tlv_create();
    int draft[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    int q_match[8] = {1, 1, 1, 0, 1, 1, 0, 1};
    int chain[8];
    int len;
    CHECK(p != NULL);

    /* Quantized view greedily accepts three tokens but commits nothing. */
    len = wubu_tlv_q_verify(p, draft, q_match, 8, chain);
    CHECK(len == 3);
    CHECK(chain[0] == 10 && chain[1] == 20 && chain[2] == 30);
    CHECK(wubu_tlv_chain_len(p) == 3);
    CHECK(wubu_tlv_committed(p) == 0);
    printf("EG-92 q-step chain_len=%d committed=%d (no commit)\n",
           wubu_tlv_chain_len(p), wubu_tlv_committed(p));
    wubu_tlv_free(p);

    /* Exact pass commits chain + accepted subtree in one pass. */
    {
        wubu_tlv_t *q = wubu_tlv_create();
        int chain_exact[3] = {1, 1, 1};
        int sub_parent[3] = {-1, 0, 1};
        int sub_ids[3] = {40, 50, 60};
        int sub_match[3] = {1, 1, 0};
        int out[32], n;
        CHECK(q != NULL);
        CHECK(wubu_tlv_q_verify(q, draft, q_match, 8, chain) == 3);
        n = wubu_tlv_exact_commit(q, chain_exact, sub_parent, sub_ids,
                                  sub_match, 3, 8, out);
        CHECK(n == 5);
        CHECK(wubu_tlv_committed(q) == 5);
        CHECK(out[0] == 10 && out[1] == 20 && out[2] == 30 &&
              out[3] == 40 && out[4] == 50);
        printf("EG-92 exact commit=%d ids=[%d %d %d %d %d]\n", n,
               out[0], out[1], out[2], out[3], out[4]);
        wubu_tlv_free(q);
    }

    /* An optimistic quantized chain is lost if the exact pass disagrees. */
    {
        wubu_tlv_t *q = wubu_tlv_create();
        int chain_exact[3] = {1, 0, 1};
        int sub_parent[3] = {-1, 0, 1};
        int sub_ids[3] = {40, 50, 60};
        int sub_match[3] = {1, 1, 1};
        int out[32];
        CHECK(q != NULL);
        CHECK(wubu_tlv_q_verify(q, draft, q_match, 8, chain) == 3);
        CHECK(wubu_tlv_exact_commit(q, chain_exact, sub_parent, sub_ids,
                                    sub_match, 3, 8, out) == 0);
        CHECK(wubu_tlv_committed(q) == 0);
        printf("EG-92 exact chain rejection commits 0 tokens\n");
        wubu_tlv_free(q);
    }

    /* Budget truncates the subtree before the exact pass. */
    {
        wubu_tlv_t *q = wubu_tlv_create();
        int chain_exact[3] = {1, 1, 1};
        int sub_parent[3] = {-1, 0, 1};
        int sub_ids[3] = {40, 50, 60};
        int sub_match[3] = {1, 1, 1};
        int out[32];
        CHECK(q != NULL);
        CHECK(wubu_tlv_q_verify(q, draft, q_match, 8, chain) == 3);
        CHECK(wubu_tlv_exact_commit(q, chain_exact, sub_parent, sub_ids,
                                    sub_match, 3, 4, out) == 4);
        CHECK(wubu_tlv_exact_commit(NULL, chain_exact, sub_parent, sub_ids,
                                    sub_match, 3, 4, out) == -1);
        CHECK(wubu_tlv_exact_commit(q, NULL, sub_parent, sub_ids,
                                    sub_match, 3, 4, out) == -1);
        printf("EG-92 budget=4 truncates subtree to 1 node, commit=4\n");
        wubu_tlv_free(q);
    }

    if (failures) { fprintf(stderr, "test_eg92_two_level_verify: FAIL (%d)\n", failures); return 1; }
    printf("test_eg92_two_level_verify: PASS\n");
    return 0;
}
