/* test_eg11_grammar.c -- EG-11 gate: grammar trie automaton, batched token
 * masks, invalid-token rejection and jump-forward on completion.
 * Prints: test_eg11_grammar: PASS. */
#include <stdio.h>

#include "wubu_grammar_fsm.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define BIT(w, t) (((w)[(t) / 64] >> ((t) % 64)) & 1ULL)

int main(void) {
    const int alt_a[1] = { 10 };
    const int alt_b[2] = { 20, 30 };
    wubu_grammar_state_t s, s2, states[2];
    uint64_t mask[2];

    wubu_grammar_fsm_t *g = wubu_grammar_fsm_create(4 /* EOS */);
    CHECK(g != NULL);
    if (!g) return 1;
    CHECK(wubu_grammar_fsm_add(g, alt_a, 1) == 0);
    CHECK(wubu_grammar_fsm_add(g, alt_b, 2) == 0);
    CHECK(wubu_grammar_fsm_compile(g) == 0);
    printf("EG-11 trie nodes=%d\n", wubu_grammar_fsm_node_count(g));
    CHECK(wubu_grammar_fsm_node_count(g) == 4);

    wubu_grammar_fsm_state_init(&s);
    CHECK(wubu_grammar_fsm_mask(g, &s, mask, 1) == 0);
    CHECK(BIT(mask, 10) && BIT(mask, 20));
    CHECK(!BIT(mask, 4));  /* EOS not legal at the root */
    CHECK(!BIT(mask, 30));
    printf("EG-11 root mask: token10=%d token20=%d token30=%d eos=%d\n",
           (int)BIT(mask, 10), (int)BIT(mask, 20),
           (int)BIT(mask, 30), (int)BIT(mask, 4));

    /* Complete the single-token alternative, then jump forward. */
    CHECK(wubu_grammar_fsm_advance(g, &s, 10) == 1);
    CHECK(wubu_grammar_fsm_complete(&s));
    CHECK(wubu_grammar_fsm_jump_forward(g, &s, mask, 1) == 1);
    CHECK(mask[0] == ~(uint64_t)0);
    printf("EG-11 jump-forward: all bits set\n");

    /* Two-token alternative: intermediate state allows only 30. */
    wubu_grammar_fsm_state_init(&s2);
    CHECK(wubu_grammar_fsm_advance(g, &s2, 20) == 0);
    CHECK(!wubu_grammar_fsm_complete(&s2));
    CHECK(wubu_grammar_fsm_mask(g, &s2, mask, 1) == 0);
    CHECK(BIT(mask, 30));
    CHECK(!BIT(mask, 10));
    CHECK(!BIT(mask, 4));
    CHECK(wubu_grammar_fsm_advance(g, &s2, 30) == 1);
    CHECK(wubu_grammar_fsm_complete(&s2));

    /* Invalid token is rejected and does not move the state. */
    wubu_grammar_fsm_state_init(&s);
    CHECK(wubu_grammar_fsm_advance(g, &s, 99) == -1);
    CHECK(!wubu_grammar_fsm_complete(&s));
    CHECK(wubu_grammar_fsm_mask(g, &s, mask, 1) == 0);
    CHECK(BIT(mask, 10) && BIT(mask, 20));

    /* Batched masks for two independent sequences. */
    wubu_grammar_fsm_state_init(&states[0]);
    wubu_grammar_fsm_state_init(&states[1]);
    CHECK(wubu_grammar_fsm_advance(g, &states[1], 20) == 0);
    uint64_t batched[2];
    CHECK(wubu_grammar_fsm_mask_batch(g, states, 2, batched, 1) == 0);
    CHECK(BIT(batched, 10) && BIT(batched, 20));
    CHECK(!BIT(batched, 30));
    CHECK(BIT(batched + 1, 30) && !BIT(batched + 1, 10));
    printf("EG-11 batch: seq0 allows 10/20, seq1 allows 30\n");

    /* Fail closed. */
    wubu_grammar_fsm_t *extra = wubu_grammar_fsm_create(4);
    CHECK(extra != NULL);
    wubu_grammar_fsm_free(extra);
    CHECK(wubu_grammar_fsm_add(g, alt_a, 1) == -1); /* after compile */
    CHECK(wubu_grammar_fsm_add(NULL, alt_a, 1) == -1);
    CHECK(wubu_grammar_fsm_compile(NULL) == -1);
    CHECK(wubu_grammar_fsm_mask(g, NULL, mask, 1) == -1);
    CHECK(wubu_grammar_fsm_mask_batch(g, states, 2, NULL, 1) == -1);
    CHECK(wubu_grammar_fsm_advance(g, NULL, 10) == -1);

    wubu_grammar_fsm_free(g);
    if (failures) {
        fprintf(stderr, "test_eg11_grammar: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg11_grammar: PASS\n");
    return 0;
}
