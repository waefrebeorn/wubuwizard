#ifndef WUBU_GRAMMAR_FSM_H
#define WUBU_GRAMMAR_FSM_H

/*
 * wubu_grammar_fsm.h -- EG-11: structured decoding automaton (GBNF trie
 * lineage), reimplemented in C11. Compiles a set of alternative token
 * sequences into a trie, emits a batched allowed-token bitmask, and supports
 * jump-forward once a branch is complete (the grammar stops constraining).
 *
 * This is the automaton/bitmask layer; a JSON-Schema-to-GBNF converter is a
 * separate front-end and is not claimed here.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_grammar_fsm wubu_grammar_fsm_t;
typedef struct {
    int node;      /* -1 means the grammar is complete */
} wubu_grammar_state_t;

wubu_grammar_fsm_t *wubu_grammar_fsm_create(int eos_token);
void wubu_grammar_fsm_free(wubu_grammar_fsm_t *g);

/* Add one alternative (a literal token-id sequence). Compile after all
 * alternatives are added. Returns 0 or -1. */
int wubu_grammar_fsm_add(wubu_grammar_fsm_t *g, const int *tokens, int n);
int wubu_grammar_fsm_compile(wubu_grammar_fsm_t *g);

void wubu_grammar_fsm_state_init(wubu_grammar_state_t *s);

/* Allowed mask for one state: bit i is set when token i may be emitted.
 * Returns 0 or -1 on bad args. */
int wubu_grammar_fsm_mask(const wubu_grammar_fsm_t *g,
                          const wubu_grammar_state_t *s,
                          uint64_t *mask, int mask_words);

/* Batched form for B independent sequences. */
int wubu_grammar_fsm_mask_batch(const wubu_grammar_fsm_t *g,
                                const wubu_grammar_state_t *states, int batch,
                                uint64_t *masks, int mask_words);

/* Advance by a token. Returns 1 when the grammar becomes complete,
 * 0 when still active, -1 when the token is not allowed. */
int wubu_grammar_fsm_advance(const wubu_grammar_fsm_t *g,
                             wubu_grammar_state_t *s, int token);

/* Jump-forward: a completed grammar allows every token. */
int wubu_grammar_fsm_jump_forward(const wubu_grammar_fsm_t *g,
                                   const wubu_grammar_state_t *s,
                                   uint64_t *mask, int mask_words);

int wubu_grammar_fsm_node_count(const wubu_grammar_fsm_t *g);
int wubu_grammar_fsm_complete(const wubu_grammar_state_t *s);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_GRAMMAR_FSM_H */
