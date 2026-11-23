/*
 * wubu_agent_state.c -- EG-39 persistent agent state. Pure C11.
 */
#include "wubu_agent_state.h"

#include "wubu_blob_store.h"

#include <stdio.h>
#include <string.h>

void wubu_agent_store_init(wubu_agent_store_t *s) {
    if (s) memset(s, 0, sizeof(*s));
}

static wubu_agent_slot_t *find_slot(wubu_agent_store_t *s, const char *id) {
    if (!s || !id) return NULL;
    for (int i = 0; i < s->n_agents; i++)
        if (strcmp(s->slots[i].id, id) == 0) return &s->slots[i];
    return NULL;
}

int wubu_agent_state_save(wubu_agent_store_t *s, const char *id,
                          const int *tokens, int n) {
    wubu_agent_slot_t *slot;
    wubu_digest_t d;
    if (!s || !id || !*id || !tokens || n <= 0 ||
        n > WUBU_AGENT_MAX_TOKENS) return -1;

    slot = find_slot(s, id);
    if (!slot) {
        if (s->n_agents >= WUBU_AGENT_MAX) return -1;
        slot = &s->slots[s->n_agents++];
        memset(slot, 0, sizeof(*slot));
        snprintf(slot->id, sizeof(slot->id), "%s", id);
    }
    memcpy(slot->tokens, tokens, (size_t)n * sizeof(int));
    slot->n_tokens = n;
    if (wubu_sha256(tokens, (size_t)n * sizeof(int), &d) != 0) return -1;
    memcpy(slot->digest, d.bytes, sizeof(slot->digest));
    slot->turn++;
    slot->active = 1;
    return 0;
}

int wubu_agent_state_resume(wubu_agent_store_t *s, const char *id,
                            const int *tokens, int n,
                            int *prefill_tokens, int *turn) {
    wubu_agent_slot_t *slot;
    int common = 0;
    if (!s || !id || !tokens || n <= 0 || n > WUBU_AGENT_MAX_TOKENS ||
        !prefill_tokens) return -1;
    slot = find_slot(s, id);
    if (!slot || !slot->active) {
        *prefill_tokens = n;
        if (turn) *turn = 0;
        return 0;
    }
    while (common < slot->n_tokens && common < n &&
           slot->tokens[common] == tokens[common]) common++;
    if (common == n && n == slot->n_tokens) {
        *prefill_tokens = 0;
        if (turn) *turn = slot->turn;
        return 1;
    }
    if (common == slot->n_tokens && common < n) {
        *prefill_tokens = n - common;
        if (turn) *turn = slot->turn;
        if (wubu_agent_state_save(s, id, tokens, n) == 0 && turn)
            *turn = slot->turn;
        return 2;
    }
    *prefill_tokens = n; /* prefix diverged: full re-prefill */
    if (turn) *turn = slot->turn;
    if (wubu_agent_state_save(s, id, tokens, n) == 0 && turn)
        *turn = slot->turn;
    return 2;
}

const wubu_agent_slot_t *wubu_agent_slot(const wubu_agent_store_t *s,
                                         const char *id) {
    if (!s || !id) return NULL;
    for (int i = 0; i < s->n_agents; i++)
        if (strcmp(s->slots[i].id, id) == 0) return &s->slots[i];
    return NULL;
}
