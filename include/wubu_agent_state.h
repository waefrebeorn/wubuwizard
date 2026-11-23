#ifndef WUBU_AGENT_STATE_H
#define WUBU_AGENT_STATE_H

/*
 * wubu_agent_state.h -- EG-39: persistent per-agent conversation state with
 * prefix-digest resume, so an unchanged turn causes no re-prefill
 * (Hibana-style agent state persistence). Pure C11.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_AGENT_MAX 16
#define WUBU_AGENT_MAX_TOKENS 128

typedef struct {
    char id[64];
    int tokens[WUBU_AGENT_MAX_TOKENS];
    int n_tokens;
    unsigned char digest[32];
    int turn;
    int active;
} wubu_agent_slot_t;

typedef struct {
    wubu_agent_slot_t slots[WUBU_AGENT_MAX];
    int n_agents;
} wubu_agent_store_t;

void wubu_agent_store_init(wubu_agent_store_t *s);

/* Save or update a complete conversation prefix for an agent. */
int wubu_agent_state_save(wubu_agent_store_t *s, const char *id,
                          const int *tokens, int n);

/* Resume: prefill_tokens is the number of NEW tokens that must be processed.
 * Returns 1 for exact resume, 2 for prefix extension, 0 for no state. */
int wubu_agent_state_resume(wubu_agent_store_t *s, const char *id,
                            const int *tokens, int n,
                            int *prefill_tokens, int *turn);

const wubu_agent_slot_t *wubu_agent_slot(const wubu_agent_store_t *s,
                                         const char *id);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_AGENT_STATE_H */
