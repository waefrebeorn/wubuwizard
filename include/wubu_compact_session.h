#ifndef WUBU_COMPACT_SESSION_H
#define WUBU_COMPACT_SESSION_H

/*
 * wubu_compact_session.h -- EG-32b: long-lived session with a command queue
 * and padded-free compact steps. At each step boundary queued requests are
 * folded in, prompt residents are seated before decode residents, and no
 * empty slot is left between them. This models the TensorRT Edge-LLM design
 * (RuntimeSession / RequestEngine / executeCompact), not its GPU runtime.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_COMPACT_MAX_RESIDENTS 32
#define WUBU_COMPACT_MAX_QUEUE    64

enum { WUBU_COMPACT_PROMPT = 0, WUBU_COMPACT_DECODE = 1 };

typedef struct wubu_compact_session wubu_compact_session_t;

wubu_compact_session_t *wubu_compact_session_create(int capacity,
                                                    int allow_compact);
void wubu_compact_session_free(wubu_compact_session_t *p);

int wubu_compact_enqueue(wubu_compact_session_t *p, int request_id,
                         int kind, int tokens);

/* Fold queued commands, admit up to capacity, seat prompt rows first.
 * Returns the number of occupied slots, or -1 when compact is disabled. */
int wubu_compact_step(wubu_compact_session_t *p, int *out, int cap);

int wubu_compact_resident_count(const wubu_compact_session_t *p);
int wubu_compact_pending_count(const wubu_compact_session_t *p);
void wubu_compact_capability(const wubu_compact_session_t *p,
                             int *compact, int *reason);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_COMPACT_SESSION_H */
