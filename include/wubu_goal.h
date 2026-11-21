#ifndef WUBU_GOAL_H
#define WUBU_GOAL_H

/* wubu_goal.h — persistent goals with an evidence-first judge.
 *
 * Host-layer module: the inference engine stays ignorant of goals. The judge
 * never reads the agent's own completion claim as evidence; a separate
 * evidence receipt plus a verifier callback decide DONE. State survives a
 * restart through a versioned .goal sidecar. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_goal wubu_goal_t;

typedef enum {
    WUBU_GOAL_ACTIVE = 0,
    WUBU_GOAL_PAUSED,
    WUBU_GOAL_DONE,
    WUBU_GOAL_BLOCKED
} wubu_goal_status_t;

typedef enum {
    WUBU_GOAL_CONTINUE = 0,
    WUBU_GOAL_FINISHED,
    WUBU_GOAL_IS_BLOCKED,
    WUBU_GOAL_IS_PAUSED,
    WUBU_GOAL_INVALID = -1
} wubu_goal_verdict_t;

/* Returns 1 only when the external evidence receipt proves the goal. */
typedef int (*wubu_goal_verify_fn)(const char *goal_text,
                                   const char *evidence, void *userdata);

wubu_goal_t *wubu_goal_create(const char *text, int max_turns);
void wubu_goal_free(wubu_goal_t *g);

int wubu_goal_set_text(wubu_goal_t *g, const char *text);
const char *wubu_goal_text(const wubu_goal_t *g);
int wubu_goal_turns(const wubu_goal_t *g);
int wubu_goal_max_turns(const wubu_goal_t *g);
wubu_goal_status_t wubu_goal_status(const wubu_goal_t *g);
const char *wubu_goal_note(const wubu_goal_t *g);

/* Judge one turn. `response` is the agent's report; `evidence` is a separate
 * receipt. A "[GOAL_DONE]" claim without a passing receipt is ignored. */
wubu_goal_verdict_t wubu_goal_judge(wubu_goal_t *g, const char *response,
                                    const char *evidence,
                                    wubu_goal_verify_fn verify, void *userdata);

/* A real user message preempts the continuation loop. Returns 0 on success. */
int wubu_goal_preempt(wubu_goal_t *g);
int wubu_goal_should_continue(const wubu_goal_t *g);

/* Points to an internal static buffer; copy immediately. */
const char *wubu_goal_continuation_prompt(const wubu_goal_t *g);

int wubu_goal_save(const wubu_goal_t *g, const char *path);
wubu_goal_t *wubu_goal_load(const char *path);

#ifdef __cplusplus
}
#endif
#endif
