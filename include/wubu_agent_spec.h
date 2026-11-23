#ifndef WUBU_AGENT_SPEC_H
#define WUBU_AGENT_SPEC_H

/*
 * wubu_agent_spec.h -- EG-39c: Hibana-style adaptive agent speculation.
 * Acceptance streaks widen the draft depth; a changed agent turn resets it to
 * one because the prior turn's KV is no longer a valid draft source. The
 * result is clamped by a configured maximum and the remaining token budget.
 * This gates the policy arithmetic, not a measured task-throughput gain.
 */

#ifdef __cplusplus
extern "C" {
#endif

int wubu_agent_spec_depth(int turn, int last_turn, int accepted_streak,
                          int remaining_tokens, int max_depth);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_AGENT_SPEC_H */
