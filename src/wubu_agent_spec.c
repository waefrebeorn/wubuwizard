/* wubu_agent_spec.c -- EG-39c adaptive agent speculation gate. Strict C11. */
#include "wubu_agent_spec.h"

int wubu_agent_spec_depth(int turn, int last_turn, int accepted_streak,
                          int remaining_tokens, int max_depth)
{
    int depth;
    if (turn < 1 || last_turn < 0 || accepted_streak < 0 ||
        remaining_tokens < 0 || max_depth < 1) return -1;
    if (turn != last_turn) return 1;
    depth = 1;
    if (accepted_streak >= 2) ++depth;
    if (accepted_streak >= 4) ++depth;
    if (depth > max_depth) depth = max_depth;
    if (depth > remaining_tokens) depth = remaining_tokens;
    return depth;
}
