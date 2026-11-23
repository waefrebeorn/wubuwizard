/* wubu_adaptive_cap.c -- EG-66b adaptive cap gate. Strict C11. */
#include "wubu_adaptive_cap.h"

int wubu_adaptive_cap_update(int current, int accepted, int drafted,
                             int kv_free_slots, int min_cap, int max_cap)
{
    int candidate;
    if (drafted < 1 || accepted < 0 || accepted > drafted ||
        kv_free_slots < 1 || min_cap < 1 || max_cap < min_cap ||
        current < min_cap || current > max_cap) return -1;
    if (accepted * 5 >= drafted * 4) candidate = current + 1;
    else if (accepted * 5 <= drafted) candidate = current - 1;
    else candidate = current;
    if (candidate < min_cap) candidate = min_cap;
    if (candidate > max_cap) candidate = max_cap;
    if (candidate > kv_free_slots) candidate = kv_free_slots;
    return candidate;
}
