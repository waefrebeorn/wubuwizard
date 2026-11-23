/*
 * wubu_titan.c -- EG-67 tiered expert planning. Pure C11.
 */
#include "wubu_titan.h"

int wubu_titan_plan(const wubu_titan_expert_t *experts, int n, size_t cap,
                    size_t *peak, int *streamed) {
    size_t hot = 0, max_stream = 0;
    int nstream = 0;
    if (!experts || !peak || !streamed || n <= 0 || cap == 0) return -1;
    for (int i = 0; i < n; i++) {
        if (experts[i].bytes == 0 || experts[i].tier < 0 ||
            experts[i].tier > 2) return -1;
        if (experts[i].tier == 0) {
            hot += experts[i].bytes;
            if (hot > cap) return -1;
        } else {
            nstream++;
            if (experts[i].bytes > max_stream) max_stream = experts[i].bytes;
        }
    }
    /* One streamed expert at a time; its bytes join the hot set briefly. */
    if (hot + max_stream > cap) return -1;
    *peak = hot + max_stream;
    *streamed = nstream;
    return 0;
}

int wubu_titan_prefix(const int *tokens, const int *tiers, int n,
                      int *hits) {
    int count = 0;
    if (!tokens || !tiers || !hits || n <= 0) return -1;
    for (int i = 0; i < n; i++) {
        int seen = 0;
        for (int j = 0; j < i; j++)
            if (tokens[j] == tokens[i] && tiers[j] == tiers[i]) {
                seen = 1;
                break;
            }
        hits[i] = seen;
        if (seen) count++;
    }
    return count;
}
