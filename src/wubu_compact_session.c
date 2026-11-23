/* wubu_compact_session.c -- EG-32b compact session gate. Strict C11. */
#include "wubu_compact_session.h"

#include <stdlib.h>

typedef struct { int id, kind, tokens; } cmd_t;

struct wubu_compact_session {
    int capacity, allow_compact, n, qn, qhead;
    cmd_t resident[WUBU_COMPACT_MAX_RESIDENTS];
    cmd_t queue[WUBU_COMPACT_MAX_QUEUE];
};

wubu_compact_session_t *wubu_compact_session_create(int capacity,
                                                    int allow_compact)
{
    wubu_compact_session_t *p;
    if (capacity < 1 || capacity > WUBU_COMPACT_MAX_RESIDENTS ||
        allow_compact < 0 || allow_compact > 1) return NULL;
    p = (wubu_compact_session_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->capacity = capacity;
    p->allow_compact = allow_compact;
    return p;
}

void wubu_compact_session_free(wubu_compact_session_t *p) { free(p); }

int wubu_compact_enqueue(wubu_compact_session_t *p, int request_id,
                         int kind, int tokens)
{
    if (!p || request_id < 0 || tokens < 1 ||
        (kind != WUBU_COMPACT_PROMPT && kind != WUBU_COMPACT_DECODE)) return -1;
    for (int i = p->qhead; i < p->qn; ++i)
        if (p->queue[i].id == request_id) return -1;
    for (int i = 0; i < p->n; ++i)
        if (p->resident[i].id == request_id) return -1;
    if (p->qn >= WUBU_COMPACT_MAX_QUEUE) return -1;
    p->queue[p->qn].id = request_id;
    p->queue[p->qn].kind = kind;
    p->queue[p->qn].tokens = tokens;
    ++p->qn;
    return 0;
}

int wubu_compact_step(wubu_compact_session_t *p, int *out, int cap)
{
    int k = 0;
    if (!p || !out || cap < p->capacity || !p->allow_compact) return -1;
    for (int i = p->qhead; i < p->qn; ++i) {
        if (p->n == p->capacity) break;
        p->resident[p->n++] = p->queue[i];
        ++p->qhead;
    }
    if (p->qhead == p->qn) { p->qhead = 0; p->qn = 0; }
    for (int i = 0; i < p->n; ++i)
        if (p->resident[i].kind == WUBU_COMPACT_PROMPT) out[k++] = p->resident[i].id;
    for (int i = 0; i < p->n; ++i)
        if (p->resident[i].kind == WUBU_COMPACT_DECODE) out[k++] = p->resident[i].id;
    return k;
}

int wubu_compact_resident_count(const wubu_compact_session_t *p)
{
    return p ? p->n : -1;
}

int wubu_compact_pending_count(const wubu_compact_session_t *p)
{
    return p ? p->qn - p->qhead : -1;
}

void wubu_compact_capability(const wubu_compact_session_t *p,
                             int *compact, int *reason)
{
    if (!compact || !reason) return;
    *compact = p ? p->allow_compact : 0;
    *reason = p ? (p->allow_compact ? 0 : 1) : 2;
}
