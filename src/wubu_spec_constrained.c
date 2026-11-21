/* wubu_spec_constrained.c -- LP-09: masked multi-row window for constrained
 * decoding. Strict C11, opaque struct, status-as-value, no third-party deps.
 * See include/wubu_spec_constrained.h.
 *
 * Grammar (two branches at the decision point):
 *   { "tool" : "<enabled-name>" , "args" : <num> }
 *   { "tool" : "<enabled-name>" , "text" : <str> }
 * The ST_BRANCH_VAL mask admits exactly TOK_ARGS and TOK_TEXT. Each row mask
 * is derived from the same transition function, so masks and legal tokens can
 * never disagree. */
#include "wubu_spec_constrained.h"
#include <stdlib.h>
#include <string.h>

enum {
    ST_START = 0, ST_KEY_TOOL, ST_TOOL_STR, ST_TOOL_CLOSE, ST_COLON1,
    ST_NAME_OPEN, ST_NAME_VAL, ST_NAME_CLOSE, ST_COMMA, ST_BRANCH_OPEN,
    ST_BRANCH_VAL, ST_ARGS_COLON, ST_ARGS_VAL, ST_TEXT_COLON, ST_TEXT_VAL,
    ST_CLOSE, ST_DONE
};

struct wubu_spec_constrained {
    uint32_t enabled;
    int state;
};

static const int tool_tok[WUBU_TOOL_COUNT] = {
    WUBU_JTOK_SEARCH, WUBU_JTOK_CALC, WUBU_JTOK_WEATHER, WUBU_JTOK_CODE
};

/* Next state on a legal transition, or -1 when the token is illegal here. */
static int step(const wubu_spec_constrained_t *c, int state, int tok)
{
    switch (state) {
        case ST_START:       return tok == WUBU_JTOK_LBRACE ? ST_KEY_TOOL : -1;
        case ST_KEY_TOOL:    return tok == WUBU_JTOK_QUOTE   ? ST_TOOL_STR : -1;
        case ST_TOOL_STR:    return tok == WUBU_JTOK_TOOL    ? ST_TOOL_CLOSE : -1;
        case ST_TOOL_CLOSE:  return tok == WUBU_JTOK_QUOTE   ? ST_COLON1 : -1;
        case ST_COLON1:      return tok == WUBU_JTOK_COLON   ? ST_NAME_OPEN : -1;
        case ST_NAME_OPEN:   return tok == WUBU_JTOK_QUOTE   ? ST_NAME_VAL : -1;
        case ST_NAME_VAL:
            for (int t = 0; t < WUBU_TOOL_COUNT; ++t) {
                if (tok == tool_tok[t] && (c->enabled & (1u << t)))
                    return ST_NAME_CLOSE;
            }
            return -1;
        case ST_NAME_CLOSE:  return tok == WUBU_JTOK_QUOTE ? ST_COMMA : -1;
        case ST_COMMA:       return tok == WUBU_JTOK_COMMA ? ST_BRANCH_OPEN : -1;
        case ST_BRANCH_OPEN: return tok == WUBU_JTOK_QUOTE ? ST_BRANCH_VAL : -1;
        case ST_BRANCH_VAL:  /* JSON two-branch point */
            if (tok == WUBU_JTOK_ARGS) return ST_ARGS_COLON;
            if (tok == WUBU_JTOK_TEXT) return ST_TEXT_COLON;
            return -1;
        case ST_ARGS_COLON:  return tok == WUBU_JTOK_COLON ? ST_ARGS_VAL : -1;
        case ST_ARGS_VAL:    return tok == WUBU_JTOK_NUM   ? ST_CLOSE : -1;
        case ST_TEXT_COLON:  return tok == WUBU_JTOK_COLON ? ST_TEXT_VAL : -1;
        case ST_TEXT_VAL:    return tok == WUBU_JTOK_STR   ? ST_CLOSE : -1;
        case ST_CLOSE:       return tok == WUBU_JTOK_RBRACE ? ST_DONE : -1;
        default:             return -1;
    }
}

/* Derive a row's mask from the transition function itself. */
static void mask_for(const wubu_spec_constrained_t *c, int state, uint8_t *out)
{
    for (int t = 0; t < WUBU_SPEC_JSON_VOCAB; ++t)
        out[t] = (step(c, state, t) >= 0) ? 1u : 0u;
}

wubu_spec_constrained_t *wubu_spec_constrained_create(uint32_t enabled_tools)
{
    uint32_t valid = (1u << WUBU_TOOL_COUNT) - 1u;
    if (enabled_tools == 0 || (enabled_tools & ~valid) != 0) return NULL;
    wubu_spec_constrained_t *c = (wubu_spec_constrained_t *)
        calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->enabled = enabled_tools;
    c->state = ST_START;
    return c;
}

void wubu_spec_constrained_free(wubu_spec_constrained_t *c) { free(c); }

int wubu_spec_constrained_tool_token(wubu_tool_name_t name)
{
    if ((int)name < 0 || (int)name >= WUBU_TOOL_COUNT) return -1;
    return tool_tok[name];
}

int wubu_spec_constrained_window(const wubu_spec_constrained_t *c,
                                 const int *draft, int draft_len,
                                 uint8_t *window, int window_cap)
{
    if (!c || !draft || !window || draft_len < 0 ||
        window_cap < draft_len * WUBU_SPEC_JSON_VOCAB) return -2;

    int first_bad = -1;
    int state = ST_START;
    for (int r = 0; r < draft_len; ++r) {
        uint8_t *row = window + (size_t)r * WUBU_SPEC_JSON_VOCAB;
        if (first_bad < 0) {
            mask_for(c, state, row);
            int ns = step(c, state, draft[r]);
            if (ns < 0) {
                first_bad = r;
                memset(row, 0, WUBU_SPEC_JSON_VOCAB); /* invalidate at draft r */
            } else {
                state = ns;
            }
        } else {
            memset(row, 0, WUBU_SPEC_JSON_VOCAB); /* all later rows invalid */
        }
    }
    return first_bad;
}

int wubu_spec_constrained_advance(wubu_spec_constrained_t *c,
                                  const int *draft, int draft_len)
{
    if (!c || !draft || draft_len < 0) return -1;
    int consumed = 0;
    for (int i = 0; i < draft_len; ++i) {
        int ns = step(c, c->state, draft[i]);
        if (ns < 0) break;               /* stop at first illegal draft */
        c->state = ns;
        ++consumed;
    }
    return consumed;
}

int wubu_spec_constrained_commit(wubu_spec_constrained_t *c,
                                 const int *draft, const int *actual, int len)
{
    if (!c || !draft || !actual || len < 0) return -1;
    int div = len;
    for (int i = 0; i < len; ++i) {
        if (draft[i] != actual[i]) { div = i; break; } /* first divergence */
    }
    for (int i = 0; i < div; ++i) {       /* commit verified prefix only */
        int ns = step(c, c->state, actual[i]);
        if (ns < 0) return i;
        c->state = ns;
    }
    return div;
}

int wubu_spec_constrained_mask(const wubu_spec_constrained_t *c,
                               uint8_t *out, int cap)
{
    if (!c || !out || cap < WUBU_SPEC_JSON_VOCAB) return -1;
    mask_for(c, c->state, out);
    return 0;
}

int wubu_spec_constrained_state(const wubu_spec_constrained_t *c)
{
    return c ? c->state : -1;
}
