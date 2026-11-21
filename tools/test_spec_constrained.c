/* test_spec_constrained.c -- gate for LP-09 (masked multi-row constrained
 * decoding window, JSON two-branch grammar). */
#include <stdio.h>
#include <string.h>
#include "wubu_spec_constrained.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define V WUBU_SPEC_JSON_VOCAB

static int row_has(const uint8_t *row, int tok)
{
    return tok >= 0 && tok < V && row[tok] != 0;
}

static int row_count(const uint8_t *row)
{
    int n = 0;
    for (int i = 0; i < V; ++i) n += row[i] ? 1 : 0;
    return n;
}

static int row_all_zero(const uint8_t *row)
{
    for (int i = 0; i < V; ++i) if (row[i]) return 0;
    return 1;
}

int main(void)
{
    const uint32_t enabled = (1u << WUBU_TOOL_SEARCH) | (1u << WUBU_TOOL_CALC);
    wubu_spec_constrained_t *c = wubu_spec_constrained_create(enabled);
    CHECK(c != NULL);
    if (!c) return 1;

    CHECK(wubu_spec_constrained_create(0) == NULL);           /* empty mask */
    CHECK(wubu_spec_constrained_tool_token(WUBU_TOOL_SEARCH) == WUBU_JTOK_SEARCH);
    CHECK(wubu_spec_constrained_tool_token(WUBU_TOOL_COUNT) == -1);

    /* Legal args-branch document: { "tool" : "search" , "args" : 12 } */
    const int draft[14] = {
        WUBU_JTOK_LBRACE,  WUBU_JTOK_QUOTE, WUBU_JTOK_TOOL,   WUBU_JTOK_QUOTE,
        WUBU_JTOK_COLON,   WUBU_JTOK_QUOTE, WUBU_JTOK_SEARCH, WUBU_JTOK_QUOTE,
        WUBU_JTOK_COMMA,   WUBU_JTOK_QUOTE, WUBU_JTOK_ARGS,   WUBU_JTOK_COLON,
        WUBU_JTOK_NUM,     WUBU_JTOK_RBRACE
    };

    /* 1. full legal draft: window is valid; masks intact across every row. */
    {
        uint8_t win[14 * V];
        int s0 = wubu_spec_constrained_state(c);
        int fb = wubu_spec_constrained_window(c, draft, 14, win, 14 * V);
        CHECK(fb == -1);                                   /* no illegal draft */
        CHECK(wubu_spec_constrained_state(c) == s0);       /* state untouched */
        CHECK(row_count(win + 0 * V) == 1 &&
              row_has(win + 0 * V, WUBU_JTOK_LBRACE));
        /* tool-name row: only enabled names legal */
        CHECK(row_has(win + 6 * V, WUBU_JTOK_SEARCH) &&
              row_has(win + 6 * V, WUBU_JTOK_CALC));
        CHECK(!row_has(win + 6 * V, WUBU_JTOK_WEATHER) &&
              !row_has(win + 6 * V, WUBU_JTOK_CODE));
        /* JSON two-branch row: exactly {args, text} */
        CHECK(row_count(win + 10 * V) == 2 &&
              row_has(win + 10 * V, WUBU_JTOK_ARGS) &&
              row_has(win + 10 * V, WUBU_JTOK_TEXT));
        CHECK(row_has(win + 12 * V, WUBU_JTOK_NUM));       /* args value row */
    }

    /* 2. illegal draft (disabled tool name at index 6) is rejected and all
     *    rows from the first illegal draft onward are invalidated. */
    {
        int bad[14];
        uint8_t wbad[14 * V];
        memcpy(bad, draft, sizeof(draft));
        bad[8] = WUBU_JTOK_WEATHER;         /* illegal where COMMA is expected */
        int fb = wubu_spec_constrained_window(c, bad, 14, wbad, 14 * V);
        CHECK(fb == 8);
        CHECK(row_all_zero(wbad + 8 * V));
        for (int r = 9; r < 14; ++r) CHECK(row_all_zero(wbad + r * V));
        /* earlier rows remain intact */
        CHECK(row_has(wbad + 0 * V, WUBU_JTOK_LBRACE));
        CHECK(row_count(wbad + 6 * V) == 2 &&
              row_has(wbad + 6 * V, WUBU_JTOK_SEARCH));

        /* illegal value token at index 12: two-branch mask (row 10) intact */
        memcpy(bad, draft, sizeof(draft));
        bad[12] = WUBU_JTOK_WEATHER;
        fb = wubu_spec_constrained_window(c, bad, 14, wbad, 14 * V);
        CHECK(fb == 12);
        CHECK(row_count(wbad + 10 * V) == 2 &&
              row_has(wbad + 10 * V, WUBU_JTOK_ARGS) &&
              row_has(wbad + 10 * V, WUBU_JTOK_TEXT));
        CHECK(row_all_zero(wbad + 12 * V));
    }

    /* 3. advance stops at the first illegal draft token. */
    {
        wubu_spec_constrained_t *a = wubu_spec_constrained_create(enabled);
        CHECK(a != NULL);
        if (a) {
            int bad[14];
            uint8_t m[V];
            CHECK(wubu_spec_constrained_advance(a, draft, 14) == 14);
            CHECK(wubu_spec_constrained_mask(a, m, V) == 0 &&
                  row_count(m) == 0);                       /* DONE state */
            wubu_spec_constrained_free(a);

            a = wubu_spec_constrained_create(enabled);
            memcpy(bad, draft, sizeof(draft));
            bad[6] = WUBU_JTOK_WEATHER;
            CHECK(wubu_spec_constrained_advance(a, bad, 14) == 6);
            wubu_spec_constrained_free(a);
        }
    }

    /* 4. commit at divergence: verify prefix committed, mask still two-way. */
    {
        int actual[14];
        uint8_t m[V];
        wubu_spec_constrained_t *cm = wubu_spec_constrained_create(enabled);
        CHECK(cm != NULL);
        if (cm) {
            memcpy(actual, draft, sizeof(draft));
            actual[10] = WUBU_JTOK_TEXT;                    /* branch diverged */
            actual[12] = WUBU_JTOK_STR;
            CHECK(wubu_spec_constrained_commit(cm, draft, actual, 14) == 10);
            CHECK(wubu_spec_constrained_mask(cm, m, V) == 0);
            CHECK(row_count(m) == 2 &&
                  row_has(m, WUBU_JTOK_ARGS) &&
                  row_has(m, WUBU_JTOK_TEXT));
            /* re-mask the actual branch from divergence: fully legal */
            CHECK(wubu_spec_constrained_advance(cm, actual + 10, 4) == 4);
            CHECK(wubu_spec_constrained_mask(cm, m, V) == 0 &&
                  row_count(m) == 0);
            wubu_spec_constrained_free(cm);
        }
    }

    /* 5. invalid arguments return negative status values. */
    {
        uint8_t win[14 * V], small[V];
        CHECK(wubu_spec_constrained_window(c, draft, 14, win, 4) == -2);
        CHECK(wubu_spec_constrained_window(NULL, draft, 14, win, 14 * V) == -2);
        CHECK(wubu_spec_constrained_mask(c, small, 4) == -1);
        CHECK(wubu_spec_constrained_advance(NULL, draft, 14) == -1);
    }

    wubu_spec_constrained_free(c);

    if (failures == 0) {
        printf("test_spec_constrained: PASS\n");
        printf("LP-09: PASS\n");
        return 0;
    }
    return 1;
}
