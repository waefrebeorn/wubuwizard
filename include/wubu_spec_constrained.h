#ifndef WUBU_SPEC_CONSTRAINED_H
#define WUBU_SPEC_CONSTRAINED_H

/* wubu_spec_constrained.h -- LP-09: masked multi-row window for constrained
 * decoding (JSON two-branch grammar + tool-name vocabulary).
 * C11, opaque struct, status-as-value, no third-party deps. */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Tool-name enum (also a bitmask position for wubu_spec_constrained_create). */
typedef enum {
    WUBU_TOOL_SEARCH = 0,
    WUBU_TOOL_CALC,
    WUBU_TOOL_WEATHER,
    WUBU_TOOL_CODE,
    WUBU_TOOL_COUNT
} wubu_tool_name_t;

/* Token ids of the small JSON grammar. */
typedef enum {
    WUBU_JTOK_LBRACE = 0,
    WUBU_JTOK_RBRACE,
    WUBU_JTOK_QUOTE,
    WUBU_JTOK_COLON,
    WUBU_JTOK_COMMA,
    WUBU_JTOK_TOOL,
    WUBU_JTOK_ARGS,
    WUBU_JTOK_TEXT,
    WUBU_JTOK_SEARCH,
    WUBU_JTOK_CALC,
    WUBU_JTOK_WEATHER,
    WUBU_JTOK_CODE,
    WUBU_JTOK_NUM,
    WUBU_JTOK_STR,
    WUBU_JTOK_EOF,
    WUBU_JTOK_COUNT
} wubu_jtok_t;

#define WUBU_SPEC_JSON_VOCAB 16

typedef struct wubu_spec_constrained wubu_spec_constrained_t;

/* enabled_tools is a bitmask (1u << wubu_tool_name_t) of names allowed after
 * {"tool": ...}. Returns NULL on an empty/out-of-range mask or alloc failure. */
wubu_spec_constrained_t *wubu_spec_constrained_create(uint32_t enabled_tools);
void wubu_spec_constrained_free(wubu_spec_constrained_t *c);

/* Token id for a tool name, or -1 for an invalid name. */
int wubu_spec_constrained_tool_token(wubu_tool_name_t name);

/* Build a masked multi-row window for a draft of draft_len tokens.
 * `window` must hold at least draft_len*WUBU_SPEC_JSON_VOCAB bytes; row r is
 * the legal-token mask (1=legal) for draft token r given all preceding draft
 * tokens were legal. The grammar is simulated from a fresh START state; the
 * object's own state is not modified.
 * Returns the index of the first illegal draft token, or -1 if all legal.
 * From the first illegal row onward every byte is cleared (invalidated).
 * Returns -2 on invalid arguments. */
int wubu_spec_constrained_window(const wubu_spec_constrained_t *c,
                                 const int *draft, int draft_len,
                                 uint8_t *window, int window_cap);

/* Advance the object's grammar state along a legal draft prefix, stopping
 * before the first illegal token. Returns the number consumed, or -1 on
 * invalid arguments. */
int wubu_spec_constrained_advance(wubu_spec_constrained_t *c,
                                  const int *draft, int draft_len);

/* Compare draft vs actual and commit (consume into grammar state) the
 * verified prefix before the first divergence. Returns the number committed
 * (= first divergent index, or len if identical). Tokens at/after divergence
 * are left for the caller to re-mask. Returns -1 on invalid input. */
int wubu_spec_constrained_commit(wubu_spec_constrained_t *c,
                                 const int *draft, const int *actual, int len);

/* Copy the current legal-token mask (WUBU_SPEC_JSON_VOCAB bytes) into out.
 * Returns 0 on success, -1 on invalid input. */
int wubu_spec_constrained_mask(const wubu_spec_constrained_t *c,
                               uint8_t *out, int cap);

/* Current grammar state id (opaque integer, for diagnostics). */
int wubu_spec_constrained_state(const wubu_spec_constrained_t *c);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_SPEC_CONSTRAINED_H */
