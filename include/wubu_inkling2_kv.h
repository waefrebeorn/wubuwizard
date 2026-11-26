/* INKLING2 KV CACHE - see src/wubu_inkling2_kv.c for why this is per-visit and why raw conv rows are kept. */
#ifndef WUBU_INKLING2_KV_H
#define WUBU_INKLING2_KV_H

#include <stddef.h>

typedef struct {
    int    n_visit;     /* layer visits per forward = stored layers + extra loop passes */
    int    dim;         /* key/value width */
    int    window;      /* sliding window in positions; 0 = full attention, grows with context */
    int    n_pos_cap;   /* MAX rows allocated to any single visit (kept for the uniform case) */
    int    n_pos_max;   /* highest legal ABSOLUTE position; independent of the ring size, which is what wraps */
    int    n_stored;    /* positions stored so far */
    int   *cap;         /* rows per visit; NULL means every visit gets n_pos_cap */
    size_t *off;        /* element offset of each visit inside k/v; NULL means uniform stride */
    float *k, *v;       /* post-conv keys/values, [n_visit][n_pos_cap][dim]; NULL when q8 is on */
    float *kraw, *vraw; /* pre-conv rows for the 4-tap causal conv, [n_visit][3][dim] */
    unsigned char *kq, *vq; /* q8_0 blocks when q8 is on, else NULL: rows * rowq bytes each */
    float *kscr, *vscr; /* ONE dequantised row per array, returned by the accessors (read-only) */
    size_t rowq;        /* bytes per quantised row */
    int    q8;          /* 1 = keys/values are stored as q8_0 */
    struct wubu_tqvec *tq;  /* when set, rows are TurboQuant codes and q8 is also 1 (4-8 bits/coord) */
    float *kn, *vn;         /* per-row norm: tqvec quantises UNIT vectors, so the row scale is stored separately */
    float *encsc;           /* dim floats, scratch for the normalised row on store */
    /* IN2_TQ_VERIFY: parallel f32 shadow of every stored row, so a read can be checked against what was written.
     * Distinguishes a slot mismatch from a decode error from a bug outside the KV path. */
    float *kshad, *vshad;
    /* Per-visit dequantised mirror. The accessors return ONE scratch row, so every attention access re-decodes that row
     * (TurboQuant decode + 1024-point WHT); with 64 sliding visits x 1024 window rows x 1024 query rows that is ~7.5e7
     * decodes and it is 78-80 % of prefill layer time. The mirror makes it one decode per (visit,pos): 1024x fewer.
     * window_rows * dim, NOT rows-total * dim - the latter is 288 MB next to a 7.9 GB model. kitag[pos] holds the visit
     * id that filled the slot (-1 empty), so a new visit invalidates every slot for free. */
    float *kmir, *vmir;
    int   *kitag, *vitag;
    /* Rewindable row tags, one per row: pos+1 = this row holds that position, -1 = deliberately invalidated,
     * 0 = never written (behaves as before). The slot is pos % cap, so without a tag a ring slot silently serves
     * whatever position last wrapped onto it - which is what broke speculative verification. */
    int   *kpostag, *vpostag;
    size_t n_row_total;
    /* The 4-tap causal conv keeps three rows per visit in kraw/vraw, advanced by memmove. Invalidating K/V rows
     * does not undo that, so a rejected draft would convolve against draft history. These hold the pre-verification
     * state: 3 x dim per visit each (1.77 MB total at 72 visits x 1024 dim). */
    float *ksnap, *vsnap;
} in2_kv_t;

/* shared with the full-attention layer (other translation unit): same cache, same visit numbering */
extern int g_kv_window;   /* ring size of the K/V cache, 0 = linear */
in2_kv_t *in2_kv_current(int *visit, int *pos0);
void      in2_kv_advance(void);
/* Invalidate every row at or beyond from_pos and pull n_stored back. O(visits x range); the range is a rejected
 * draft, so single digits in practice. from_pos == n_stored is a no-op. */
void      in2_kv_rewind(in2_kv_t *kv, int from_pos);
/* Save/restore the conv history (kraw/vraw). Call save before a speculative verification pass and restore when
 * fewer drafts are accepted than were verified, then rewind the K/V rows to the accepted prefix. */
int       in2_kv_conv_save(in2_kv_t *kv);
/* Both caches at once: g_kv holds the residual stream history, g_kv2 the two convs. A rollback must cover both. */
int       in2_state_save(void);
void      in2_state_restore(void);
void      in2_state_rewind(int from_pos);
void      in2_kv_conv_restore(in2_kv_t *kv);
void  in2_kv_begin(int pos0);
void  in2_kv_store_row_k(in2_kv_t *kv, int visit, const float *row);
void      in2_kv_store_row_v(in2_kv_t *kv, int visit, const float *row);
void      in2_conv_stream_res(int vvis, const float *x, int n_tok, int ch, const float *w, int k, float *out);
void      in2_conv_stream_moe(int vvis, const float *x, int n_tok, int ch, const float *w, int k, float *out);

int   in2_kv_init(in2_kv_t *kv, int n_visit, int dim, int window, int n_pos_cap);
/* rows[v] = how many positions visit v must hold: the context for a full-attention visit, the sliding
 * window for the rest. This is what makes a long context affordable - see the .c file. */
int   in2_kv_init_pervisit(in2_kv_t *kv, int n_visit, int dim, int window, const int *rows);
size_t in2_kv_bytes(const in2_kv_t *kv);
void  in2_kv_free(in2_kv_t *kv);
float *in2_kv_k(in2_kv_t *kv, int visit, int pos);
/* Store a post-conv key/value row at an absolute position. This replaces writing through in2_kv_k/v: a quantised cache
 * cannot hand out a writable scratch, so the store is explicit and owns the format. */
void  in2_kv_store_post_k(in2_kv_t *kv, int visit, int pos, const float *row);
void  in2_kv_store_post_v(in2_kv_t *kv, int visit, int pos, const float *row);
float *in2_kv_v(in2_kv_t *kv, int visit, int pos);
int   in2_kv_resident(const in2_kv_t *kv, int pos);
void  in2_kv_store_raw(in2_kv_t *kv, int visit, const float *krow, const float *vrow);
const float *in2_kv_hist_k(const in2_kv_t *kv, int visit, int back);
const float *in2_kv_hist_v(const in2_kv_t *kv, int visit, int back);
void  in2_kv_tick(in2_kv_t *kv);
void  in2_kv_conv_row_dyn(const float *hist3, const float *xrow, int channels, const float *stat,
                          const float *dwrow, int n_head, int k, float *out);
void  in2_kv_conv_row(const float *hist3, const float *xrow, int channels, const float *w, int k, float *out);

#endif
