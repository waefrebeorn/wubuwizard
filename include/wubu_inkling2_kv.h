/* INKLING2 KV CACHE - see src/wubu_inkling2_kv.c for why this is per-visit and why raw conv rows are kept. */
#ifndef WUBU_INKLING2_KV_H
#define WUBU_INKLING2_KV_H

typedef struct {
    int    n_visit;     /* layer visits per forward = stored layers + extra loop passes */
    int    dim;         /* key/value width */
    int    window;      /* sliding window in positions; 0 = full attention, grows with context */
    int    n_pos_cap;   /* rows allocated per visit */
    int    n_stored;    /* positions stored so far */
    float *k, *v;       /* post-conv keys/values, [n_visit][n_pos_cap][dim] */
    float *kraw, *vraw; /* pre-conv rows for the 4-tap causal conv, [n_visit][3][dim] */
} in2_kv_t;

/* shared with the full-attention layer (other translation unit): same cache, same visit numbering */
in2_kv_t *in2_kv_current(int *visit, int *pos0);
void      in2_kv_advance(void);
void  in2_kv_begin(int pos0);
void  in2_kv_store_row_k(in2_kv_t *kv, int visit, const float *row);
void      in2_kv_store_row_v(in2_kv_t *kv, int visit, const float *row);
void      in2_conv_stream_res(int vvis, const float *x, int n_tok, int ch, const float *w, int k, float *out);
void      in2_conv_stream_moe(int vvis, const float *x, int n_tok, int ch, const float *w, int k, float *out);

int   in2_kv_init(in2_kv_t *kv, int n_visit, int dim, int window, int n_pos_cap);
void  in2_kv_free(in2_kv_t *kv);
float *in2_kv_k(in2_kv_t *kv, int visit, int pos);
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
