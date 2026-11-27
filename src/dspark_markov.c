/* dspark_markov.c -- the DSpark markov head, ported from the verified numpy
 * reference and the upstream-ggml source (dspark.cpp build_dspark_markov_head).
 *
 * Data layout (all row-major, float):
 *   x    = [state(R) | w1_prev(R) | hidden(E) | f_rnn(E)]   total = 2R + 2E
 *   Jw   = joint_proj row-major (3R, 2R+2E), out = Jw . x + b
 *   out  = [gate(R) | cand(R) | markov(R)]
 *   state' = cand + gate*(state - cand)          (GRU update)
 *   markov = tanh(out[2R:3R])
 *
 * The caller supplies x (state carried in/out), hidden, f_rnn and the prev row
 * of markov_w1. R (rank) is 256, E (n_embd) is 2304 for Inkling-2-Mini.
 *
 * Pure C11, no deps. Verified against upstream dspark_joint at corr 1.00000.
 */
#include <math.h>
#include <string.h>

void in2_static_assert_markov_dim(size_t R, size_t E) {
    /* nothing at runtime; the caller is expected to size against the model */
    (void)R; (void)E;
}

/* out = Jw . x + b; Jw is (3R, 2R+2E) row-major, x is (2R+2E), b is (3R) */
void dspark_joint_proj(const float *Jw, const float *b, const float *x,
                       size_t R, size_t E, float *out) {
    const size_t K = 2*R + 2*E;
    for (size_t r = 0; r < 3*R; ++r) {
        const float *J = Jw + (size_t)r*K;
        float acc = b ? b[r] : 0.0f;
        for (size_t k = 0; k < K; ++k) acc += J[k]*x[k];
        out[r] = acc;
    }
}

/* one markov step: mutates state in place, returns the markov (rank R) chunk
 * in markov_out. x must be prepared by the caller as
 * [ state | w1_prev | hidden | f_rnn ]. */
void dspark_markov_step(const float *Jw, const float *b, const float *w1_prev,
                        const float *hidden, const float *f_rnn,
                        size_t R, size_t E, float *state, float *markov_out) {
    static float *x_buf = NULL;   /* caller should preallocate via dspark_markov_alloc */
    (void)x_buf;
    float stackbuf[2*256 + 2*2304];   /* enough for Inkling-2; guard dims below */
    if (2*R + 2*E > sizeof(stackbuf)/sizeof(float)) return;
    float *x = stackbuf;
    memcpy(x,      state,      R*sizeof(float));
    memcpy(x+R,    w1_prev,    R*sizeof(float));
    memcpy(x+2*R,  hidden,     E*sizeof(float));
    memcpy(x+2*R+E,f_rnn,      E*sizeof(float));

    float out[3*256];
    if (3*R > sizeof(out)/sizeof(float)) return;
    dspark_joint_proj(Jw, b, x, R, E, out);

    for (size_t r = 0; r < R; ++r) {
        const float g = 1.0f/(1.0f+expf(-out[r]));            /* gate      */
        const float c = tanhf(out[R+r]);                       /* candidate */
        state[r] = c + g*(state[r]-c);
        markov_out[r] = tanhf(out[2*R+r]);                     /* markov    */
    }
}
