#ifndef WUBU_SAMPLER_CHAIN_H
#define WUBU_SAMPLER_CHAIN_H

/*
 * wubu_sampler_chain.h -- EG-12: ordered, composable logit stages
 * (MLX / llama.cpp sampler chain lineage), reimplemented in C11.
 *
 * Each stage either transforms logits (repetition, temperature) or filters
 * candidates (top-k, top-p, min-p, typical-p). Stages run in insertion
 * order, so order is observable. The final pick uses a deterministic RNG.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_SAMPLE_REPETITION = 0,
    WUBU_SAMPLE_TEMPERATURE,
    WUBU_SAMPLE_TOP_K,
    WUBU_SAMPLE_TOP_P,
    WUBU_SAMPLE_MIN_P,
    WUBU_SAMPLE_TYPICAL_P
} wubu_sampler_stage_t;

typedef struct wubu_sampler_chain wubu_sampler_chain_t;

wubu_sampler_chain_t *wubu_sampler_chain_create(int max_stages);
void wubu_sampler_chain_free(wubu_sampler_chain_t *c);

/* Append a stage. param: penalty (repetition), temperature, k, p, p, p. */
int wubu_sampler_chain_add(wubu_sampler_chain_t *c,
                            wubu_sampler_stage_t stage, double param);

/* Apply all stages to logits[n] in order. mask_out[n] receives 1/0.
 * prev_tokens may be NULL when n_prev == 0. Returns allowed count, -1 on
 * bad args. */
int wubu_sampler_chain_apply(const wubu_sampler_chain_t *c,
                              float *logits, int n,
                              const int *prev_tokens, int n_prev,
                              unsigned char *mask_out);

/* Deterministic pick among allowed candidates. seed 0 means argmax. */
int wubu_sampler_chain_pick(const wubu_sampler_chain_t *c,
                            const float *logits, const unsigned char *mask,
                            int n, uint64_t seed);

int wubu_sampler_chain_stage_count(const wubu_sampler_chain_t *c);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SAMPLER_CHAIN_H */
