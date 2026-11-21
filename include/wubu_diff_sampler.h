/* wubu_diff_sampler.h -- LP-13: fused diffusion logits/sampler.
 *
 * One canvas x vocab pass over a (row-major) logits buffer computes, for every
 * canvas position (row):
 *   - the softmax distribution (stable, max-subtracted) and its Shannon
 *     entropy (nats),
 *   - the diffusion stopping decision (entropy <= stop_entropy => confident,
 *     iterative decoding may stop at this position),
 *   - a bit-exact Gumbel-max token draw driven by a deterministic per-row RNG.
 *
 * Two entry points are provided over the SAME numeric primitives:
 *   wubu_diff_sample_fused   -- entropy, stopping and sampling share one
 *                               combined per-row pass (cache-hot, single
 *                               traversal of the row after normalization).
 *   wubu_diff_sample_unfused -- a deliberately simple reference that calls
 *                               separate softmax / entropy / stop / sample
 *                               passes.  It exists so the gate can assert
 *                               that fusion changed NOTHING numerically.
 *
 * Both are bit-exact with respect to each other: identical token ids, sampled
 * probabilities, entropies and decisions under the project's -ffast-math flags.
 *
 * Pure C11, opaque-free plain API with status-as-value return codes.  No
 * third-party headers; only <stdint.h>/<stddef.h> are needed by clients.
 */
#ifndef WUBU_DIFF_SAMPLER_H
#define WUBU_DIFF_SAMPLER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status values (status-as-value API). */
enum {
    WUBU_DIFF_OK      =  0,  /* success */
    WUBU_DIFF_ERR_ARG = -1,  /* NULL buffer or invalid dimension */
    WUBU_DIFF_ERR_NOMEM = -2 /* scratch allocation failed */
};

/* Immutable configuration for a sampler call.  `canvas_rows` is the number
 * of canvas positions (diffusion slots); `vocab` is the per-position logit
 * count.  `logits` is a contiguous canvas_rows*vocab float array, row-major. */
typedef struct {
    int      canvas_rows;   /* number of canvas positions, must be > 0 */
    int      vocab;         /* vocabulary size per position, must be > 0 */
    float    stop_entropy;  /* stop a position when entropy <= this (nats) */
    uint64_t seed;          /* base seed for the deterministic Gumbel RNG */
} wubu_diff_sampler_config_t;

/* Per-position output (one entry per canvas row). */
typedef struct {
    int   token;    /* sampled token id in [0, vocab) */
    float prob;     /* probability assigned to the sampled token */
    float entropy;  /* Shannon entropy of the row distribution (nats) */
    int   stopped;  /* 1 if entropy <= stop_entropy, else 0 */
} wubu_diff_result_t;

/* Validate config/buffers without doing work.  Returns WUBU_DIFF_OK or
 * WUBU_DIFF_ERR_ARG. */
int wubu_diff_sampler_validate(const wubu_diff_sampler_config_t *cfg,
                               const float *logits,
                               const wubu_diff_result_t *out);

/* Fused path: entropy + stopping + Gumbel sampling in one combined pass. */
int wubu_diff_sample_fused(const wubu_diff_sampler_config_t *cfg,
                           const float *logits,
                           wubu_diff_result_t *out);

/* Unfused reference: separate softmax / entropy / stop / sample passes. */
int wubu_diff_sample_unfused(const wubu_diff_sampler_config_t *cfg,
                             const float *logits,
                             wubu_diff_result_t *out);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_DIFF_SAMPLER_H */
