/* test_diffusion_sampler.c -- LP-13 gate.
 *
 * Proves, with real -ffast-math C11 code, that the fused canvas x vocab pass
 * is BIT-IDENTICAL to the simple unfused reference:
 *   - sampled token ids match,
 *   - sampled probabilities match bit-for-bit,
 *   - entropies match bit-for-bit,
 *   - stopping decisions match,
 * and that the mechanism is real (entropy/stopping semantics, deterministic
 * seeded Gumbel draws, seed sensitivity).  Prints 'test_diffusion_sampler: PASS'.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_diff_sampler.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

static uint32_t fbits(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static int result_bit_eq(const wubu_diff_result_t *a, const wubu_diff_result_t *b)
{
    return a->token == b->token &&
           fbits(a->prob) == fbits(b->prob) &&
           fbits(a->entropy) == fbits(b->entropy) &&
           a->stopped == b->stopped;
}

#define ROWS 64
#define VOCAB 256
#define STOP_H 0.35f
#define SEED 0x123456789ABCDEF0ULL

int main(void)
{
    wubu_diff_sampler_config_t cfg;
    float *logits;
    wubu_diff_result_t *fused, *unfused, *again, *other_seed, *all_stop;
    int r, i, differing = 0;

    logits = (float *)malloc(sizeof(float) * ROWS * VOCAB);
    fused   = (wubu_diff_result_t *)calloc(ROWS, sizeof(*fused));
    unfused = (wubu_diff_result_t *)calloc(ROWS, sizeof(*unfused));
    again   = (wubu_diff_result_t *)calloc(ROWS, sizeof(*again));
    other_seed = (wubu_diff_result_t *)calloc(ROWS, sizeof(*other_seed));
    all_stop   = (wubu_diff_result_t *)calloc(ROWS, sizeof(*all_stop));
    if (!logits || !fused || !unfused || !again || !other_seed || !all_stop) {
        fprintf(stderr, "test_diffusion_sampler: FAIL (allocation)\n");
        return 1;
    }

    /* Build a canvas x vocab logits buffer with a mix of confident (peaked)
     * and uncertain (near-uniform) diffusion positions. */
    for (r = 0; r < ROWS; ++r) {
        for (i = 0; i < VOCAB; ++i) {
            float val;
            if ((r & 1) == 0) {
                /* peaked: one dominant logit -> low entropy -> should stop */
                val = (i == (r % VOCAB)) ? 10.0f : -6.0f;
            } else {
                /* near-uniform: high entropy -> should keep decoding */
                val = 0.0f;
            }
            logits[(size_t)r * VOCAB + i] = val;
        }
    }

    cfg.canvas_rows = ROWS;
    cfg.vocab       = VOCAB;
    cfg.stop_entropy = STOP_H;
    cfg.seed        = SEED;

    /* Argument validation is a real status-as-value path. */
    CHECK(wubu_diff_sampler_validate(&cfg, logits, fused) == WUBU_DIFF_OK);
    CHECK(wubu_diff_sampler_validate(NULL, logits, fused) == WUBU_DIFF_ERR_ARG);
    CHECK(wubu_diff_sampler_validate(&cfg, NULL, fused) == WUBU_DIFF_ERR_ARG);
    {
        wubu_diff_sampler_config_t bad = cfg;
        bad.vocab = 0;
        CHECK(wubu_diff_sample_fused(&bad, logits, fused) == WUBU_DIFF_ERR_ARG);
    }

    /* The two implementations must agree bit-for-bit on every row. */
    CHECK(wubu_diff_sample_fused(&cfg, logits, fused) == WUBU_DIFF_OK);
    CHECK(wubu_diff_sample_unfused(&cfg, logits, unfused) == WUBU_DIFF_OK);
    for (r = 0; r < ROWS; ++r) {
        if (!result_bit_eq(&fused[r], &unfused[r])) {
            fprintf(stderr,
                    "row %d: fused(t=%d,p=%08x,H=%08x,stop=%d) != "
                    "unfused(t=%d,p=%08x,H=%08x,stop=%d)\n",
                    r, fused[r].token, fbits(fused[r].prob),
                    fbits(fused[r].entropy), fused[r].stopped,
                    unfused[r].token, fbits(unfused[r].prob),
                    fbits(unfused[r].entropy), unfused[r].stopped);
            failures++;
        }
    }

    /* Mechanism checks: entropy semantics, stopping, valid draws. */
    for (r = 0; r < ROWS; ++r) {
        const wubu_diff_result_t *u = &unfused[r];
        CHECK(u->token >= 0 && u->token < VOCAB);
        CHECK(u->prob > 0.0f && u->prob <= 1.0f);
        CHECK(u->entropy >= 0.0f);
        if ((r & 1) == 0) {
            CHECK(u->entropy < STOP_H);
            CHECK(u->stopped == 1);
        } else {
            CHECK(u->entropy > 4.0f);          /* ln(256) ~= 5.545 */
            CHECK(u->stopped == 0);
        }
    }

    /* Determinism: same seed -> identical fused results. */
    CHECK(wubu_diff_sample_fused(&cfg, logits, again) == WUBU_DIFF_OK);
    for (r = 0; r < ROWS; ++r)
        CHECK(result_bit_eq(&fused[r], &again[r]));

    /* Seed sensitivity: a different seed must move at least one draw. */
    {
        wubu_diff_sampler_config_t s2 = cfg;
        s2.seed = SEED ^ 0xA5A5A5A5A5A5A5A5ULL;
        CHECK(wubu_diff_sample_fused(&s2, logits, other_seed) == WUBU_DIFF_OK);
        for (r = 0; r < ROWS; ++r)
            if (other_seed[r].token != fused[r].token) differing++;
        CHECK(differing > 0);
    }

    /* Stopping threshold is honoured: a huge threshold stops every position. */
    {
        wubu_diff_sampler_config_t hs = cfg;
        hs.stop_entropy = 1000.0f;
        CHECK(wubu_diff_sample_fused(&hs, logits, all_stop) == WUBU_DIFF_OK);
        for (r = 0; r < ROWS; ++r) CHECK(all_stop[r].stopped == 1);
    }

    free(logits);
    free(fused);
    free(unfused);
    free(again);
    free(other_seed);
    free(all_stop);

    if (failures) {
        fprintf(stderr, "test_diffusion_sampler: FAIL (%d checks)\n", failures);
        return 1;
    }
    printf("test_diffusion_sampler: PASS\n");
    return 0;
}
