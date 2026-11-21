/* test_spec_seed.c -- gate for LP-08 (deterministic seeded rejection sampling). */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "wubu_spec_seed.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define N_OUT 12

int main(void)
{
    const uint64_t seed = 0x123456789ABCDEF0ULL;
    const float probs[8] = { .9f, .8f, .7f, .6f, .5f, .4f, .3f, .2f };
    int a[N_OUT], b[N_OUT], shorta[6], ids[N_OUT];

    /* 1. same seed -> identical accepted prefix (and identical full stream) */
    int na = wubu_spec_seed_accept(probs, 8, seed, a, N_OUT);
    int nb = wubu_spec_seed_accept(probs, 8, seed, b, N_OUT);
    int ns = wubu_spec_seed_accept(probs, 8, seed, shorta, 6);
    CHECK(na == N_OUT && nb == N_OUT && ns == 6);
    CHECK(memcmp(a, b, (size_t)N_OUT * sizeof(int)) == 0);
    CHECK(memcmp(shorta, a, (size_t)6 * sizeof(int)) == 0); /* prefix */
    for (int i = 0; i < na; ++i) CHECK(a[i] >= 0 && a[i] < 8);

    /* 2. a different seed changes the accepted stream (real rejection path) */
    {
        int diff_found = 0;
        for (uint64_t s2 = seed + 1; s2 < seed + 16 && !diff_found; ++s2) {
            int n2 = wubu_spec_seed_accept(probs, 8, s2, ids, N_OUT);
            CHECK(n2 == N_OUT);
            if (memcmp(a, ids, (size_t)N_OUT * sizeof(int)) != 0) diff_found = 1;
        }
        CHECK(diff_found);
    }

    /* 3. rejection actually rejects: zero probability accepts nothing */
    {
        const float zero[8] = { 0 };
        CHECK(wubu_spec_seed_accept(zero, 8, seed, ids, N_OUT) == 0);
    }

    /* 4. invalid inputs return a negative status */
    CHECK(wubu_spec_seed_accept(NULL, 8, seed, ids, N_OUT) == -1);
    CHECK(wubu_spec_seed_accept(probs, 0, seed, ids, N_OUT) == -1);
    CHECK(wubu_spec_seed_accept(probs, 8, seed, NULL, N_OUT) == -1);

    if (failures == 0) {
        printf("test_spec_seed: PASS\n");
        printf("LP-08: PASS\n");
        return 0;
    }
    return 1;
}
