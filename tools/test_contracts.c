/*
 * test_contracts.c -- the RUNTIME CONTRACTS gate (roadmap #5: "keeps
 * self-modification bounded by something stronger than loss went
 * down on the last batch").
 *
 * Asserts:
 *   1. a contract-clean mutation passes (all probes within bounds)
 *   2. a contract-violating mutation is REJECTED (ball closure blown,
 *      quant error too big) even though its loss improved
 *   3. the finite guard catches NaN/Inf
 *   4. the set expands via versioned hive meta-cells (only grows)
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "wubu_contracts.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_contracts (the runtime floor) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_contracts_t ct;
    if (wubu_contracts_init(&ct, &tissue) != 0) FAIL("contracts init");
    printf("  default set: %d contracts (ball/exp/route/quant/finite)\n", ct.n);

    /* 1. a contract-clean mutation passes */
    float clean[5] = { 1e-5f, 1e-6f, 0.4f, 1e-3f, 1.0f };
    int v = wubu_contracts_check(&ct, clean);
    printf("  clean mutation: %d violations (0 = pass)\n", v);
    if (v != 0) FAIL("clean mutation violated a contract");

    /* 2. a violating mutation is rejected — the ball closure blown
     * (0.5 >> 1e-3) and the quant error too big (0.05 >> 1e-2).
     * Its loss improved, but the contract floor holds. */
    float bad[5] = { 0.5f, 1e-6f, 0.4f, 0.05f, 1.0f };
    v = wubu_contracts_check(&ct, bad);
    printf("  violating mutation: %d violations (rejected despite loss gain)\n", v);
    if (v < 2) FAIL("the ball/quant violations were not caught (%d)", v);

    /* 3. the finite guard catches NaN */
    float nanbuf[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    if (!wubu_contracts_finite(nanbuf, 8)) FAIL("finite buffer flagged");
    nanbuf[3] = NAN;
    if (wubu_contracts_finite(nanbuf, 8)) FAIL("NaN not caught");
    nanbuf[3] = 4; nanbuf[5] = INFINITY;
    if (wubu_contracts_finite(nanbuf, 8)) FAIL("Inf not caught");
    printf("  finite guard: clean+NaN+Inf all detected\n");

    /* 4. the set expands via versioned hive meta-cells */
    uint32_t vnew = wubu_contracts_add(&ct, WUBU_CT_EXP, 1e-5f, 1);
    if (vnew == 0) FAIL("expansion failed");
    size_t live = wubu_hive_live(&tissue);
    printf("  expansion: version %u added (hive meta-cells %zu)\n", vnew, live);
    if (vnew < 5) FAIL("version did not grow (%u)", vnew);
    if (live < 1) FAIL("the meta-cell did not land in the hive");

    char stats[256];
    wubu_contracts_stats(&ct, stats, sizeof(stats));
    printf("  stats: %s\n", stats);
    if (ct.n_violations == 0) FAIL("violations not counted");

    printf("=== ALL CONTRACTS TESTS PASSED (the floor is runtime) ===\n");
    return 0;
}
