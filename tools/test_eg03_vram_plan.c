/* test_eg03_vram_plan.c -- EG-03 gate: metadata-first VRAM planner.
 * Covers the 6 GB / 8.49 GB refusal, streaming fallback, UMA-corrected
 * measurement, context shrink, 80% rule and recovery wait.
 * Prints: test_eg03_vram_plan: PASS. */
#include <stdio.h>
#include <string.h>

#include "wubu_vram_plan.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define GB (1024ULL * 1024ULL * 1024ULL)
#define INKling_IQ2_XXS_BYTES 8486236352ULL

static void show(const char *tag, const wubu_vram_plan_t *p) {
    printf("EG-03 %-16s decision=%-6s usable=%lluMB weight=%lluMB "
           "ctx=%d wait=%dms\n", tag, wubu_vram_plan_name(p->decision),
           (unsigned long long)(p->usable_bytes >> 20),
           (unsigned long long)(p->weight_bytes >> 20),
           p->allowed_ctx, p->wait_ms);
}

int main(void) {
    wubu_vram_plan_t p;

    /* 1. The T1 refusal: 8.49 GB quant cannot fit a 6 GB device. */
    CHECK(wubu_vram_plan_make(6 * GB, 6 * GB, INKling_IQ2_XXS_BYTES,
                              8192, 4096, 0, &p) == 0);
    show("8.49GB/no-stream", &p);
    CHECK(p.decision == WUBU_VRAM_PLAN_REFUSE);
    CHECK(p.allowed_ctx == 0);
    CHECK(p.usable_bytes == (6 * GB / 10) * 8); /* 80% rule */

    /* 2. Same model with SSD streaming: weights streamed, KV still fits. */
    CHECK(wubu_vram_plan_make(6 * GB, 6 * GB, INKling_IQ2_XXS_BYTES,
                              8192, 4096, 1, &p) == 0);
    show("8.49GB/stream", &p);
    CHECK(p.decision == WUBU_VRAM_PLAN_STREAM);
    CHECK(p.allowed_ctx == 4096);

    /* 3. UMA correction: raw cudaMemGetInfo says 3 GB, corrected says 5 GB.
     * A 2.5 GB quant is refused by the raw figure but fits when corrected. */
    CHECK(wubu_vram_plan_make(6 * GB, 5 * GB, (uint64_t)(2.5 * GB),
                              8192, 4096, 0, &p) == 0);
    show("UMA-corrected", &p);
    CHECK(p.decision == WUBU_VRAM_PLAN_FIT);
    CHECK(p.allowed_ctx == 4096);

    CHECK(wubu_vram_plan_make(6 * GB, 3 * GB, (uint64_t)(2.5 * GB),
                              8192, 4096, 0, &p) == 0);
    show("UMA-raw", &p);
    CHECK(p.decision == WUBU_VRAM_PLAN_REFUSE);

    /* 4. Context shrink when KV is large but weights fit. */
    CHECK(wubu_vram_plan_make(6 * GB, 6 * GB, 2 * GB,
                              1024 * 1024, 4096, 0, &p) == 0);
    show("large-KV", &p);
    CHECK(p.decision == WUBU_VRAM_PLAN_SHRINK);
    CHECK(p.allowed_ctx > 0 && p.allowed_ctx < 4096);

    /* 5. Recovery wait on a nearly empty device. */
    CHECK(wubu_vram_plan_make(6 * GB, 64ULL * 1024 * 1024, 1 * GB,
                              8192, 512, 0, &p) == 0);
    show("near-empty", &p);
    CHECK(p.wait_ms > 0);
    CHECK(p.decision == WUBU_VRAM_PLAN_REFUSE);

    /* 6. Fail closed on bad input. */
    CHECK(wubu_vram_plan_make(6 * GB, 6 * GB, 1 * GB, 0, 4096, 0, &p) == -1);
    CHECK(wubu_vram_plan_make(0, 6 * GB, 1 * GB, 8192, 4096, 0, &p) == -1);

    if (failures) {
        fprintf(stderr, "test_eg03_vram_plan: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg03_vram_plan: PASS\n");
    return 0;
}
