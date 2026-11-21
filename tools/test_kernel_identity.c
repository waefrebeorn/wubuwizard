/* test_kernel_identity.c -- gate for LP-10 (compile-time kernel specialization).
 *
 * Families are compile-time data rows. The gate proves two things:
 *   1. SELECTION: each family key resolves to its own specialized body, both
 *      before and after a new family row is registered.
 *   2. IDENTITY STABILITY: adding a new family row leaves every existing
 *      kernel-body identity byte-identical, because identity is a pure
 *      function of the body descriptor (never of row index/count/family key).
 *
 * Prints: test_kernel_identity: PASS
 */
#include <stdio.h>
#include <string.h>
#include "wubu_kernel_spec.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

/* ---- compile-time family specialization table (data, not branches) ---- */
static const wubu_kernel_row_t baseline_rows[] = {
    { /* llama-style: GQA, partial rotary, full attention, dense */
      "llama", { WUBU_ATTN_GQA, 64, 0, WUBU_MOE_NONE,
                 { 2, 3 }, 2 }
    },
    { /* qwen-style: GQA, rotary, sliding window, dense */
      "qwen", { WUBU_ATTN_GQA, 128, 4096, WUBU_MOE_NONE,
                { 151643, 151644, 151645 }, 3 }
    },
    { /* mixtral-style: GQA, rotary, full attention, top-8 MoE */
      "mixtral", { WUBU_ATTN_GQA, 48, 0, WUBU_MOE_TOP8,
                   { 2 }, 1 }
    }
};
#define N_BASELINE ((int)(sizeof(baseline_rows) / sizeof(baseline_rows[0])))

/* The new family row added AFTER baseline identities are recorded. */
static const wubu_kernel_row_t new_row = {
    "deepseek", { WUBU_ATTN_MLA, 64, 0, WUBU_MOE_TOP2,
                  { 1, 2, 3, 4 }, 4 }
};

static int body_equal(const wubu_kernel_body_t *a, const wubu_kernel_body_t *b)
{
    if (a->attn != b->attn || a->rotary_pairs != b->rotary_pairs ||
        a->window_tokens != b->window_tokens || a->moe != b->moe ||
        a->n_stop_tokens != b->n_stop_tokens) return 0;
    return memcmp(a->stop_tokens, b->stop_tokens,
                  (size_t)a->n_stop_tokens * sizeof(int)) == 0;
}

int main(void)
{
    uint64_t before[N_BASELINE];
    uint64_t after[N_BASELINE];
    uint64_t direct[N_BASELINE];
    wubu_kernel_spec_t *spec;
    const wubu_kernel_row_t *sel;
    int i;

    spec = wubu_kernel_spec_create();
    CHECK(spec != NULL);
    if (!spec) return 1;

    /* Register the baseline families. */
    for (i = 0; i < N_BASELINE; ++i) {
        CHECK(wubu_kernel_spec_register(spec, &baseline_rows[i]) == 1);
        before[i] = wubu_kernel_spec_identity(spec, baseline_rows[i].family);
        direct[i] = wubu_kernel_body_identity(&baseline_rows[i].body);
        CHECK(before[i] != 0ULL);
        CHECK(before[i] == direct[i]);  /* registry identity == descriptor identity */
    }
    CHECK(wubu_kernel_spec_count(spec) == (size_t)N_BASELINE);

    /* Selection correctness before adding the new row. */
    for (i = 0; i < N_BASELINE; ++i) {
        sel = wubu_kernel_spec_select(spec, baseline_rows[i].family);
        CHECK(sel != NULL);
        if (sel) {
            CHECK(strcmp(sel->family, baseline_rows[i].family) == 0);
            CHECK(body_equal(&sel->body, &baseline_rows[i].body));
        }
    }

    /* Identity is deterministic (pure function). */
    CHECK(wubu_kernel_body_identity(&baseline_rows[0].body) == before[0]);
    CHECK(wubu_kernel_spec_identity(spec, "llama") == before[0]);

    /* Distinct descriptors hash distinctly; unknown family is "not found". */
    {
        wubu_kernel_body_t tweaked = baseline_rows[0].body;
        tweaked.window_tokens = 128;
        CHECK(wubu_kernel_body_identity(&tweaked) != before[0]);
    }
    CHECK(wubu_kernel_spec_select(spec, "not-a-family") == NULL);
    CHECK(wubu_kernel_spec_identity(spec, "not-a-family") == 0ULL);

    /* ---- THE GATE: add a new family row ---- */
    CHECK(wubu_kernel_spec_register(spec, &new_row) == 1);
    CHECK(wubu_kernel_spec_count(spec) == (size_t)N_BASELINE + 1);

    /* Existing kernel-body identities must be byte-identical. */
    for (i = 0; i < N_BASELINE; ++i)
        after[i] = wubu_kernel_spec_identity(spec, baseline_rows[i].family);
    CHECK(memcmp(before, after, sizeof(before)) == 0);
    for (i = 0; i < N_BASELINE; ++i) {
        CHECK(after[i] == direct[i]);  /* unchanged and still descriptor-pure */
    }

    /* Selection still correct for all pre-existing families. */
    for (i = 0; i < N_BASELINE; ++i) {
        sel = wubu_kernel_spec_select(spec, baseline_rows[i].family);
        CHECK(sel != NULL);
        if (sel) {
            CHECK(strcmp(sel->family, baseline_rows[i].family) == 0);
            CHECK(body_equal(&sel->body, &baseline_rows[i].body));
            CHECK(wubu_kernel_spec_identity(spec, sel->family) ==
                  wubu_kernel_body_identity(&sel->body));
        }
    }

    /* The newly added family selects correctly and owns its own identity. */
    sel = wubu_kernel_spec_select(spec, "deepseek");
    CHECK(sel != NULL);
    if (sel) {
        CHECK(body_equal(&sel->body, &new_row.body));
        CHECK(wubu_kernel_spec_identity(spec, "deepseek") ==
              wubu_kernel_body_identity(&new_row.body));
        CHECK(wubu_kernel_spec_identity(spec, "deepseek") != 0ULL);
    }

    wubu_kernel_spec_free(spec);

    if (failures == 0) printf("test_kernel_identity: PASS\n");
    else fprintf(stderr, "test_kernel_identity: FAIL (%d)\n", failures);
    return failures ? 1 : 0;
}
