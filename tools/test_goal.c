/* test_goal.c — gate for LP-01 (persistent goal + evidence-first judge). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "wubu_goal.h"

static int verify_ok = 1;

static int fake_verify(const char *goal, const char *evidence, void *ud)
{
    (void)goal; (void)evidence; (void)ud;
    return verify_ok;
}

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void)
{
    const char *path = "C:/Users/eman5/AppData/Local/hermes/cache/scratch/goal_pad_test.goal";

    wubu_goal_t *g = wubu_goal_create("ship the persistent goal pad", 4);
    CHECK(g != NULL);
    CHECK(strcmp(wubu_goal_text(g), "ship the persistent goal pad") == 0);
    CHECK(wubu_goal_turns(g) == 0);
    CHECK(wubu_goal_max_turns(g) == 4);
    CHECK(wubu_goal_status(g) == WUBU_GOAL_ACTIVE);
    CHECK(wubu_goal_should_continue(g) == 1);

    /* 1. incomplete report cannot complete the goal */
    CHECK(wubu_goal_judge(g, "still working", "", fake_verify, NULL) == WUBU_GOAL_CONTINUE);
    CHECK(wubu_goal_turns(g) == 1);

    /* 2. self-report alone is NOT evidence (report isolation) */
    CHECK(wubu_goal_judge(g, "[GOAL_DONE] trust me", "", fake_verify, NULL) == WUBU_GOAL_CONTINUE);
    CHECK(wubu_goal_status(g) == WUBU_GOAL_ACTIVE);

    /* 3. verifier refusal blocks completion even with a receipt */
    verify_ok = 0;
    CHECK(wubu_goal_judge(g, "[GOAL_DONE]", "[PASS] receipt", fake_verify, NULL) == WUBU_GOAL_CONTINUE);
    verify_ok = 1;

    /* 4. separate evidence receipt + verifier passes the gate */
    CHECK(wubu_goal_judge(g, "[GOAL_DONE]", "[PASS] receipt", fake_verify, NULL) == WUBU_GOAL_FINISHED);
    CHECK(wubu_goal_status(g) == WUBU_GOAL_DONE);
    CHECK(wubu_goal_should_continue(g) == 0);
    CHECK(strstr(wubu_goal_continuation_prompt(g), "standing goal") != NULL);
    wubu_goal_free(g);

    /* 5. explicit blocked reason is terminal, with the reason kept */
    g = wubu_goal_create("needs hardware", 4);
    CHECK(wubu_goal_judge(g, "[BLOCKED: no GPU]", "[PASS]", fake_verify, NULL) == WUBU_GOAL_IS_BLOCKED);
    CHECK(wubu_goal_status(g) == WUBU_GOAL_BLOCKED);
    CHECK(strcmp(wubu_goal_note(g), "no GPU") == 0);
    wubu_goal_free(g);

    /* 6. real user message preempts and freezes the loop */
    g = wubu_goal_create("preempt me", 4);
    CHECK(wubu_goal_preempt(g) == 0);
    CHECK(wubu_goal_status(g) == WUBU_GOAL_PAUSED);
    CHECK(wubu_goal_judge(g, "[GOAL_DONE]", "[PASS]", fake_verify, NULL) == WUBU_GOAL_IS_PAUSED);
    CHECK(wubu_goal_turns(g) == 0); /* no turn consumed while paused */
    wubu_goal_free(g);

    /* 7. turn budget exhaustion pauses with a reason */
    g = wubu_goal_create("bounded loop", 2);
    CHECK(wubu_goal_judge(g, "one", "", fake_verify, NULL) == WUBU_GOAL_CONTINUE);
    CHECK(wubu_goal_judge(g, "two", "", fake_verify, NULL) == WUBU_GOAL_IS_PAUSED);
    CHECK(strcmp(wubu_goal_note(g), "turn budget exhausted") == 0);
    wubu_goal_free(g);

    /* 8. sidecar round-trips the versioned state */
    g = wubu_goal_create("survive restart", 6);
    CHECK(wubu_goal_judge(g, "working", "", fake_verify, NULL) == WUBU_GOAL_CONTINUE);
    CHECK(wubu_goal_save(g, path) == 0);
    wubu_goal_t *r = wubu_goal_load(path);
    CHECK(r != NULL);
    if (r) {
        CHECK(strcmp(wubu_goal_text(r), "survive restart") == 0);
        CHECK(wubu_goal_turns(r) == 1);
        CHECK(wubu_goal_max_turns(r) == 6);
        CHECK(wubu_goal_status(r) == WUBU_GOAL_ACTIVE);
        wubu_goal_free(r);
    }
    wubu_goal_free(g);

    /* 9. corrupt / stale sidecar is refused, never half-loaded */
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f) { fputs("BADMAGIC-not-a-goal", f); fclose(f); }
    CHECK(wubu_goal_load(path) == NULL);

    if (failures == 0) printf("test_goal: PASS (9 gates)\n");
    return failures ? 1 : 0;
}
