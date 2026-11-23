/* test_eg34_logits.c -- EG-34 gate: typical-p + watermark logits processors. */
#include <stdio.h>
#include <math.h>

#include "wubu_logits_proc.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    wubu_logits_proc_t *p = wubu_logits_proc_create(0.5f, 0.0f, 0x1234ULL);
    float logits[4] = {2.0f, 1.0f, 0.5f, 0.0f};
    int kept;
    CHECK(p != NULL);

    /* Typical-p 0.5 keeps only the highest-surprisal-mass token. */
    kept = wubu_logits_proc_apply(p, logits, 4, 0);
    CHECK(kept == 1);
    CHECK(logits[0] == 2.0f);
    CHECK(logits[1] < -1e30f && logits[2] < -1e30f && logits[3] < -1e30f);
    printf("EG-34 typical_p=0.5 kept=%d of 4\n", kept);

    /* Keyed green-list watermark biases only green tokens. */
    {
        wubu_logits_proc_t *w = wubu_logits_proc_create(1.0f, 1.0f, 0x1234ULL);
        float lg[8];
        double before = 0.0, after = 0.0, eb = 0.0, er = 0.0;
        int greens = 0;
        CHECK(w != NULL);
        for (int i = 0; i < 8; ++i) {
            lg[i] = 0.0f;
            CHECK(wubu_logits_proc_is_green(w, 3, i) ==
                  wubu_logits_proc_is_green(w, 3, i));
            if (wubu_logits_proc_is_green(w, 3, i)) ++greens;
        }
        CHECK(wubu_logits_proc_apply(w, lg, 8, 3) == 8);
        for (int i = 0; i < 8; ++i) {
            int green = wubu_logits_proc_is_green(w, 3, i);
            CHECK(lg[i] == (green ? 1.0f : 0.0f));
            if (green) { before += 0.125; after += exp(1.0); }
            else after += 1.0;
        }
        eb = before; er = after;
        CHECK(er > eb);
        printf("EG-34 watermark greens=%d/8 bias=1.0 green_mass %.3f -> %.3f\n",
               greens, eb / 8.0, er / (er + 8 - eb));
        CHECK(wubu_logits_proc_is_green(w, -1, 0) == -1);
        CHECK(wubu_logits_proc_create(0.0f, 1.0f, 1) == NULL);
        CHECK(wubu_logits_proc_create(1.5f, 1.0f, 1) == NULL);
        CHECK(wubu_logits_proc_create(1.0f, -1.0f, 1) == NULL);
        CHECK(wubu_logits_proc_apply(NULL, lg, 8, 0) == -1);
        CHECK(wubu_logits_proc_apply(w, lg, 0, 0) == -1);
        wubu_logits_proc_free(w);
    }

    wubu_logits_proc_free(p);
    if (failures) { fprintf(stderr, "test_eg34_logits: FAIL (%d)\n", failures); return 1; }
    printf("test_eg34_logits: PASS\n");
    return 0;
}
