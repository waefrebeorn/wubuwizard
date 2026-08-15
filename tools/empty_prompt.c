/* empty_prompt.c -- CLIP encode of "" vs a normal prompt: check stats.
 * The uncond embedding must look like a real text embedding (not zeros,
 * not garbage). sd.cpp uses the same "" encoding for CFG. */
#define WUBU_HOSTED
#include "wubu_sd_clip.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_clip_t *c = wubu_sd_clip_load(g);
    float ctx[77*768], pool[768];
    float ectx[77*768], epool[768];
    if (wubu_sd_clip_encode(c, "a serene anime landscape", ctx, pool) != 0) { printf("encode failed\n"); return 1; }
    if (wubu_sd_clip_encode(c, "", ectx, epool) != 0) { printf("empty encode failed\n"); return 1; }
    double m1 = 0, m2 = 0, r1 = 0, r2 = 0;
    for (int i = 0; i < 77*768; i++) { m1 += ctx[i]; m2 += ectx[i]; r1 += ctx[i]*ctx[i]; r2 += ectx[i]*ectx[i]; }
    m1 /= 77*768; m2 /= 77*768;
    r1 = sqrt(r1/(77*768) - m1*m1); r2 = sqrt(r2/(77*768) - m2*m2);
    printf("cond  : mean=%+.5f rms=%+.5f\n", m1, r1);
    printf("empty : mean=%+.5f rms=%+.5f\n", m2, r2);
    /* are they identical? (if empty encoding fails, ectx may equal ctx or zeros) */
    double maxd = 0;
    for (int i = 0; i < 77*768; i++) { double d = fabs(ctx[i]-ectx[i]); if (d > maxd) maxd = d; }
    printf("max|cond-empty| = %g %s\n", maxd, maxd < 1e-6 ? "IDENTICAL (BUG!)" : "different (ok)");
    /* tokenize check: what tokens does "" produce? print via a direct call */
    return 0;
}
