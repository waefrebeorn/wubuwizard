/* tok_probe.c -- print token IDs for the prompt and "". */
#define _GNU_SOURCE
#include "wubu_sd_clip.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* need the bpe internals — expose via the clip header if possible */
int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_clip_t *c = wubu_sd_clip_load(g);
    /* use the public encode and check the CONTEXT directly: how many
     * tokens are non-EOS-padding? For a real prompt, positions after the
     * text should be <49407> (EOS) padding. */
    float ctx[77*768], pool[768], ectx[77*768], epool[768];
    wubu_sd_clip_encode(c, "a serene anime landscape, soft morning light, detailed", ctx, pool);
    wubu_sd_clip_encode(c, "", ectx, epool);
    /* token boundaries: the embedding changes abruptly at token boundaries.
     * Find where the real tokens end by checking context row differences. */
    printf("cond ctx rows 0..9 first3: \n");
    for (int t = 0; t < 10; t++)
        printf("  t=%2d: %+.4f %+.4f %+.4f\n", t, ctx[t*768+0], ctx[t*768+1], ctx[t*768+2]);
    printf("empty ctx rows 0..9 first3: \n");
    for (int t = 0; t < 10; t++)
        printf("  t=%2d: %+.4f %+.4f %+.4f\n", t, ectx[t*768+0], ectx[t*768+1], ectx[t*768+2]);
    return 0;
}
