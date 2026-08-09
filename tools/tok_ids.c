/* tok_ids.c -- dump our BPE token IDs for the test prompt. */
#define _GNU_SOURCE
#include "wubu_sd_clip.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_clip_t *c = wubu_sd_clip_load(g);
    /* use a prompt with punctuation to expose the split difference */
    const char *prompts[] = {
        "a serene anime landscape, soft morning light, detailed",
        "a serene anime landscape",
        "cat, dog",
        "hello, world",
    };
    for (int p = 0; p < 4; p++) {
        float ctx[77*768], pool[768];
        wubu_sd_clip_encode(c, prompts[p], ctx, pool);
        /* token boundary detection: rows that differ from the NEXT row are
         * token changes; but simpler — we can't get IDs from the public API.
         * Instead, replicate bpe_encode here by including the internals... */
        printf("prompt[%d]='%s' encoded (can't print ids via public API)\n", p, prompts[p]);
    }
    return 0;
}
