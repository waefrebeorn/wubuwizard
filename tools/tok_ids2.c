/* tok_ids2.c -- compile the clip .c INLINE to access bpe_encode internals. */
#define WUBU_HOSTED
#include "../src/wubu_sd_clip.c"  /* pull in the static internals */

int main(int argc, char **argv) {
    gguf_ctx *g = gguf_open(argv[1]);
    wubu_sd_clip_t *c = wubu_sd_clip_load(g);
    if (!c) { printf("load failed\n"); return 1; }
    const char *prompts[] = {
        "a serene anime landscape, soft morning light, detailed",
        "a serene anime landscape",
        "cat, dog",
        "hello, world",
        "a photo of a cat",
    };
    for (int p = 0; p < 5; p++) {
        int ids[77];
        int n = bpe_encode(c->bpe, prompts[p], ids);
        printf("'%s' -> %d tokens: [", prompts[p], n);
        for (int i = 0; i < n && i < 20; i++) printf("%d%s", ids[i], i+1<20 && i+1<n ? " " : "");
        printf("]\n");
    }
    return 0;
}
