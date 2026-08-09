#include "wubu_model.h"
#include "wubu_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, argv[1])) return 1;
    mdl.enable_moe = true;

    wubu_tokenizer_t tok;
    if (!wubu_tokenizer_init(&tok, argv[1])) return 1;

    int ids[8], n = wubu_tokenizer_encode(&tok, "Hello", ids, 8);
    printf("prompt tokens: %d\n", n);
    float *emb = (float *)malloc(1024 * sizeof(float));
    FILE *ef = fopen("data/embeddings_248320_1024.bin.raw", "rb");
    if (ef) { fseek(ef, (long long)ids[0] * 1024 * 4, SEEK_SET);
              fread(emb, 4, 1024, ef); fclose(ef); }
    printf("emb[0..3]=%.4f %.4f %.4f %.4f\n", emb[0], emb[1], emb[2], emb[3]);

    float *logits = (float *)malloc(mdl.vocab_size * sizeof(float));
    mdl.skip_output_proj = false;
    wubu_model_forward_from_embd(&mdl, emb, 1, 1, logits);

    int nan = 0; float mx = -1e30f; int mi = -1;
    for (int i = 0; i < mdl.vocab_size; i++) {
        if (logits[i] != logits[i]) { nan++; continue; }
        if (logits[i] > mx) { mx = logits[i]; mi = i; }
    }
    printf("logits: %d NaN, argmax=%d (%.3f)\n", nan, mi, mx);
    /* top 5 */
    for (int k = 0; k < 5; k++) {
        float mv = -1e30f; int mvi = -1;
        for (int i = 0; i < mdl.vocab_size; i++)
            if (logits[i] > mv) { mv = logits[i]; mvi = i; }
        logits[mvi] = -1e30f;
        char pb[256]; int pc = wubu_tokenizer_decode(&tok, &mvi, 1, pb, 256);
        pb[pc] = 0;
        printf("  top%d: id=%d logit=%.3f piece='%s'\n", k+1, mvi, mv, pb);
    }
    wubu_model_free(&mdl);
    return 0;
}
