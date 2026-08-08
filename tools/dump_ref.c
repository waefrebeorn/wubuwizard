// dump_ref.c — Dump reference logits and per-layer hidden states from llama.cpp
// Compile: gcc -o /tmp/dump_ref dump_ref.c -I/home/wubu/llama.cpp -I/home/wubu/llama.cpp/common
//   -I/home/wubu/llama.cpp/ggml/include -L/home/wubu/llama.cpp/build/bin
//   -Wl,-rpath,/home/wubu/llama.cpp/build/bin -lggml-cpu -lggml -lllama -lllama-common
//   -lm -lpthread -ldl -lstdc++
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llama.h"

#define D_MODEL 2048

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    
    // Initialize llama backend
    llama_backend_init();
    
    // Model params
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    
    // Load model using new API
    struct llama_model *model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    
    // Create context
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 64;
    ctx_params.embeddings = true;
    
    struct llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    
    // Prepare input from argv[2] or tokenize default "cat"
    const char *prompt = argc > 2 ? argv[2] : "cat";
    llama_token tokens[64];
    int n_tokens = llama_tokenize(
        llama_model_get_vocab(model),
        prompt, (int32_t)strlen(prompt),
        tokens, 64, false, false);
    printf("Prompt: %s (%d tokens)\n", prompt, n_tokens);
    if (n_tokens <= 0) {
        fprintf(stderr, "Failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }
    // Print tokens
    printf("Tokens: ");
    for (int i = 0; i < n_tokens; i++) printf("%d ", tokens[i]);
    printf("\n");
    llama_batch batch = llama_batch_get_one(tokens, n_tokens);
    
    // Run forward pass
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    
    // Get logits
    float *logits = llama_get_logits_ith(ctx, 0);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    printf("llama.cpp output: n_vocab=%d\n", n_vocab);
    
    // Dump final hidden state (before output proj)
    const float *embd = llama_get_embeddings_ith(ctx, 0);
    if (embd) {
        FILE *he = fopen("/tmp/ref_final_hidden.bin", "wb");
        if (he) { fwrite(embd, sizeof(float), D_MODEL, he); fclose(he); }
        printf("  Hidden state saved to /tmp/ref_final_hidden.bin\n");
    }
    
    // Dump logits
    
    // Dump logits
    FILE *f = fopen("/tmp/llama_logits_new.bin", "wb");
    if (f) {
        fwrite(logits, sizeof(float), n_vocab, f);
        fclose(f);
        printf("  Logits saved to /tmp/llama_logits_new.bin (%.1f MB)\n",
               n_vocab * 4.0 / 1024 / 1024);
    }
    
    // Print top-10
    float *cpy = (float *)malloc(n_vocab * sizeof(float));
    memcpy(cpy, logits, n_vocab * sizeof(float));
    printf("  Top-10 logits:\n");
    for (int k = 0; k < 10; k++) {
        float best = -1e30f; int best_idx = -1;
        for (int i = 0; i < n_vocab; i++) {
            if (cpy[i] > best) { best = cpy[i]; best_idx = i; }
        }
        cpy[best_idx] = -1e30f;
        printf("    [%d] val=%.4f\n", best_idx, (double)best);
    }
    free(cpy);
    
    // Dump hidden states from each layer
    // (llama.cpp doesn't easily expose per-layer hidden states via public API,
    //  so we skip this for now)
    
    printf("\nNOTE: Per-layer hidden states from llama.cpp require source modification.\n");
    printf("Skipping per-layer comparison for now.\n");
    
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
