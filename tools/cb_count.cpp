// cb_count.cpp - count cb_eval invocations + dump any named tensors.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <set>
static long g_calls = 0, g_ask = 0;
static std::set<std::string> named;
static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    (void)ud; g_calls++;
    if (ask) { g_ask++; return false; }
    const char * n = ggml_get_name(t);
    if (n && n[0] && named.insert(n).second) {
        fprintf(stderr, "NAMED: %s ne0=%lld ne1=%lld\n", n,
                (long long)(t->ne[0]), (long long)(t->ne[1]));
    }
    return true;
}
int main(int argc, char ** argv) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_load_model_from_file(argv[1], mparams);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 64; cparams.n_batch = 64;
    cparams.cb_eval = cb;
    llama_context * ctx = llama_new_context_with_model(model, cparams);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token tokens[16];
    int n = llama_tokenize(vocab, "Hello", 5, tokens, 16, true, false);
    llama_decode(ctx, llama_batch_get_one(tokens, n));
    llama_decode(ctx, llama_batch_get_one(tokens, n));
    fprintf(stderr, "calls=%ld ask=%ld named=%zu\n", g_calls, g_ask, named.size());
    llama_free(ctx); llama_free_model(model);
    return 0;
}
