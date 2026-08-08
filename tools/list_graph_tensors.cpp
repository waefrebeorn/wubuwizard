// list_graph_tensors.cpp - enumerate llama.cpp graph tensor names via cb_eval.
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <set>

static std::set<std::string> seen;
static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    (void)ud;
    if (ask) return false;
    const char * n = ggml_get_name(t);
    if (n && n[0] && seen.insert(n).second)
        fprintf(stderr, "TENSOR: %s\n", n);
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
    llama_free(ctx); llama_free_model(model);
    return 0;
}
