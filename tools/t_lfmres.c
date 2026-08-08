/* t_lfmres.c — resolve roles for LFM layer 0, print what comes back */
#include "gguf_reader.h"
#include "wubu_gguf_names.h"
#include <stdio.h>
int main(int argc, char **argv) {
    gguf_ctx *ctx = gguf_open(argv[1]);
    wubu_gguf_names_t names;
    wubu_gguf_names_detect(ctx, &names);
    int roles[] = {WUBU_T_ATTN_QKV, WUBU_T_ATTN_Q, WUBU_T_ATTN_K, WUBU_T_ATTN_V,
                   WUBU_T_ATTN_O, WUBU_T_ATTN_GATE, WUBU_T_SSM_OUT, WUBU_T_ATTN_NORM,
                   WUBU_T_POST_ATTN_NORM, WUBU_T_FFN_NORM, WUBU_T_FFN_GATE, WUBU_T_FFN_UP, WUBU_T_FFN_DOWN};
    const char *rnames[] = {"ATTN_QKV","ATTN_Q","ATTN_K","ATTN_V","ATTN_O","ATTN_GATE",
        "SSM_OUT","ATTN_NORM","POST_ATTN_NORM","FFN_NORM","FFN_GATE","FFN_UP","FFN_DOWN"};
    for (int i = 0; i < 13; i++) {
        gguf_tensor_info *t = wubu_gguf_find(ctx, 0, roles[i]);
        printf("%-15s -> %s\n", rnames[i], t ? t->name : "NULL");
    }
    return 0;
}
