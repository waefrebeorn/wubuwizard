/* t_ffn_res.c — does the resolver find the dense FFN roles for Qwen3.5? */
#include "gguf_reader.h"
#include "wubu_gguf_names.h"
#include <stdio.h>
int main(int argc, char **argv) {
    gguf_ctx *ctx = gguf_open(argv[1]);
    wubu_gguf_names_t names;
    wubu_gguf_names_detect(ctx, &names);
    for (int l = 0; l < 24; l++) {
        gguf_tensor_info *tg = wubu_gguf_find(ctx, l, WUBU_T_FFN_GATE);
        gguf_tensor_info *tu = wubu_gguf_find(ctx, l, WUBU_T_FFN_UP);
        gguf_tensor_info *td = wubu_gguf_find(ctx, l, WUBU_T_FFN_DOWN);
        if (l < 4 || l == 23)
            printf("L%d: gate=%s up=%s down=%s\n",
                l, tg ? tg->name : "NULL", tu ? tu->name : "NULL", td ? td->name : "NULL");
    }
    return 0;
}
