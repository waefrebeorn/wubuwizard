/**
 * test_vm_init.c — Quick test for moondream weight loading.
 * Build: gcc -O3 -I include -o test_vm_init tools/test_vm_init.c src/wubu_vision_moondream.o -lm -ljson-c
 * Usage: ./test_vm_init
 */
#include "wubu_vision_moondream.h"
#include <stdio.h>

int main(void) {
    vm_state_t state;
    if (vm_init(&state, "data/moondream3_vision_weights.bin",
                "data/moondream3_vision_index.json")) {
        printf("vm_init OK! loaded=%d\n", state.loaded);
        printf("patch_emb_weight[0]=%f\n", state.w.patch_emb_weight[0]);
        printf("patch_emb_bias[0]=%f\n", state.w.patch_emb_bias[0]);
        printf("blocks[0].ln1_weight[0]=%f\n", state.w.blocks[0].ln1_weight[0]);
        printf("blocks[0].attn_qkv_weight[0]=%f\n", state.w.blocks[0].attn_qkv_weight[0]);
        printf("blocks[26].mlp_fc2_bias[0]=%f\n", state.w.blocks[26].mlp_fc2_bias[0]);
        printf("proj_mlp_fc1_weight[0]=%f\n", state.w.proj_mlp_fc1_weight[0]);
        printf("proj_mlp_fc2_weight[0]=%f\n", state.w.proj_mlp_fc2_weight[0]);
        printf("pos_emb[0]=%f\n", state.w.pos_emb[0]);
        vm_free(&state);
        return 0;
    } else {
        printf("vm_init FAILED\n");
        return 1;
    }
}
