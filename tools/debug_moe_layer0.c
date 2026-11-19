#include "wubu_model.h"
#include "wubu_ssm.h"
#include "wubu_moe.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(void) {
    wubu_model_t mdl;
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) return 1;
    mdl.enable_moe = true;
    
    int D = D_MODEL;
    float *x = (float *)malloc(D * sizeof(float));
    if (mdl.use_embedding_file) {
        FILE *f = fopen("data/qwen36_embeddings_c.bin.raw", "rb");
        fseek(f, 248044LL * D * sizeof(float), SEEK_SET);
        fread(x, sizeof(float), D, f);
        fclose(f);
    }
    
    // MANUAL layer 0 forward to dump intermediate states
    float *normed = (float *)malloc(D * sizeof(float));
    float *attn_out = (float *)malloc(D * sizeof(float));
    float *normed2 = (float *)malloc(D * sizeof(float));
    float *ffn_out = (float *)malloc(D * sizeof(float));
    
    wubu_layer_t *layer = &mdl.layers[0];
    
    // Pre-attention RMS Norm
    wubu_rms_norm(1, 1, D, x, layer->attn_norm_weight, 1e-6f, normed);
    
    // SSM forward
    float *ssm_state = mdl.ssm_states + 0 * SSM_V_HEADS * SSM_D_STATE * SSM_D_STATE;
    float *conv_state = mdl.conv_states + 0 * (CONV_KERNEL - 1) * CONV_DIM;
    wubu_ssm_forward(normed, 1, 1, &layer->ssm, ssm_state, conv_state, attn_out, NULL, NULL, NULL);
    
    // Residual
    for (int i = 0; i < D; i++) x[i] += attn_out[i];
    
    // Save pre-MoE hidden state
    FILE *f = fopen("/tmp/our_pre_moe_layer0.bin", "wb");
    fwrite(x, sizeof(float), D, f); fclose(f);
    
    // Post-attention RMS Norm (MoE input)
    wubu_rms_norm(1, 1, D, x, layer->post_attn_norm_weight, 1e-6f, normed2);
    
    // Save MoE input
    f = fopen("/tmp/our_moe_input_layer0.bin", "wb");
    fwrite(normed2, sizeof(float), D, f); fclose(f);
    
    // MoE forward
    wubu_moe_forward(normed2, 1, 1, &layer->moe, ffn_out, NULL);
    
    // Save MoE output
    f = fopen("/tmp/our_moe_output_layer0.bin", "wb");
    fwrite(ffn_out, sizeof(float), D, f); fclose(f);
    
    // Also dump router scores
    float *scores = (float *)malloc(N_EXPERTS * sizeof(float));
    wubu_moe_router(normed2, 1, 1, layer->moe.ffn_gate_inp, scores);
    
    // Save scores
    f = fopen("/tmp/our_router_scores_layer0.bin", "wb");
    fwrite(scores, sizeof(float), N_EXPERTS, f); fclose(f);
    
    // Print top-8 experts from our scores
    int top8_idx[8]; float top8_val[8];
    for (int i = 0; i < 8; i++) top8_val[i] = -1e30f;
    for (int e = 0; e < N_EXPERTS; e++) {
        float v = scores[e];
        for (int k = 0; k < 8; k++) {
            if (v > top8_val[k]) {
                for (int j = 7; j > k; j--) { top8_val[j] = top8_val[j-1]; top8_idx[j] = top8_idx[j-1]; }
                top8_val[k] = v; top8_idx[k] = e;
                break;
            }
        }
    }
    printf("Our top-8 experts (by raw score):\n");
    for (int k = 0; k < 8; k++)
        printf("  [%d] expert %d score=%.6f\n", k, top8_idx[k], top8_val[k]);
    
    free(scores);
    free(ffn_out);
    free(normed2);
    free(attn_out);
    free(normed);
    free(x);
    wubu_model_free(&mdl);
    return 0;
}
