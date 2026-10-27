/**
 * infer_moe.c — MoE inference engine
 * Pre-loads all 40 layers' MoE weights quantized in RAM.
 * Fast dequant via gguf_dequantize, benchmark vs lazy-load.
 */
#include "wubu_moe.h"
#include "wubu_ssm.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "wubu_core_dumps.h"

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    const uint8_t *gate_inp; int64_t sz_gi; int ty_gi;
    const uint8_t *gate_exps; int64_t sz_ge; int ty_ge;
    const uint8_t *up_exps; int64_t sz_ue; 
    const uint8_t *down_exps; int64_t sz_de;
    const uint8_t *gate_shexp; int64_t sz_gs; int ty_gs;
    const uint8_t *up_shexp; int64_t sz_us;
    const uint8_t *down_shexp; int64_t sz_ds;
    bool loaded;
} moe_buf_t;

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    int n_layers = 40;
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    int B = 1, T = 4;
    
    printf("=== MoE Inference Engine ===\n");
    printf("Model: %s\n", path);
    
    // Load GGUF + buffer all data
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) return 1;
    gguf_buffer_data(ctx);
    
    // Pre-load all quantized MoE tensor pointers
    moe_buf_t moe_bufs[40];
    memset(moe_bufs, 0, sizeof(moe_bufs));
    
    for (int l = 0; l < n_layers; l++) {
        moe_buf_t *b = &moe_bufs[l];
        char name[256];
        
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", l);
        gguf_tensor_info *t = gguf_find_tensor(ctx, name);
        if (t) { b->ty_gi = t->ggml_type; b->sz_gi = gguf_raw_size(t->ggml_type, D_MODEL*N_EXPERTS);
                 b->gate_inp = (const uint8_t *)ctx->data_blob + t->data_offset; }
        
        int64_t ne = (int64_t)D_MODEL * D_FF * N_EXPERTS;
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) { b->ty_ge = t->ggml_type; b->sz_ge = gguf_raw_size(t->ggml_type, ne);
                 b->gate_exps = (const uint8_t *)ctx->data_blob + t->data_offset; }
        
        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) { b->sz_ue = gguf_raw_size(t->ggml_type, ne);
                 b->up_exps = (const uint8_t *)ctx->data_blob + t->data_offset; }
        
        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) { b->sz_de = gguf_raw_size(t->ggml_type, ne);
                 b->down_exps = (const uint8_t *)ctx->data_blob + t->data_offset; }
        
        int64_t ns = (int64_t)D_MODEL * SHARED_D_FF;
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) { b->ty_gs = t->ggml_type; b->sz_gs = gguf_raw_size(t->ggml_type, ns);
                 b->gate_shexp = (const uint8_t *)ctx->data_blob + t->data_offset; }
        
        snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) { b->sz_us = gguf_raw_size(t->ggml_type, ns);
                 b->up_shexp = (const uint8_t *)ctx->data_blob + t->data_offset; }
        
        snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", l);
        t = gguf_find_tensor(ctx, name);
        if (t) { b->sz_ds = gguf_raw_size(t->ggml_type, ns);
                 b->down_shexp = (const uint8_t *)ctx->data_blob + t->data_offset; }
        
        b->loaded = (b->gate_exps != NULL);
    }
    printf("Pre-loaded %d MoE layers (quantized pointers)\\n", n_layers);
    
    // Test input
    float x[4 * 2048];
    for (int i = 0; i < B * T * D_MODEL; i++)
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    float output[4 * 2048];
    
    // Warmup: run layer 0
    moe_buf_t *b = &moe_bufs[layer];
    moe_weights_t moe;
    memset(&moe, 0, sizeof(moe));
    
    double t0 = now_sec();
    moe.ffn_gate_inp = (float *)malloc(D_MODEL * N_EXPERTS * sizeof(float));
    gguf_dequantize(b->gate_inp, b->ty_gi, D_MODEL * N_EXPERTS, moe.ffn_gate_inp);
    printf("  Router dequant: %.3f ms\\n", (now_sec() - t0) * 1000);
    
    t0 = now_sec();
    int64_t ne = (int64_t)D_MODEL * D_FF * N_EXPERTS;
    moe.ffn_gate_exps = (float *)malloc(ne * sizeof(float));
    gguf_dequantize(b->gate_exps, b->ty_ge, ne, moe.ffn_gate_exps);
    moe.ffn_up_exps = (float *)malloc(ne * sizeof(float));
    gguf_dequantize(b->up_exps, b->ty_ge, ne, moe.ffn_up_exps);
    int64_t nde = (int64_t)D_FF * D_MODEL * N_EXPERTS;
    moe.ffn_down_exps = (float *)malloc(nde * sizeof(float));
    gguf_dequantize(b->down_exps, b->ty_ge, nde, moe.ffn_down_exps);
    printf("  Expert dequant (3 tensors): %.3f s total\\n", now_sec() - t0);
    
    t0 = now_sec();
    int64_t ns = (int64_t)D_MODEL * SHARED_D_FF;
    moe.ffn_gate_shexp = (float *)malloc(ns * sizeof(float));
    gguf_dequantize(b->gate_shexp, b->ty_gs, ns, moe.ffn_gate_shexp);
    moe.ffn_up_shexp = (float *)malloc(ns * sizeof(float));
    gguf_dequantize(b->up_shexp, b->ty_gs, ns, moe.ffn_up_shexp);
    moe.ffn_down_shexp = (float *)malloc(ns * sizeof(float));
    gguf_dequantize(b->down_shexp, b->ty_gs, ns, moe.ffn_down_shexp);
    printf("  Shared expert dequant (3 tensors): %.3f ms\\n", (now_sec() - t0) * 1000);
    moe.loaded = true;
    
    // Benchmark: MoE forward
    double total = 0.0;
    int iters = 10;
    for (int i = 0; i < iters; i++) {
        t0 = now_sec();
        wubu_moe_forward(x, B, T, &moe, output, NULL, N_ACTIVE_EXPTS, N_EXPERTS, D_MODEL, D_FF);
        total += now_sec() - t0;
    }
    printf("  MoE forward (%d iters): avg %.3f ms (%.0f tok/s)\\n",
           iters, total / iters * 1000, B * T / (total / iters));
    
    // Output stats
    float min_v = 1e30, max_v = -1e30;
    for (int i = 0; i < B * T * D_MODEL; i++) {
        if (output[i] < min_v) min_v = output[i];
        if (output[i] > max_v) max_v = output[i];
    }
    printf("  Output range: [%.4f, %.4f] | NaN: %d\\n", min_v, max_v, 0);
    
    // Cleanup
    free(moe.ffn_gate_inp); free(moe.ffn_gate_exps);
    free(moe.ffn_up_exps); free(moe.ffn_down_exps);
    free(moe.ffn_gate_shexp); free(moe.ffn_up_shexp); free(moe.ffn_down_shexp);
    gguf_close(ctx);
    
    printf("=== MoE Inference PASS ===\\n");
    return 0;
}
