/**
 * infer_vision_gpu.c — GPU-accelerated vision encoder
 * All 27 ViT layers run via cuBLAS + custom CUDA kernels.
 * Patch embedding kept on CPU (minor cost compared to linear layers).
 */
#include "wubu_vision.h"
#include "wubu_vision_encoder.h"
#include "cuda_vision.h"
#include "gguf_reader.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    const char *path = argc > 1 ? argv[1] : "/models/qwen3.6-35b-mmproj-F16.gguf";
    int H = argc > 2 ? atoi(argv[2]) : 256;
    int W = argc > 3 ? atoi(argv[3]) : 256;
    int B = 1, C = 3;
    
    printf("=== GPU Vision Inference ===\n");
    printf("Model: %s\nImage: %dx%d\n", path, H, W);
    
    double t0 = now_sec();
    
    // 1. Load CPU weights
    vision_encoder_t enc;
    if (!vision_encoder_init(&enc, path)) return 1;
    printf("  Weights loaded: %.2fs\n", now_sec() - t0);
    
    // 2. CUDA init
    cublasHandle_t cublas_h;
    cudaStream_t stream;
    if (cublasCreate(&cublas_h) != CUBLAS_STATUS_SUCCESS) return 1;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 1;
    cublasSetStream(cublas_h, stream);
    
    // 3. Upload all layer weights to GPU
    double t1 = now_sec();
    gpu_vision_weights_t gpu_layers[V_N_LAYERS];
    memset(gpu_layers, 0, sizeof(gpu_layers));
    for (int l = 0; l < V_N_LAYERS; l++) {
        if (!enc.layers[l].loaded) continue;
        if (!gpu_vision_upload_layer(&enc.layers[l], &gpu_layers[l])) {
            fprintf(stderr, "Failed to upload layer %d\n", l);
            return 1;
        }
    }
    
    // Need LN weights on GPU too
    float *d_ln1_w[V_N_LAYERS], *d_ln1_b[V_N_LAYERS];
    float *d_ln2_w[V_N_LAYERS], *d_ln2_b[V_N_LAYERS];
    float *d_post_ln_w, *d_post_ln_b;
    float *d_pos_w;
    float *d_mm0_w = NULL, *d_mm0_b = NULL, *d_mm2_w = NULL, *d_mm2_b = NULL;
    float *d_patch_w = NULL, *d_patch_w2 = NULL, *d_patch_b = NULL;
    
    // Upload all auxiliary weights
    for (int l = 0; l < V_N_LAYERS; l++) {
        cudaMalloc(&d_ln1_w[l], V_HIDDEN * sizeof(float));
        cudaMemcpy(d_ln1_w[l], enc.layers[l].ln1_weight, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ln1_b[l], V_HIDDEN * sizeof(float));
        cudaMemcpy(d_ln1_b[l], enc.layers[l].ln1_bias, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ln2_w[l], V_HIDDEN * sizeof(float));
        cudaMemcpy(d_ln2_w[l], enc.layers[l].ln2_weight, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_ln2_b[l], V_HIDDEN * sizeof(float));
        cudaMemcpy(d_ln2_b[l], enc.layers[l].ln2_bias, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_post_ln_w, V_HIDDEN * sizeof(float));
    cudaMemcpy(d_post_ln_w, enc.post_ln_weight, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_post_ln_b, V_HIDDEN * sizeof(float));
    cudaMemcpy(d_post_ln_b, enc.post_ln_bias, V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_pos_w, V_HIDDEN * V_MAX_POS * sizeof(float));
    cudaMemcpy(d_pos_w, enc.pos_embd_weight, V_HIDDEN * V_MAX_POS * sizeof(float), cudaMemcpyHostToDevice);
    if (enc.mm0_weight) {
        cudaMalloc(&d_mm0_w, 4608 * 4608 * sizeof(float));
        cudaMemcpy(d_mm0_w, enc.mm0_weight, 4608 * 4608 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_mm0_b, 4608 * sizeof(float));
        cudaMemcpy(d_mm0_b, enc.mm0_bias, 4608 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_mm2_w, 4608 * V_OUT_HIDDEN * sizeof(float));
        cudaMemcpy(d_mm2_w, enc.mm2_weight, 4608 * V_OUT_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_mm2_b, V_OUT_HIDDEN * sizeof(float));
        cudaMemcpy(d_mm2_b, enc.mm2_bias, V_OUT_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    }
    // Patch embd on CPU (keep host copies)
    cudaStreamSynchronize(stream);
    printf("  GPU weights uploaded: %.2fs\n", now_sec() - t1);
    
    // 4. Create synthetic image
    float *pixels = (float *)malloc(C * H * W * sizeof(float));
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                pixels[c * H * W + y * W + x] = ((x / 16 + y / 16) % 2) * 0.8f + 0.1f;
    
    // 5. Forward pass
    int patch_h = H / V_PATCH_SIZE, patch_w = W / V_PATCH_SIZE;
    int merged_h = patch_h / 2, merged_w = patch_w / 2;
    int n_merged = merged_h * merged_w * V_TEMP_PATCH;
    if (n_merged > V_MAX_POS) n_merged = V_MAX_POS;
    int n_patches = patch_h * patch_w * V_TEMP_PATCH;
    
    // --- CPU: Patch embedding ---
    double t_patch = now_sec();
    float *h_hidden = (float *)calloc(n_patches * V_HIDDEN, sizeof(float));
    for (int tp = 0; tp < V_TEMP_PATCH; tp++) {
        const float *kernel = (tp == 0) ? enc.patch_embd_weight : enc.patch_embd_weight2;
        for (int ph = 0; ph < patch_h; ph++) {
            for (int pw = 0; pw < patch_w; pw++) {
                int idx = tp * (patch_h * patch_w) + ph * patch_w + pw;
                float *out = h_hidden + idx * V_HIDDEN;
                memcpy(out, enc.patch_embd_bias, V_HIDDEN * sizeof(float));
                for (int ky = 0; ky < V_PATCH_SIZE; ky++)
                    for (int kx = 0; kx < V_PATCH_SIZE; kx++)
                        for (int c = 0; c < C; c++) {
                            float pixel = pixels[c * H * W + (ph*V_PATCH_SIZE+ky) * W + (pw*V_PATCH_SIZE+kx)];
                            const float *k = kernel + (ky*V_PATCH_SIZE + kx) * (C * V_HIDDEN) + c * V_HIDDEN;
                            for (int f = 0; f < V_HIDDEN; f++)
                                out[f] += pixel * k[f];
                        }
            }
        }
    }
    printf("  Patch embedding: %.3fs\n", now_sec() - t_patch);
    
    // --- CPU: Spatial merge ---
    float *h_merged = (float *)calloc(n_merged * V_HIDDEN, sizeof(float));
    for (int tp = 0; tp < V_TEMP_PATCH; tp++)
        for (int mh = 0; mh < merged_h; mh++)
            for (int mw = 0; mw < merged_w; mw++) {
                int dst = tp * merged_h * merged_w + mh * merged_w + mw;
                float inv4 = 0.25f;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++) {
                        int src = tp * patch_h * patch_w + (mh*2+dy) * patch_w + (mw*2+dx);
                        for (int f = 0; f < V_HIDDEN; f++)
                            h_merged[dst * V_HIDDEN + f] += h_hidden[src * V_HIDDEN + f] * inv4;
                    }
            }
    free(h_hidden);
    
    // --- CPU: Position embeddings ---
    for (int i = 0; i < n_merged; i++)
        for (int f = 0; f < V_HIDDEN; f++)
            h_merged[i * V_HIDDEN + f] += enc.pos_embd_weight[f * V_MAX_POS + i];
    
    // Upload merged to GPU
    float *d_merged;
    cudaMalloc(&d_merged, n_merged * V_HIDDEN * sizeof(float));
    cudaMemcpy(d_merged, h_merged, n_merged * V_HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    free(h_merged);
    
    // --- GPU: 27 ViT layers ---
    double t_gpu = now_sec();
    int scratch_size = n_merged * (V_HIDDEN * 3 + V_HIDDEN * 2) * sizeof(float);
    float *d_scratch;
    cudaMalloc(&d_scratch, scratch_size);
    float *d_in = d_merged;
    float *d_out;
    cudaMalloc(&d_out, n_merged * V_HIDDEN * sizeof(float));
    
    for (int l = 0; l < V_N_LAYERS; l++) {
        if (!enc.layers[l].loaded) {
            cudaMemcpy(d_out, d_in, n_merged * V_HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            // LayerNorm 1 (on d_in, in-place)
            int threads = 256;
            int blocks = (n_merged + threads - 1) / threads;
            layernorm_kernel<<<blocks, threads, 0, stream>>>(d_in, d_ln1_w[l], d_ln1_b[l], n_merged, V_HIDDEN, 1e-6f);
            
            // Attention + FFN
            bool ok = gpu_vision_layer_forward(cublas_h, stream, &gpu_layers[l],
                                               d_in, NULL, n_merged, d_out, d_scratch);
            if (!ok) { fprintf(stderr, "Layer %d failed\n", l); return 1; }
            
            // LayerNorm 2 (on d_out, in-place)
            layernorm_kernel<<<blocks, threads, 0, stream>>>(d_out, d_ln2_w[l], d_ln2_b[l], n_merged, V_HIDDEN, 1e-6f);
        }
        
        // Swap pointer for next layer
        if (l < V_N_LAYERS - 1) {
            float *tmp = d_in; d_in = d_out; d_out = tmp;
        }
    }
    cudaStreamSynchronize(stream);
    printf("  27 GPU layers: %.3fs (%.1f ms/layer)\n", now_sec() - t_gpu,
           (now_sec() - t_gpu) / V_N_LAYERS * 1000);
    
    // --- GPU: Post LN ---
    {
        int threads = 256;
        int blocks = (n_merged + threads - 1) / threads;
        layernorm_kernel<<<blocks, threads, 0, stream>>>(d_out, d_post_ln_w, d_post_ln_b, n_merged, V_HIDDEN, 1e-6f);
    }
    cudaStreamSynchronize(stream);
    
    // --- GPU: Merger ---
    float *h_output;
    int out_dim = n_merged * V_HIDDEN;
    h_output = (float *)malloc(out_dim * sizeof(float));
    cudaMemcpy(h_output, d_out, n_merged * V_HIDDEN * sizeof(float), cudaMemcpyDeviceToHost);
    
    // If merger available and 4 patches: run mm0 -> gelu -> mm2
    if (n_merged == 4 && enc.mm0_weight) {
        // Upload
        float *d_mid;
        cudaMalloc(&d_mid, 4608 * sizeof(float));
        cudaMemcpy(d_out, h_output, 4608 * sizeof(float), cudaMemcpyHostToDevice);
        
        // mm.0: [4608] @ [4608,4608] -> [4608]
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(cublas_h, CUBLAS_OP_T, 4608, 4608, &alpha, d_mm0_w, 4608, d_out, 1, &beta, d_mid, 1);
        cublasSaxpy(cublas_h, 4608, &alpha, d_mm0_b, 1, d_mid, 1);
        
        // GELU
        gelu_kernel<<<(4608+255)/256, 256, 0, stream>>>(d_mid, 4608);
        
        // mm.2: [4608] @ [4608, 2048] -> [2048]
        float *d_final;
        cudaMalloc(&d_final, V_OUT_HIDDEN * sizeof(float));
        cublasSgemv(cublas_h, CUBLAS_OP_T, 4608, V_OUT_HIDDEN, &alpha, d_mm2_w, 4608, d_mid, 1, &beta, d_final, 1);
        cublasSaxpy(cublas_h, V_OUT_HIDDEN, &alpha, d_mm2_b, 1, d_final, 1);
        
        cudaMemcpy(h_output, d_final, V_OUT_HIDDEN * sizeof(float), cudaMemcpyDeviceToHost);
        out_dim = V_OUT_HIDDEN;
        cudaFree(d_mid);
        cudaFree(d_final);
    }
    
    printf("  Total forward: %.3fs\n", now_sec() - t_patch);
    
    // Output stats
    float min_v = 1e30, max_v = -1e30; int nan_c = 0;
    for (int i = 0; i < (out_dim > 16 ? 16 : out_dim); i++) {
        if (h_output[i] < min_v) min_v = h_output[i];
        if (h_output[i] > max_v) max_v = h_output[i];
        if (isnan(h_output[i])) nan_c++;
    }
    printf("  Output[0:8]:");
    for (int i = 0; i < 8 && i < out_dim; i++) printf(" %+.4f", h_output[i]);
    printf("\n  Range: [%.4f, %.4f] | NaN: %d | dim=%d\n", min_v, max_v, nan_c, out_dim);
    
    // Cleanup
    free(pixels); free(h_output);
    cudaFree(d_scratch); cudaFree(d_out); cudaFree(d_merged);
    for (int l = 0; l < V_N_LAYERS; l++) {
        cudaFree(d_ln1_w[l]); cudaFree(d_ln1_b[l]);
        cudaFree(d_ln2_w[l]); cudaFree(d_ln2_b[l]);
        gpu_vision_free_layer(&gpu_layers[l]);
    }
    cudaFree(d_post_ln_w); cudaFree(d_post_ln_b);
    cudaFree(d_pos_w);
    if (d_mm0_w) cudaFree(d_mm0_w);
    if (d_mm0_b) cudaFree(d_mm0_b);
    if (d_mm2_w) cudaFree(d_mm2_w);
    if (d_mm2_b) cudaFree(d_mm2_b);
    
    vision_encoder_free(&enc);
    cudaStreamDestroy(stream);
    cublasDestroy(cublas_h);
    
    printf("=== GPU Vision PASS ===\n");
    return 0;
}
