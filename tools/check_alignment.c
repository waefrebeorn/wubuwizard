/**
 * check_alignment.c — Verify cache-line alignment of GGUF weight tensors.
 *
 * DDR4 optimal burst is 64 bytes (one cache line). Tensors not 64-byte
 * aligned cause split loads — 2× latency per misaligned access.
 *
 * Usage: ./check_alignment [model.gguf]
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#define MODEL_PATH "/home/wubu2/models/qwen3.6-35b-a3b-UD-IQ2_M.gguf"

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : MODEL_PATH;
    
    gguf_ctx *ctx = gguf_open(path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", path); return 1; }
    
    // Buffer the data blob
    if (!gguf_buffer_data(ctx)) {
        fprintf(stderr, "Failed to buffer data\n");
        gguf_close(ctx);
        return 1;
    }
    
    uintptr_t blob_base = (uintptr_t)ctx->data_blob;
    printf("=== Alignment Check: %s ===\n\n", path);
    printf("Data blob base address: 0x%" PRIxPTR "\n", blob_base);
    printf("Blob 64-byte aligned:   %s\n", (blob_base & 63) == 0 ? "YES ✅" : "NO ❌");
    printf("Blob 32-byte aligned:   %s\n", (blob_base & 31) == 0 ? "YES ✅" : "NO ❌");
    printf("\n");
    
    // Check key tensors
    const char *key_tensors[] = {
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_exps.weight",
        "blk.0.ffn_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.0.ffn_gate_shexp.weight",
        "blk.20.ffn_gate_exps.weight",
        "blk.39.ffn_gate_exps.weight",
        NULL
    };
    
    int misaligned_64 = 0;
    int misaligned_32 = 0;
    int total = 0;
    
    printf("%-45s %-12s %-12s  %s\n", "Tensor", "Offset", "Align 64", "Size");
    printf("%-45s %-12s %-12s  %s\n", "-----", "------", "--------", "----");
    
    for (int i = 0; key_tensors[i]; i++) {
        gguf_tensor_info *t = gguf_find_tensor(ctx, key_tensors[i]);
        if (!t) { printf("%-45s %-12s %-12s\n", key_tensors[i], "NOT FOUND", ""); continue; }
        total++;
        
        uintptr_t offset = blob_base + t->data_offset;
        int64_t n_elems = 1;
        for (int d = 0; d < t->n_dims; d++) n_elems *= t->dims[d];
        int64_t rsz = gguf_raw_size(t->ggml_type, n_elems);
        const char *align64 = (offset & 63) == 0 ? "✅" : "❌";
        const char *align32 = (offset & 31) == 0 ? "✅" : "❌";
        
        if ((offset & 63) != 0) misaligned_64++;
        if ((offset & 31) != 0) misaligned_32++;
        
        printf("%-45s 0x%010" PRIxPTR "  %-12s  %s  %lld bytes\n",
               key_tensors[i], offset, align64, align32,
               (long long)rsz);
    }
    
    printf("\n--- Summary ---\n");
    printf("Tensors checked: %d\n", total);
    printf("64-byte aligned: %d/%d (%d%%)\n",
           total - misaligned_64, total,
           (total - misaligned_64) * 100 / total);
    printf("32-byte aligned: %d/%d (%d%%)\n",
           total - misaligned_32, total,
           (total - misaligned_32) * 100 / total);
    
    if (misaligned_64 > 0) {
        printf("\n⚠️  %d tensors are NOT 64-byte aligned — possible split-load penalty\n", misaligned_64);
    }
    if ((blob_base & 63) != 0) {
        printf("💡 Fix: posix_memalign(&blob, 64, size) for data_blob allocation\n");
    }
    
    gguf_close(ctx);
    return misaligned_64;
}
