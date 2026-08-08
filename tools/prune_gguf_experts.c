/**
 * prune_gguf_experts.c — Prune non-essential expert weights from a GGUF.
 * Keeps only top-N routed experts per layer, creating a much smaller model.
 *
 * For Qwen3.6-35B-A3B: pruning 256→10 experts per layer reduces
 * model from 10.7GB to ~3.3GB. Fits with headroom in 11GB RAM.
 *
 * Does NOT need profiling data — keeps first N experts (indices 0..N-1).
 * For per-layer custom expert selection, use prune_gguf_experts_with_map.
 *
 * Build: gcc -O3 -I include -o prune_gguf_experts tools/prune_gguf_experts.c src/gguf_reader.o -lm
 * Usage: ./prune_gguf_experts input.gguf output.gguf <experts_per_layer>
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

// Copy n bytes verbatim
static int64_t copy_verbatim(FILE *dst, FILE *src, int64_t n, const char *label) {
    char buf[65536];
    int64_t total = 0;
    while (total < n) {
        int64_t chunk = (n - total) > (int64_t)sizeof(buf) ? (int64_t)sizeof(buf) : (n - total);
        size_t nr = fread(buf, 1, chunk, src);
        if (nr <= 0) { fprintf(stderr, "Read error at %s +%lld\n", label, (long long)total); break; }
        fwrite(buf, 1, nr, dst);
        total += nr;
    }
    return total;
}

// Check if tensor name is a routed expert weight
static int is_routed_expert_tensor(const char *name) {
    return (strstr(name, "ffn_gate_exps.weight") || 
            strstr(name, "ffn_up_exps.weight") ||
            strstr(name, "ffn_down_exps.weight")) &&
           !strstr(name, "shexp");
}

// Check if tensor name is a shared expert weight
static int is_shared_expert_tensor(const char *name) {
    return (strstr(name, "ffn_gate_shexp") || 
            strstr(name, "ffn_up_shexp") ||
            strstr(name, "ffn_down_shexp"));
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <top_n>\n", argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];
    int top_n = atoi(argv[3]);
    if (top_n < 1 || top_n > 256) {
        fprintf(stderr, "top_n must be 1-256\n");
        return 1;
    }

    // Open input
    gguf_ctx *ctx = gguf_open(in_path);
    if (!ctx) { fprintf(stderr, "Failed to open %s\n", in_path); return 1; }
    
    // Buffer the data blob for random access
    if (!gguf_buffer_data(ctx)) {
        fprintf(stderr, "Failed to buffer data blob\n");
        gguf_close(ctx);
        return 1;
    }
    
    FILE *fin = fopen(in_path, "rb");
    if (!fin) { gguf_close(ctx); return 1; }
    
    FILE *fout = fopen(out_path, "wb");
    if (!fout) { fclose(fin); gguf_close(ctx); return 1; }

    // ===== 1. Copy header (first 24 bytes) =====
    uint8_t header[24];
    fread(header, 24, 1, fin);
    fwrite(header, 24, 1, fout);
    
    // ===== 2. Read and copy KV metadata =====
    int64_t kv_start = 24;
    int64_t kv_len = ctx->tensors_offset - (uint64_t)kv_start;
    fseek(fin, kv_start, SEEK_SET);
    copy_verbatim(fout, fin, kv_len, "KV metadata");
    
    // ===== 3. Rewrite tensor info =====
    int64_t ti_start = ftell(fout);
    int n_expert_pruned = 0;
    int64_t bytes_saved = 0;
    
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        
        // Write tensor name (length-prefixed string)
        int name_len = (int)strlen(t->name);
        fwrite(&name_len, 4, 1, fout);
        fwrite(t->name, 1, name_len, fout);
        
        // Check if this is a routed expert tensor
        int is_expert = is_routed_expert_tensor(t->name);
        int64_t new_dims[4] = {t->dims[0], t->dims[1], t->dims[2], t->dims[3]};
        
        if (is_expert) {
            // Prune: reduce N_EXPERTS dim from dims[2] to top_n
            int64_t old_n = t->dims[2];
            int64_t new_n = (old_n > top_n) ? top_n : old_n;
            new_dims[2] = new_n;
            
            int64_t old_elems = t->dims[0] * t->dims[1] * t->dims[2] * (t->dims[3] ? t->dims[3] : 1);
            int64_t new_elems = t->dims[0] * t->dims[1] * new_n * (t->dims[3] ? t->dims[3] : 1);
            int64_t old_raw = gguf_raw_size(t->ggml_type, old_elems);
            int64_t new_raw = gguf_raw_size(t->ggml_type, new_elems);
            bytes_saved += old_raw - new_raw;
            n_expert_pruned++;
            
            // Use stdio to printf to avoid ggml type name issues
            printf("  %s: %lld→%lld experts (%lld→%lld bytes)\n",
                   t->name, (long long)old_n, (long long)new_n,
                   (long long)old_raw, (long long)new_raw);
        } else if (is_shared_expert_tensor(t->name)) {
            // Shared experts kept as-is (but need correct data offset)
            printf("  %s: kept (shared expert)\n", t->name);
        }
        
        // Write dimensions
        int n_dims = is_expert ? 3 : t->n_dims;  // expert tensors always have 3 dims (keep or pruned)
        fwrite(&n_dims, 4, 1, fout);
        for (int d = 0; d < n_dims; d++)
            fwrite(&new_dims[d], 8, 1, fout);
        
        // Write type
        fwrite(&t->ggml_type, 4, 1, fout);
        
        // Write placeholder data_offset (will be patched after we calculate positions)
        uint64_t placeholder = 0;
        fwrite(&placeholder, 8, 1, fout);
    }
    
    int64_t ti_end = ftell(fout);
    
    // ===== 4. Calculate new data offsets and write data =====
    // Data blob starts at the next multiple of GGUF alignment (32 or 4096)
    uint32_t alignment = ctx->alignment ? ctx->alignment : 32;
    int64_t data_start = ti_end;
    int64_t pad = (alignment - (data_start % alignment)) % alignment;
    if (pad > 0) {
        uint8_t zeros[4096] = {0};
        fwrite(zeros, 1, pad, fout);
        data_start += pad;
    }
    
    // Now we need to go back and patch the data_offset for all tensors
    // First pass: compute offsets
    int64_t offset = data_start;
    
    // Read the raw tensor info section from output (we wrote placeholders)
    // and patch the data_offset values
    fseek(fout, ti_start, SEEK_SET);
    
    // We need to re-iterate through tensors, computing offsets
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        
        // Skip past name + n_dims + dims + type in the output
        int name_len = (int)strlen(t->name);
        int n_dims = is_routed_expert_tensor(t->name) ? 3 : t->n_dims;
        int64_t entry_size = 4 + name_len + 4 + n_dims * 8 + 4 + 8; // name_len+name+n_dims+dims+type+offset
        int64_t next_entry = ftell(fout) + entry_size;
        
        // Calculate data size
        int is_expert = is_routed_expert_tensor(t->name);
        int64_t elems;
        if (is_expert) {
            int64_t new_n = (t->dims[2] > top_n) ? top_n : t->dims[2];
            elems = t->dims[0] * t->dims[1] * new_n * (t->dims[3] ? t->dims[3] : 1);
        } else {
            elems = t->dims[0] * t->dims[1] * t->dims[2] * (t->dims[3] ? t->dims[3] : 1);
        }
        int64_t raw_size = gguf_raw_size(t->ggml_type, elems);
        
        // Patch data_offset
        fseek(fout, next_entry - 8, SEEK_SET);  // seek to the 8-byte offset field
        fwrite(&offset, 8, 1, fout);
        
        // Advance to next tensor info entry for next iteration
        offset += raw_size;
        fseek(fout, next_entry, SEEK_SET);
    }
    
    // ===== 5. Write tensor data =====
    // Seek to data start in output
    fseek(fout, data_start, SEEK_SET);
    
    // For each tensor, copy data from input blob
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        const uint8_t *blob = (const uint8_t *)ctx->data_blob;
        
        int is_expert = is_routed_expert_tensor(t->name);
        int64_t total_elems = t->dims[0] * t->dims[1] * t->dims[2] * (t->dims[3] ? t->dims[3] : 1);
        int64_t raw_size = gguf_raw_size(t->ggml_type, total_elems);
        
        if (is_expert) {
            // Keep only first top_n experts' data
            int64_t orig_elems = total_elems;
            int64_t per_expert_elems = t->dims[0] * t->dims[1];  // elements per expert
            int64_t per_expert_bytes = gguf_raw_size(t->ggml_type, per_expert_elems);
            int64_t orig_n = t->dims[2];
            int64_t keep_n = (orig_n > top_n) ? top_n : orig_n;
            
            for (int e = 0; e < keep_n; e++) {
                const uint8_t *src = blob + t->data_offset + e * per_expert_bytes;
                fwrite(src, 1, per_expert_bytes, fout);
            }
            
            if (orig_n != keep_n) {
                printf("  Pruned %s: kept %d/%lld experts\n", t->name, keep_n, (long long)orig_n);
            }
        } else {
            // Copy entire tensor
            const uint8_t *src = blob + t->data_offset;
            fwrite(src, 1, raw_size, fout);
        }
    }
    
    int64_t file_size = ftell(fout);
    double gb_saved = (double)bytes_saved / (1024.0 * 1024.0 * 1024.0);
    printf("\n=== GGUF Pruning Complete ===\n");
    printf("  Experts per layer: %d\n", top_n);
    printf("  Expert tensors pruned: %d\n", n_expert_pruned);
    printf("  Bytes saved: %.2f GB\n", gb_saved);
    printf("  Output file: %s (%.2f GB)\n", out_path, (double)file_size / 1e9);
    
    fclose(fout);
    fclose(fin);
    gguf_close(ctx);
    return 0;
}
