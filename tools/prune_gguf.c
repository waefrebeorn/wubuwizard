/**
 * prune_gguf.c — Prune non-essential experts from GGUF model.
 * Uses gguf_reader to parse the input, then writes a pruned output.
 *
 * Usage: ./prune_gguf model.gguf output.gguf 10
 *
 * Keeps top-N experts (first N by index, 0..N-1 per layer).
 */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>

#define CHUNK 65536

static int is_routed_expert(const char *name) {
    if (strstr(name, "shexp")) return 0;
    return strstr(name, "ffn_gate_exps.weight") ||
           strstr(name, "ffn_up_exps.weight") ||
           strstr(name, "ffn_down_exps.weight");
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <top_n>\n", argv[0]);
        return 1;
    }
    const char *inpath = argv[1];
    const char *outpath = argv[2];
    int top_n = atoi(argv[3]);
    if (top_n < 1 || top_n > 256) {
        fprintf(stderr, "top_n 1-256\n"); return 1;
    }

    // Use gguf_reader to parse header + KV + tensor info
    gguf_ctx *ctx = gguf_open(inpath);
    if (!ctx) { fprintf(stderr, "gguf_open failed\n"); return 1; }
    fprintf(stderr, "DEBUG: gguf_open OK, ctx=%p\n", (void*)ctx);
    fflush(stderr);
    
    // Buffer data blob for reading (reads entire model into RAM)
    fprintf(stderr, "DEBUG: buffering data blob...\n");
    fflush(stderr);
    if (!gguf_buffer_data(ctx)) {
        fprintf(stderr, "WARNING: data buffering failed (OOM?), continuing with file reads\n");
    }
    fprintf(stderr, "DEBUG: buffer complete\n");
    fflush(stderr);

    FILE *fi = fopen(inpath, "rb");
    FILE *fo = fopen(outpath, "wb");
    if (!fi || !fo) { perror("fopen"); return 1; }

    // Buffer data blob for reading
    gguf_buffer_data(ctx);

    int64_t n = ctx->n_tensors;
    uint32_t align = ctx->alignment ? ctx->alignment : 32;
    
    fprintf(stderr, "ctx=%p n=%lld align=%d\n", (void*)ctx, (long long)n, align);
    fflush(stderr);
    
    fprintf(stderr, "tensors_offset=%llu, data_blob_offset=%llu\n",
            (unsigned long long)ctx->tensors_offset,
            (unsigned long long)ctx->data_blob_offset);
    fflush(stderr);

    // Reopen for writing
    fi = fopen(inpath, "rb");
    fo = fopen(outpath, "wb");
    
    // 1. Write header
    uint32_t v32 = ctx->version;
    uint64_t nt = ctx->n_tensors, nkv = ctx->n_kv;
    fwrite("GGUF", 1, 4, fo);
    fwrite(&v32, 4, 1, fo);
    fwrite(&nt, 8, 1, fo);
    fwrite(&nkv, 8, 1, fo);
    
    // 2. Copy KV metadata verbatim (from file between header and tensor info)
    // Header is 24 bytes. Tensor info starts after KV.
    // We know the data blob offset. Everything between 24 and data_blob_offset
    // is KV + tensor info. We need just the KV part.
    // 
    // The simplest way: read KV entries from the file to find their size,
    // copy them, then write tensor info ourselves.
    // 
    // OR: just open the input with gguf_reader, seek past header, 
    // read KV entries to measure their total size, copy them verbatim.
    // 2. Copy KV metadata verbatim (from file between header and tensor info)
    int64_t kv_bytes = ctx->tensors_offset - 24;  // all bytes between header and tensor info
    
    fseek(fi, 24, SEEK_SET);
    uint8_t *kvdata = (uint8_t *)malloc(kv_bytes);
    fread(kvdata, 1, kv_bytes, fi);
    fwrite(kvdata, 1, kv_bytes, fo);
    free(kvdata);
    
    printf("  KV metadata: %lld bytes\n", (long long)kv_bytes);
    
    // 3. Write new tensor info entries
    // Compute new raw sizes first
    int64_t *new_raw = (int64_t *)calloc(n, sizeof(int64_t));
    int pruned_count = 0;
    int64_t orig_total = 0, new_total = 0;
    
    for (int64_t i = 0; i < n; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        int64_t elems = 1;
        for (int d = 0; d < t->n_dims; d++) elems *= t->dims[d];
        int64_t raw = gguf_raw_size(t->ggml_type, elems);
        
        if (is_routed_expert(t->name)) {
            int64_t per_expert = t->dims[0] * t->dims[1];
            if (t->n_dims >= 3) {
                int64_t orig_n = t->dims[2];
                int64_t keep_n = (orig_n > top_n) ? top_n : orig_n;
                int64_t new_elems = t->dims[0] * t->dims[1] * keep_n;
                new_raw[i] = gguf_raw_size(t->ggml_type, new_elems);
                pruned_count++;
                printf("  Prune %s: %lld→%lld experts\n", t->name,
                       (long long)orig_n, (long long)keep_n);
            } else {
                new_raw[i] = raw;
            }
        } else {
            new_raw[i] = raw;
        }
        orig_total += raw;
        new_total += new_raw[i];
    }
    
    // Write tensor info + data offsets
    int64_t ti_start = ftell(fo);
    uint64_t data_offset = 0;
    
    for (int64_t i = 0; i < n; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        int is_exp = is_routed_expert(t->name);
        
        // Name
        uint64_t nl = (uint64_t)strlen(t->name);
        fwrite(&nl, 8, 1, fo);   // GGUF v3: name_len is uint64
        fwrite(t->name, 1, nl, fo);
        
        // Dims
        if (is_exp && t->n_dims >= 3) {
            int nd = t->n_dims;
            fwrite(&nd, 4, 1, fo);
            fwrite(&t->dims[0], 8, 1, fo);
            fwrite(&t->dims[1], 8, 1, fo);
            int64_t new_d2 = (t->dims[2] > top_n) ? top_n : t->dims[2];
            fwrite(&new_d2, 8, 1, fo);
            if (nd > 3) fwrite(&t->dims[3], 8, 1, fo);
        } else {
            fwrite(&t->n_dims, 4, 1, fo);
            for (int d = 0; d < t->n_dims; d++)
                fwrite(&t->dims[d], 8, 1, fo);
        }
        
        // Type
        fwrite(&t->ggml_type, 4, 1, fo);
        
        // Data offset (relative to data start)
        fwrite(&data_offset, 8, 1, fo);
        data_offset += new_raw[i];
    }
    
    // 4. Alignment padding before data
    int64_t ti_end = ftell(fo);
    int64_t pad = (align - (ti_end % align)) % align;
    int64_t data_start = ti_end + pad;
    if (pad > 0) {
        uint8_t zeros[4096] = {0};
        fwrite(zeros, 1, pad, fo);
    }
    
    // 5. Write data
    const uint8_t *blob = (const uint8_t *)ctx->data_blob;
    for (int64_t i = 0; i < n; i++) {
        gguf_tensor_info *t = &ctx->tensors[i];
        int is_exp = is_routed_expert(t->name);
        
        if (is_exp && t->n_dims >= 3) {
            int64_t per_expert_elems = t->dims[0] * t->dims[1];
            int64_t per_expert_bytes = gguf_raw_size(t->ggml_type, per_expert_elems);
            int64_t keep_n = (t->dims[2] > top_n) ? top_n : t->dims[2];
            
            for (int e = 0; e < keep_n; e++) {
                fwrite(blob + t->data_offset + e * per_expert_bytes, 1, per_expert_bytes, fo);
            }
        } else {
            int64_t elems = 1;
            for (int d = 0; d < t->n_dims; d++) elems *= t->dims[d];
            int64_t raw = gguf_raw_size(t->ggml_type, elems);
            fwrite(blob + t->data_offset, 1, raw, fo);
        }
    }
    
    int64_t file_size = ftell(fo);
    printf("\n=== Done ===\n");
    printf("  Original: %.2f GB\n", (double)orig_total / 1e9);
    printf("  Pruned:   %.2f GB (%.0f%%)\n", (double)new_total / 1e9, 100.0*new_total/orig_total);
    printf("  Saved:    %.2f GB\n", (double)(orig_total - new_total) / 1e9);
    printf("  File:     %s (%.2f GB)\n", outpath, (double)file_size / 1e9);
    printf("  Pruned %d expert tensors\n", pruned_count);
    
    fclose(fo);
    fclose(fi);
    free(new_raw);
    gguf_close(ctx);
    return 0;
}
