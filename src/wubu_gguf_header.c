#include "gguf_reader.h"
#include "wubu_gguf_header.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>

// ========== Read Helpers ==========

static uint64_t read_u64(FILE *f) {
    uint64_t v;
    if (fread(&v, 8, 1, f) != 1) return 0;
    return v;
}

static uint32_t read_u32(FILE *f) {
    uint32_t v;
    if (fread(&v, 4, 1, f) != 1) return 0;
    return v;
}

static int32_t read_i32(FILE *f) {
    int32_t v;
    if (fread(&v, 4, 1, f) != 1) return 0;
    return v;
}

static int64_t read_i64(FILE *f) {
    int64_t v;
    if (fread(&v, 8, 1, f) != 1) return 0;
    return v;
}

static float read_f32(FILE *f) {
    float v;
    if (fread(&v, 4, 1, f) != 1) return 0;
    return v;
}

static void read_str(FILE *f, char *buf, int max_len) {
    uint64_t len = read_u64(f);
    if (len >= (uint64_t)max_len) len = max_len - 1;
    size_t n = fread(buf, 1, len, f);
    buf[n] = '\0';
    // Skip remaining if truncated
    if (len > (uint64_t)n) fseek(f, len - n, SEEK_CUR);
}

// Float16 → Float32 (exported: wubu_weight.c and dequant paths use it)
float gguf_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    if (exp == 0) {
        // Subnormal: value = (-1)^sign * mant/1024 * 2^(-14)
        uint32_t normal_f32 = (sign << 31) | ((1 + 112) << 23) | (mant << 13);
        float normal_val;
        memcpy(&normal_val, &normal_f32, 4);
        if (sign) {
            return normal_val + 6.103515625e-5f;  // 2^(-14), adds because normal_val is negative
        } else {
            return normal_val - 6.103515625e-5f;  // 2^(-14)
        }
    }
    if (exp == 31) {
        // Inf or NaN: propagate to float32
        // FP16: exp=31, mant=0 → Inf, mant!=0 → NaN
        // FP32: exp=255, mant=0 → Inf, mant!=0 → NaN (mant shifted << 13)
        uint32_t f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
        float result;
        memcpy(&result, &f32, 4);
        return result;
    }
    uint32_t f32 = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    memcpy(&result, &f32, 4);
    return result;
}
gguf_ctx* gguf_open(const char *path) {
    gguf_ctx *ctx = calloc(1, sizeof(gguf_ctx));
    if (!ctx) return NULL;
    
    FILE *f = fopen(path, "rb");
    if (!f) { free(ctx); return NULL; }
    ctx->file = f;
    
    // Magic
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) {
        fclose(f); free(ctx); return NULL;
    }
    
    ctx->version = read_u32(f);
    ctx->n_tensors = read_i64(f);
    ctx->n_kv = read_i64(f);
    ctx->alignment = 32; // default
    
    fprintf(stderr, "GGUF v%u, %ld tensors, %ld KV pairs\n", 
            ctx->version, ctx->n_tensors, ctx->n_kv);
    
    // Skip KV pairs
    for (int64_t i = 0; i < ctx->n_kv; i++) {
        uint64_t key_len = read_u64(f);
        fseek(f, key_len, SEEK_CUR);
        
        int32_t typ = read_i32(f);
        
        switch (typ) {
            case 8: { // string
                uint64_t vlen = read_u64(f);
                fseek(f, vlen, SEEK_CUR);
                break;
            }
            case 4: case 5: case 6: fseek(f, 4, SEEK_CUR); break; // u32/i32/f32
            case 7: fseek(f, 1, SEEK_CUR); break; // bool
            case 10: case 11: fseek(f, 8, SEEK_CUR); break; // u64/i64
            case 9: { // array
                int32_t arr_type = read_i32(f);
                uint64_t arr_len = read_u64(f);
                if (arr_type == 8) {
                    for (uint64_t j = 0; j < arr_len; j++) {
                        uint64_t slen = read_u64(f);
                        fseek(f, slen, SEEK_CUR);
                    }
                } else {
                    int elem_size = 4;
                    if (arr_type == 0 || arr_type == 1 || arr_type == 7) elem_size = 1;
                    else if (arr_type == 2 || arr_type == 3) elem_size = 2;
                    else if (arr_type == 10 || arr_type == 11 || arr_type == 12) elem_size = 8;
                    fseek(f, arr_len * elem_size, SEEK_CUR);
                }
                break;
            }
            default: fseek(f, 4, SEEK_CUR); break;
        }
    }
    
    // Record tensor info file offset
    ctx->tensors_offset = ftell(f);
    
    // Read tensor info
    ctx->tensors = calloc(ctx->n_tensors, sizeof(gguf_tensor_info));
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        read_str(f, ctx->tensors[i].name, sizeof(ctx->tensors[i].name));
        ctx->tensors[i].n_dims = read_u32(f);
        for (int d = 0; d < ctx->tensors[i].n_dims && d < 4; d++) {
            ctx->tensors[i].dims[d] = read_i64(f);
        }
        ctx->tensors[i].ggml_type = read_i32(f);
        ctx->tensors[i].data_offset = read_u64(f);
    }
    
    // Data blob start
    long data_start = ftell(f);
    long pad = (ctx->alignment - (data_start % ctx->alignment)) % ctx->alignment;
    ctx->data_blob_offset = data_start + pad;
    
    fprintf(stderr, "Tensor info end at offset %ld, aligned to %lu (pad=%ld, alignment=%ld)\n", 
            data_start, ctx->data_blob_offset, pad, ctx->alignment);
    
    return ctx;
}

gguf_tensor_info* gguf_find_tensor(gguf_ctx *ctx, const char *name) {
    for (int64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}
float gguf_read_kv_f32(const char *path, const char *key, float default_val) {
    FILE *f = fopen(path, "rb");
    if (!f) return default_val;
    
    // GGUF header after magic(4): version(4) + n_tensors(8) + n_kv(8)
    fseek(f, 4, SEEK_SET);
    uint32_t version = read_u32(f); (void)version;
    int64_t n_tensors = read_i64(f); (void)n_tensors;
    int64_t n_kv = read_i64(f);
    
    for (int64_t i = 0; i < n_kv; i++) {
        uint64_t key_len = read_u64(f);
        char kbuf[256];
        size_t read_len = key_len < 255 ? key_len : 255;
        if (fread(kbuf, 1, read_len, f) != read_len) { fclose(f); return default_val; }
        kbuf[read_len] = '\0';
        if (read_len < key_len) fseek(f, key_len - read_len, SEEK_CUR);
        
        int32_t typ = read_i32(f);
        
        if (strcmp(kbuf, key) == 0) {
            if (typ == 6) { // f32
                float val;
                if (fread(&val, sizeof(float), 1, f) == 1) { fclose(f); return val; }
            }
            fclose(f);
            return default_val;
        } else {
            // Skip value based on type
            switch (typ) {
                case 0: { uint8_t v; fread(&v,1,1,f); break; }
                case 1: { int8_t v; fread(&v,1,1,f); break; }
                case 2: { uint16_t v; fread(&v,2,1,f); break; }
                case 3: { int16_t v; fread(&v,2,1,f); break; }
                case 4: { uint32_t v; fread(&v,4,1,f); break; }
                case 5: { int32_t v; fread(&v,4,1,f); break; }
                case 6: { float v; fread(&v,4,1,f); break; }
                case 7: { uint64_t v; fread(&v,8,1,f); break; }
                case 8: { int64_t v; fread(&v,8,1,f); break; }
                case 9: { double v; fread(&v,8,1,f); break; }
                case 10: { // bool
                    uint8_t v;
                    fread(&v,1,1,f);
                    break;
                }
                default: {
                    uint64_t arr_len = read_u64(f);
                    int32_t arr_typ = read_i32(f);
                    for (uint64_t a = 0; a < arr_len; a++) {
                        uint8_t discard[8];
                        fread(discard, 1, 8, f);
                    }
                    break;
                }
            }
        }
    }
    
    fclose(f);
    return default_val;
}
void gguf_close(gguf_ctx *ctx) {
    if (ctx) {
        if (ctx->file) fclose(ctx->file);
        if (ctx->data_blob) {
            if (ctx->data_blob_is_mmap)
                munmap(ctx->data_blob_mmap_base, ctx->data_blob_mmap_len);
            else
                free(ctx->data_blob);
        }
        free(ctx->tensors);
        free(ctx);
    }
}

/* Count of tokenizer.ggml.tokens (the vocab size) by re-walking the KV
 * section — the tokenizer needs the MODEL's vocab count to pick the right
 * cache file (data/vocab_<N>.bin). Returns -1 if absent. */
int64_t gguf_tokenizer_token_count(gguf_ctx *ctx) {
    if (!ctx || !ctx->file) return -1;
    fseek(ctx->file, 16, SEEK_SET);  /* magic(4) ver(4) n_tensors(8) n_kv(8) */
    int64_t n_kv = 0;
    if (fread(&n_kv, 8, 1, ctx->file) != 1) return -1;
    for (int64_t ki = 0; ki < n_kv; ki++) {
        uint64_t klen = 0;
        if (fread(&klen, 8, 1, ctx->file) != 1) return -1;
        char key[256];
        if (klen >= sizeof(key)) { fseek(ctx->file, (long)klen, SEEK_CUR); continue; }
        if (fread(key, 1, (size_t)klen, ctx->file) != klen) return -1;
        key[klen] = 0;
        int32_t vtype = 0;
        if (fread(&vtype, 4, 1, ctx->file) != 1) return -1;
        if (vtype == 9) {  /* array */
            int32_t arr_type = 0; uint64_t arr_len = 0;
            if (fread(&arr_type, 4, 1, ctx->file) != 1) return -1;
            if (fread(&arr_len, 8, 1, ctx->file) != 1) return -1;
            if (strcmp(key, "tokenizer.ggml.tokens") == 0 && arr_type == 8)
                return (int64_t)arr_len;
            if (arr_type == 8) {
                for (uint64_t j = 0; j < arr_len; j++) {
                    uint64_t slen = 0;
                    if (fread(&slen, 8, 1, ctx->file) != 1) return -1;
                    fseek(ctx->file, (long)slen, SEEK_CUR);
                }
            } else {
                int elem_size = 4;
                if (arr_type == 0 || arr_type == 1 || arr_type == 7) elem_size = 1;
                else if (arr_type == 2 || arr_type == 3) elem_size = 2;
                else if (arr_type == 10 || arr_type == 11 || arr_type == 12) elem_size = 8;
                fseek(ctx->file, (long)(arr_len * (uint64_t)elem_size), SEEK_CUR);
            }
        } else if (vtype == 8) {  /* string */
            uint64_t slen = 0;
            if (fread(&slen, 8, 1, ctx->file) != 1) return -1;
            fseek(ctx->file, (long)slen, SEEK_CUR);
        } else {
            int esz = 4;
            if (vtype == 0 || vtype == 1 || vtype == 7) esz = 1;
            else if (vtype == 2 || vtype == 3) esz = 2;
            else if (vtype == 10 || vtype == 11 || vtype == 12) esz = 8;
            fseek(ctx->file, esz, SEEK_CUR);
        }
    }
    return -1;
}
int64_t gguf_read_kv_i64(gguf_ctx *ctx, const char *want, int64_t def) {
    if (!ctx || !ctx->file) return def;
    fseek(ctx->file, 16, SEEK_SET);
    int64_t n_kv = 0;
    if (fread(&n_kv, 8, 1, ctx->file) != 1) return def;
    for (int64_t ki = 0; ki < n_kv; ki++) {
        uint64_t klen = 0;
        if (fread(&klen, 8, 1, ctx->file) != 1) return def;
        char key[256];
        if (klen >= sizeof(key)) { fseek(ctx->file, (long)klen, SEEK_CUR); continue; }
        if (fread(key, 1, (size_t)klen, ctx->file) != klen) return def;
        key[klen] = 0;
        int32_t vtype = 0;
        if (fread(&vtype, 4, 1, ctx->file) != 1) return def;
        if (vtype == 9) {  /* array — int arrays return the FIRST element
                            * (llama.cpp allows head_count_kv as a list) */
            int32_t arr_type = 0; uint64_t arr_len = 0;
            if (fread(&arr_type, 4, 1, ctx->file) != 1) return def;
            if (fread(&arr_len, 8, 1, ctx->file) != 1) return def;
            if (strcmp(key, want) == 0 && (arr_type == 4 || arr_type == 5) && arr_len > 0) {
                int32_t x; if (fread(&x, 4, 1, ctx->file) == 1) return (int64_t)x;
                return def;
            }
            if (arr_type == 8) {
                for (uint64_t j = 0; j < arr_len; j++) {
                    uint64_t slen = 0;
                    if (fread(&slen, 8, 1, ctx->file) != 1) return def;
                    fseek(ctx->file, (long)slen, SEEK_CUR);
                }
            } else {
                int esz = 4;
                if (arr_type == 0 || arr_type == 1 || arr_type == 7) esz = 1;
                else if (arr_type == 2 || arr_type == 3) esz = 2;
                else if (arr_type == 10 || arr_type == 11 || arr_type == 12) esz = 8;
                fseek(ctx->file, (long)(arr_len * (uint64_t)esz), SEEK_CUR);
            }
            continue;
        }
        if (strcmp(key, want) == 0) {
            int64_t v = def;
            if (vtype == 0 || vtype == 1) { int8_t x; if (fread(&x, 1, 1, ctx->file) == 1) v = x; }
            else if (vtype == 2 || vtype == 3) { int16_t x; if (fread(&x, 2, 1, ctx->file) == 1) v = x; }
            else if (vtype == 4 || vtype == 5) { int32_t x; if (fread(&x, 4, 1, ctx->file) == 1) v = x; }
            else if (vtype == 6) { float x; if (fread(&x, 4, 1, ctx->file) == 1) v = (int64_t)x; }
            else if (vtype == 7) { uint8_t x; if (fread(&x, 1, 1, ctx->file) == 1) v = x; }
            else if (vtype == 10 || vtype == 11 || vtype == 12) { int64_t x; if (fread(&x, 8, 1, ctx->file) == 1) v = x; }
            return v;
        }
        /* skip the value */
        if (vtype == 8) {
            uint64_t slen = 0;
            if (fread(&slen, 8, 1, ctx->file) != 1) return def;
            fseek(ctx->file, (long)slen, SEEK_CUR);
        } else if (vtype == 0 || vtype == 1 || vtype == 7) {
            fseek(ctx->file, 1, SEEK_CUR);
        } else if (vtype == 2 || vtype == 3) {
            fseek(ctx->file, 2, SEEK_CUR);
        } else if (vtype == 4 || vtype == 5 || vtype == 6) {
            fseek(ctx->file, 4, SEEK_CUR);
        } else {
            fseek(ctx->file, 8, SEEK_CUR);
        }
    }
    return def;
}
int gguf_kv_get_i32(gguf_ctx *ctx, const char *key, int *out) {
    int64_t v = gguf_read_kv_i64(ctx, key, INT64_MIN);
    if (v == INT64_MIN) return 0;
    if (out) *out = (int)v;
    return 1;
}
int gguf_kv_get_f32(gguf_ctx *ctx, const char *key, float *out) {
    int64_t v = gguf_read_kv_i64(ctx, key, INT64_MIN);
    if (v == INT64_MIN) return 0;
    if (out) *out = (float)v;
    return 1;
}
int gguf_kv_get_i32_arr(gguf_ctx *ctx, const char *key, int *out, int max) {
    /* the walker returns the FIRST element of an int array; the LFM2.5
     * per-layer head_count_kv needs the FULL array — walk and collect. */
    if (!ctx || !ctx->file || !out || max <= 0) return 0;
    fseek(ctx->file, 16, SEEK_SET);
    int64_t n_kv = 0;
    if (fread(&n_kv, 8, 1, ctx->file) != 1) return 0;
    for (int64_t ki = 0; ki < n_kv; ki++) {
        uint64_t klen = 0;
        if (fread(&klen, 8, 1, ctx->file) != 1) return 0;
        char k[256];
        if (klen >= sizeof(k)) { fseek(ctx->file, (long)klen, SEEK_CUR); continue; }
        if (fread(k, 1, (size_t)klen, ctx->file) != klen) return 0;
        k[klen] = 0;
        int32_t vt = 0;
        if (fread(&vt, 4, 1, ctx->file) != 1) return 0;
        if (strcmp(k, key) == 0 && vt == 9) {
            int32_t at = 0; uint64_t al = 0;
            if (fread(&at, 4, 1, ctx->file) != 1) return 0;
            if (fread(&al, 8, 1, ctx->file) != 1) return 0;
            if ((at == 4 || at == 5) && al > 0) {
                int n = (int)(al < (uint64_t)max ? al : (uint64_t)max);
                for (int i = 0; i < n; i++)
                    if (fread(&out[i], 4, 1, ctx->file) != 1) return 0;
                return n;
            }
            return 0;
        }
        /* skip */
        if (vt == 8) { uint64_t sl; if (fread(&sl, 8, 1, ctx->file) == 1) fseek(ctx->file, (long)sl, SEEK_CUR); }
        else if (vt == 9) {
            int32_t at; uint64_t al;
            if (fread(&at, 4, 1, ctx->file) != 1 || fread(&al, 8, 1, ctx->file) != 1) return 0;
            if (at == 8) { for (uint64_t j = 0; j < al; j++) { uint64_t sl; if (fread(&sl, 8, 1, ctx->file) == 1) fseek(ctx->file, (long)sl, SEEK_CUR); } }
            else { int esz = 4; if (at == 0 || at == 1 || at == 7) esz = 1; else if (at == 2 || at == 3) esz = 2; else if (at == 10 || at == 11 || at == 12) esz = 8; fseek(ctx->file, (long)(al * (uint64_t)esz), SEEK_CUR); }
        } else {
            int esz = 4; if (vt == 0 || vt == 1 || vt == 7) esz = 1; else if (vt == 2 || vt == 3) esz = 2; else if (vt == 10 || vt == 11 || vt == 12) esz = 8;
            fseek(ctx->file, esz, SEEK_CUR);
        }
    }
    return 0;
}
int gguf_buffer_data(gguf_ctx *ctx) {
    if (ctx->data_blob) return 1;  // already buffered

    // Get file size
    fseek(ctx->file, 0, SEEK_END);
    long file_size = ftell(ctx->file);
    uint64_t blob_size = file_size - ctx->data_blob_offset;
    if (blob_size == 0) return 0;

#if defined(_POSIX_MAPPED_FILES)
    /* mmap requires a page-aligned offset — GGUF's blob offset is only
     * 32-aligned, so map from the aligned base and shift the pointer. */
    {
        const long pg = sysconf(_SC_PAGESIZE);
        const off_t aligned_off = ((off_t)ctx->data_blob_offset / pg) * pg;
        const size_t shift = (size_t)(ctx->data_blob_offset - aligned_off);
        void *base = mmap(NULL, blob_size + shift, PROT_READ, MAP_PRIVATE,
                          fileno(ctx->file), aligned_off);
        if (base != MAP_FAILED) {
            ctx->data_blob = (uint8_t *)base + shift;
            ctx->data_blob_size = blob_size;
            ctx->data_blob_is_mmap = 1;
            ctx->data_blob_mmap_base = base;
            ctx->data_blob_mmap_len = blob_size + shift;
            /* Prefetch + 2MB huge pages on the ALIGNED base (madvise needs
             * page alignment): kills the per-4KB major-fault stall. */
#if defined(__linux__)
            madvise(base, ctx->data_blob_mmap_len, MADV_WILLNEED);
            madvise(base, ctx->data_blob_mmap_len, MADV_HUGEPAGE);
#endif
            fprintf(stderr, "  GGUF data blob mmap'd: %lu MB\n", (unsigned long)(blob_size / (1024*1024)));
            return 1;
        }
        fprintf(stderr, "  gguf_buffer_data: mmap failed, falling back to malloc+fread\n");
    }
#endif

    {
    // Fallback: 64-byte aligned allocation for optimal DDR4 burst reads
    fseek(ctx->file, ctx->data_blob_offset, SEEK_SET);
    void *blob = NULL;
    if (posix_memalign(&blob, 64, blob_size) != 0) {
        fprintf(stderr, "gguf_buffer_data: posix_memalign failed for %lu bytes\n", (unsigned long)blob_size);
        fclose(ctx->file);
        ctx->file = NULL;
        return 0;
    }
    ctx->data_blob = blob;

    size_t n_read = fread(ctx->data_blob, 1, blob_size, ctx->file);
    if (n_read != blob_size) {
        fprintf(stderr, "gguf_buffer_data: read %zu/%lu bytes\n", n_read, (unsigned long)blob_size);
        free(ctx->data_blob);
        ctx->data_blob = NULL;
        return 0;
    }
    
    ctx->data_blob_size = blob_size;
    fprintf(stderr, "  GGUF data blob buffered: %lu MB\n", (unsigned long)(blob_size / (1024*1024)));
    return 1;
    }
}
