#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_fp16.h"

/* Q4_K dequant helper (replicated from wubuwizard gguf_reader.c) */
#define QK_K 256
#define Q4_K_BLOCK_SIZE 144

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void dequantize_q4_K_row(const uint8_t *data, float *output, int64_t n_elems) {
    int64_t n_blocks = (n_elems + QK_K - 1) / QK_K;
    for (int64_t b = 0; b < n_blocks; b++) {
        const uint8_t *block = data + b * Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block, 2);
        memcpy(&dmin_bits, block + 2, 2);
        float d = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);
        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;
        int is = 0;
        for (int j = 0; j < QK_K; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            float d1 = d * sc; float m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            float d2 = d * sc; float m2 = dmin * m;
            const uint8_t *bq = qs + j/2;
            int base = b * QK_K + j;
            for (int l = 0; l < 32 && (base + l) < n_elems; l++)
                output[base + l] = d1 * (bq[l] & 0xF) - m1;
            for (int l = 0; l < 32 && (base + 32 + l) < n_elems; l++)
                output[base + 32 + l] = d2 * (bq[l] >> 4) - m2;
            is += 2;
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const int N = 1000;
    
    // Use wubuwizard gguf_reader to find tensor info
    // (Parse GGUF header ourselves to get offset and type)
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return 1; }
    
    // Skip magic + version + n_tensors + n_kv
    char magic[4]; fread(magic, 1, 4, f);
    uint32_t version; fread(&version, 4, 1, f);
    int64_t n_tensors; fread(&n_tensors, 8, 1, f);
    int64_t n_kv; fread(&n_kv, 8, 1, f);
    
    // Skip KV pairs
    for (int64_t i = 0; i < n_kv; i++) {
        uint64_t key_len; fread(&key_len, 8, 1, f);
        fseek(f, key_len, SEEK_CUR);
        int32_t typ; fread(&typ, 4, 1, f);
        switch (typ) {
            case 8: { // string
                uint64_t vlen; fread(&vlen, 8, 1, f);
                fseek(f, vlen, SEEK_CUR);
                break;
            }
            case 4: case 5: case 6: fseek(f, 4, SEEK_CUR); break;
            case 7: fseek(f, 1, SEEK_CUR); break;
            case 10: case 11: fseek(f, 8, SEEK_CUR); break;
            case 9: {
                int32_t arr_type; fread(&arr_type, 4, 1, f);
                uint64_t arr_len; fread(&arr_len, 8, 1, f);
                if (arr_type == 8) {
                    for (uint64_t j = 0; j < arr_len; j++) {
                        uint64_t slen; fread(&slen, 8, 1, f);
                        fseek(f, slen, SEEK_CUR);
                    }
                } else {
                    int es = (arr_type == 0||arr_type==1||arr_type==7) ? 1 :
                             (arr_type == 2||arr_type==3) ? 2 :
                             (arr_type == 10||arr_type==11||arr_type==12) ? 8 : 4;
                    fseek(f, arr_len * es, SEEK_CUR);
                }
                break;
            }
            default: fseek(f, 4, SEEK_CUR); break;
        }
    }
    
    // Read tensor info, find output.weight
    uint64_t tensor_offset = 0;
    int ggml_type = -1;
    int64_t dims[4] = {0};
    int n_dims = 0;
    int found = 0;
    uint32_t alignment = 32;
    
    for (int64_t i = 0; i < n_tensors; i++) {
        uint64_t name_len; fread(&name_len, 8, 1, f);
        char tname[256]; 
        size_t nr = fread(tname, 1, name_len < 255 ? name_len : 255, f);
        tname[nr] = '\0';
        if (name_len > 255) fseek(f, name_len - 255, SEEK_CUR);
        
        uint32_t nd; fread(&nd, 4, 1, f);
        int64_t td[4] = {0};
        for (int d = 0; d < (int)nd && d < 4; d++) fread(&td[d], 8, 1, f);
        int32_t gt; fread(&gt, 4, 1, f);
        uint64_t off; fread(&off, 8, 1, f);
        
        if (strcmp(tname, "output.weight") == 0) {
            tensor_offset = off;
            ggml_type = gt;
            n_dims = (int)nd;
            memcpy(dims, td, sizeof(int64_t) * (nd < 4 ? nd : 4));
            found = 1;
        }
    }
    
    if (!found) {
        fprintf(stderr, "output.weight not found\n");
        fclose(f); return 1;
    }
    
    long data_start = ftell(f);
    long pad = (alignment - (data_start % alignment)) % alignment;
    uint64_t data_blob_offset = data_start + pad;
    
    printf("output.weight: type=%d dims=[%ld,%ld] tensor_offset=%lu data_blob_offset=%lu file_offset=%lu\n",
           ggml_type, (long)dims[0], (long)dims[1],
           (unsigned long)tensor_offset, (unsigned long)data_blob_offset,
           (unsigned long)(data_blob_offset + tensor_offset));
    
    int64_t total_elems = 1;
    for (int d = 0; d < n_dims; d++) total_elems *= dims[d];
    int n_read = (N < total_elems) ? N : (int)total_elems;
    
    // Read raw Q4_K blocks for first N elements
    int64_t n_blocks = (n_read + QK_K - 1) / QK_K;
    size_t raw_size = n_blocks * Q4_K_BLOCK_SIZE;
    
    fseek(f, data_blob_offset + tensor_offset, SEEK_SET);
    uint8_t *raw = (uint8_t *)malloc(raw_size);
    if (!raw) { fprintf(stderr, "malloc failed\n"); fclose(f); return 1; }
    size_t n_bytes = fread(raw, 1, raw_size, f);
    fclose(f);
    
    printf("Read %zu raw bytes (%lld blocks of %d bytes)\n", n_bytes, (long long)n_blocks, Q4_K_BLOCK_SIZE);
    printf("Total tensor elements: %lld, reading first %d\n", (long long)total_elems, n_read);
    
    // Dequantize using replicated wubuwizard Q4_K dequant
    float *buf = (float *)malloc(n_blocks * QK_K * sizeof(float));
    if (!buf) { free(raw); return 1; }
    
    dequantize_q4_K_row(raw, buf, n_read);
    
    // Compute stats
    double sum = 0, sum2 = 0;
    float vmin = buf[0], vmax = buf[0];
    for (int i = 0; i < n_read; i++) {
        sum += buf[i];
        sum2 += (double)buf[i] * (double)buf[i];
        if (buf[i] < vmin) vmin = buf[i];
        if (buf[i] > vmax) vmax = buf[i];
    }
    double mean = sum / n_read;
    double rms = sqrt(sum2 / n_read);
    
    printf("\n=== WubuWizard Dequant Stats (first %d elements of output.weight) ===\n", n_read);
    printf("  Min:   %+.10f\n", (double)vmin);
    printf("  Max:   %+.10f\n", (double)vmax);
    printf("  Mean:  %+.10f\n", mean);
    printf("  RMS:   %+.10f\n", rms);
    
    printf("\nFirst 20 values:\n");
    for (int i = 0; i < 20 && i < n_read; i++) {
        printf("  [%d] %+.10f\n", i, (double)buf[i]);
    }
    
    // Write values to binary file for comparison
    const char *outpath = "/home/wubu/vault/bins/output_weight_wubuwizard_f32.bin";
    FILE *fout = fopen(outpath, "wb");
    if (fout) {
        fwrite(buf, sizeof(float), n_read, fout);
        fclose(fout);
        printf("\nWrote %d float values to %s\n", n_read, outpath);
    } else {
        fprintf(stderr, "Failed to write %s\n", outpath);
    }
    
    free(buf);
    free(raw);
    return 0;
}
