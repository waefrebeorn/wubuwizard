#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "wubu_fp16.h"

int main() {
    FILE *f = fopen("/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf", "rb");
    // Output.weight data starts at GGUF data blob offset + tensor data_offset
    // data_blob_offset = 10990048 (from gguf_open output)
    // output.weight data_offset = 0 (first tensor in the data section)
    // Actually need to read tensor header to find offset
    
    // Simpler: gguf_open already found it. Let's compute manually.
    // GGUF: skip magic(4) + version(4) + n_tensors(8) + n_kv(8) = 24
    // Then KV pairs (we need to skip them)
    // Then tensor headers
    
    fseek(f, 24, SEEK_SET); // past header
    // Find tensor count from header
    uint64_t n_tensors, n_kv;
    fseek(f, 8, SEEK_SET); // skip magic + version
    fread(&n_tensors, 8, 1, f);
    fread(&n_kv, 8, 1, f);
    
    // Skip KV
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t klen; fread(&klen, 8, 1, f); fseek(f, klen, SEEK_CUR);
        uint32_t vtype; fread(&vtype, 4, 1, f);
        switch (vtype) {
            case 0: fseek(f, 4, SEEK_CUR); break;
            case 1: fseek(f, 1, SEEK_CUR); break;
            case 2: fseek(f, 1, SEEK_CUR); break;
            case 3: fseek(f, 8, SEEK_CUR); break;
            case 4: fseek(f, 4, SEEK_CUR); break;
            case 5: fseek(f, 8, SEEK_CUR); break;
            case 6: case 7: case 8: { uint64_t al; fread(&al, 8, 1, f); fseek(f, al, SEEK_CUR); break; }
            default: { uint64_t al; fread(&al, 8, 1, f); fseek(f, al, SEEK_CUR); break; }
        }
    }
    
    // Find output.weight
    uint64_t out_data_offset = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint32_t ndims; fread(&ndims, 4, 1, f);
        uint64_t nlen; fread(&nlen, 8, 1, f);
        char name[256]; memset(name,0,256); fread(name, 1, nlen < 255 ? nlen : 255, f);
        if (nlen >= 255) fseek(f, nlen - 255, SEEK_CUR);
        // Read all dims
        uint64_t dims[4];
        for (int d = 0; d < ndims && d < 4; d++) fread(&dims[d], 8, 1, f);
        uint32_t gtype; fread(&gtype, 4, 1, f);
        fread(&out_data_offset, 8, 1, f);
        if (strcmp(name, "output.weight") == 0) {
            printf("Found output.weight at data offset %lu (file offset %lu)\n",
                   out_data_offset, 10990048 + out_data_offset);
            break;
        }
    }
    
    uint64_t data_pos = 10990048 + out_data_offset;
    fseek(f, data_pos, SEEK_SET);
    
    // Read blocks 0 and 1
    for (int b = 0; b < 2; b++) {
        uint8_t block[176];
        fread(block, 1, 176, f);
        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits, block, 2);
        memcpy(&dmin_bits, block+2, 2);
        float d = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);
        
        printf("\nBlock %d: d_bits=%u dmin_bits=%u d=%f dmin=%f\n", b, d_bits, dmin_bits, d, dmin);
        printf("  scales[0:11]: ");
        for (int i = 0; i < 12; i++) printf("%02x ", block[4+i]);
        printf("\n");
        printf("  qh[0:7]: ");
        for (int i = 0; i < 8; i++) printf("%02x ", block[16+i]);
        printf("\n");
        printf("  qs[0:15]: ");
        for (int i = 0; i < 16; i++) printf("%02x ", block[48+i]);
        printf("\n");
        
        // Dequant first 64 elements with both formulas
        printf("  First 16 dequant (new formula - llama.cpp compatible):\n    ");
        for (int l = 0; l < 16; l++) {
            uint8_t q_low = block[48 + l] & 0xF;
            int is = 0;
            uint8_t sc0, m0; 
            sc0 = block[4] & 63; m0 = block[8] & 63;
            float val = d * sc0 * q_low - dmin * m0;
            printf("%+.6f ", val);
        }
        printf("\n  First 16 dequant (old formula - signed q-8):\n    ");
        for (int l = 0; l < 16; l++) {
            uint8_t q_low = block[48 + l] & 0xF;
            int is = 0;
            uint8_t sc0; 
            sc0 = block[4] & 63;
            float val = d * sc0 * (int8_t)(q_low - 8);
            printf("%+.6f ", val);
        }
        printf("\n");
    }
    
    fclose(f);
    return 0;
}
