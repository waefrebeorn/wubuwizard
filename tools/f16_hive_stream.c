/* f16_hive_stream.c -- "store all" the f16 weights as packed u32 hi/lo words, streamed through a WuBuHIVE
 * ring (wb_hive replaces a FIFO ring).
 *
 *  1) The f16/bf16 hi-lo endian trick (WuBuMM docs/H3_F16_ADAPTATION.md §6): f16 is a 16-bit WORD with the same
 *     bit layout on every platform; endianness only matters if the two bytes are read apart. bf16 is the HI 16
 *     bits of an f32 -> decode with a BIT REINTERPRET, u=(uint32)bf16<<16; memcpy(&f,&u,4), NEVER the numeric
 *     cast (float)((uint32)bf16<<16) which decoded 1.0 (0x3F80) to 1.065e9. Packing two f16 into one u32 fills
 *     the 32-bit bus: on little-endian x86 the FIRST word is the LOW half, u32 = w[2j+1]<<16 | w[2j].
 *
 *  2) WuBuHIVE (wb_hive): "replaces FIFO rings and malloc/free chains" - stable contiguous slots + skipfield +
 *     freelist. We use it as the backing store for a bounded ring of fixed CHUNK slabs carrying packed weights
 *     in stream order; the ring index lives over hive slots whose addresses never move while live.
 *
 * Build: gcc -O2 -std=c11 -I <hive>/include -o f16_hive_stream f16_hive_stream.c <hive>/src/wb_hive.c -lm
 * usage: f16_hive_stream <f16dir> [chunkKB] [ringSlots]
 */
#include "wb_hive.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>
#include <time.h>

static float bf16_hi_lo(uint16_t w) { uint32_t u = (uint32_t)w << 16; float f; memcpy(&f,&u,4); return f; }
static float bf16_wrong_numeric(uint16_t w) { return (float)((uint32_t)w << 16); }

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <f16dir> [chunkKB=1024] [ringSlots=16]\n", argv[0]); return 2; }
    size_t chunkKB = argc > 2 ? (size_t)atoi(argv[2]) : 1024;
    size_t ringSlots = argc > 3 ? (size_t)atoi(argv[3]) : 16;
    if (chunkKB < 4) chunkKB = 4;
    const size_t CHUNK = chunkKB * 1024;

    printf("hi-lo trick check: 0x3F80 bf16 -> hi/lo decode = %.6f (want 1.000000); numeric cast = %.0f (the bug)\n",
           bf16_hi_lo(0x3F80), bf16_wrong_numeric(0x3F80));

    wb_hive hive; if (wb_hive_init(&hive, CHUNK, 8) != 0) { fprintf(stderr,"hive init fail\n"); return 3; }
    void **ring = (void**)calloc(ringSlots, sizeof(void*));
    size_t head = 0, tail = 0, live = 0;
    #define PUSH(p) do{ ring[head] = (p); head = (head+1)%ringSlots; if(live<ringSlots) live++; \
                        else { wb_hive_drop(&hive, ring[tail]); tail=(tail+1)%ringSlots; } }while(0)

    WIN32_FIND_DATAA fd; char pat[700]; snprintf(pat,sizeof pat,"%s\\*.f16", argv[1]);
    HANDLE hd = FindFirstFileA(pat,&fd); if (hd==INVALID_HANDLE_VALUE){ fprintf(stderr,"no f16 in %s\n",argv[1]); return 4; }
    unsigned long long total_u32 = 0, total_bytes = 0, nslabs = 0, ntensors = 0, verify_fail = 0;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    do {
        const char *cf = fd.cFileName;
        if (cf[0]=='.' || strstr(cf,"_ref.f32")) continue;
        char p[900]; snprintf(p,sizeof p,"%s\\%s", argv[1], cf);
        FILE *f=fopen(p,"rb"); if(!f) continue;
        long long d0,d1; if(fread(&d0,8,1,f)!=1||fread(&d1,8,1,f)!=1){fclose(f);continue;}
        long long nw = d0*d1;
        uint16_t *words = (uint16_t*)malloc((size_t)nw*2);
        if(!words || fread(words,2,(size_t)nw,f)!=(size_t)nw){ if(words)free(words); fclose(f); continue; }
        fclose(f); ntensors++;
        long long nu = (nw+1)/2;
        uint32_t *packed = (uint32_t*)malloc((size_t)nu*4);
        for (long long j=0;j<nu;j++){
            uint32_t lo = words[2*j]; uint32_t hi = (2*j+1 < nw) ? words[2*j+1] : 0;
            packed[j] = (hi<<16) | lo;                /* little-endian: first word is the LOW half */
        }
        total_u32 += nu; total_bytes += (unsigned long long)nw*2;
        size_t u32_per_chunk = CHUNK/4;
        for (size_t off=0; off<(size_t)nu; off+=u32_per_chunk){
            size_t n = (off+u32_per_chunk <= (size_t)nu) ? u32_per_chunk : ((size_t)nu - off);
            void *slot = wb_hive_emit(&hive);
            if(!slot){ fprintf(stderr,"hive full\n"); free(packed); free(words); return 5; }
            uint32_t *dst = (uint32_t*)slot;
            memcpy(dst, packed+off, n*4);
            for (size_t j=0;j<n;j++){
                uint32_t u = dst[j];
                uint16_t lo = (uint16_t)(u & 0xFFFF), hi = (uint16_t)(u>>16);
                long long wi = (long long)off + (long long)j;
                if (2*wi   < nw && words[2*wi]   != lo) verify_fail++;
                if (2*wi+1 < nw && words[2*wi+1] != hi) verify_fail++;
            }
            PUSH(slot); nslabs++;
        }
        free(packed); free(words);
    } while (FindNextFileA(hd,&fd));
    FindClose(hd);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double sec = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;

    int hv = wb_hive_validate(&hive);
    printf("packed ALL: %llu tensors, %llu f16 words, %llu u32 hi/lo words = %.2f GB -> %llu slabs of %zu KB\n",
        ntensors, total_bytes/2, total_u32, (double)total_bytes/1e9, nslabs, chunkKB);
    printf("bit-exact round-trip: %s (mismatches=%llu)\n", verify_fail==0?"PASS":"FAIL", verify_fail);
    printf("hive ring: %zu slots live, budget %zu slabs (%zu KB), validate=%d, %.1f slabs/s, %.2f GB/s pack+verify\n",
        live, ringSlots, ringSlots*chunkKB, hv, (double)nslabs/sec, (double)total_bytes/sec/1e9);
    free(ring); wb_hive_clear(&hive);
    return (verify_fail==0 && hv==0) ? 0 : 1;
}
