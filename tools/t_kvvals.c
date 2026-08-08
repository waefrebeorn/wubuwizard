/* t_kvvals.c — print values of selected GGUF KV pairs (little-endian). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint64_t rd_u64(FILE *f) { uint8_t b[8]; if (fread(b,1,8,f)!=8) return 0; uint64_t v=0; for(int i=0;i<8;i++) v|=((uint64_t)b[i])<<(8*i); return v; }
static int rd_u32(FILE *f) { uint8_t b[4]; if (fread(b,1,4,f)!=4) return 0; return (b[3]<<24)|(b[2]<<16)|(b[1]<<8)|b[0]; }
static int rd_i32(FILE *f) { return (int)rd_u32(f); }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 1;
    fread((char[4]){0},1,4,f);
    rd_u32(f); /* version */
    rd_u64(f); /* n_tensors */
    uint64_t n_kv = rd_u64(f);
    const char *want[] = {"lfm2.block_count","lfm2.embedding_length","lfm2.feed_forward_length",
        "lfm2.attention.head_count","lfm2.attention.head_count_kv","lfm2.shortconv.l_cache",
        "lfm2.rope.freq_base","lfm2.attention.layer_norm_rms_epsilon", NULL};
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t kl = rd_u64(f);
        char *k = malloc(kl+1); fread(k,1,kl,f); k[kl]=0;
        uint32_t t = rd_u32(f);
        int wanted = 0;
        for (int w = 0; want[w]; w++) if (!strcmp(k, want[w])) { wanted = 1; break; }
        if (wanted) {
            printf("%s: ", k);
            if (t == 4 || t == 5) printf("i32=%d\n", rd_i32(f));
            else if (t == 6) { uint32_t b; fread(&b,4,1,f); float v; memcpy(&v,&b,4); printf("f32=%.6g\n", v); }
            else if (t == 9) {
                uint32_t et = rd_u32(f);
                uint64_t cnt = rd_u64(f);
                printf("arr[t=%u n=%llu] =", et, (unsigned long long)cnt);
                for (uint64_t j = 0; j < cnt; j++) {
                    if (et == 5) { int v = rd_i32(f); if (j < 8 || j >= cnt-2) printf(" %d", v); else if (j==8) printf(" ..."); }
                    else if (et == 4) { uint32_t v = rd_u32(f); if (j < 8) printf(" %u", v); }
                    else if (et == 8) { uint64_t sl = rd_u64(f); fseek(f, sl, SEEK_CUR); }
                    else { fseek(f, 4, SEEK_CUR); }
                }
                printf("\n");
            } else { printf("type %u\n", t); }
        } else {
            if (t == 9) { uint32_t et = rd_u32(f); uint64_t cnt = rd_u64(f);
                for (uint64_t j = 0; j < cnt; j++) {
                    if (et == 8) { uint64_t sl = rd_u64(f); fseek(f, sl, SEEK_CUR); }
                    else if (et == 5 || et == 4 || et == 6) fseek(f, 4, SEEK_CUR);
                    else if (et == 10 || et == 11 || et == 12) fseek(f, 8, SEEK_CUR);
                    else if (et == 0 || et == 1 || et == 7) fseek(f, 1, SEEK_CUR);
                    else if (et == 2 || et == 3) fseek(f, 2, SEEK_CUR);
                }
            } else if (t == 8) { uint64_t sl = rd_u64(f); fseek(f, sl, SEEK_CUR); }
            else if (t == 4 || t == 5 || t == 6) fseek(f, 4, SEEK_CUR);
            else if (t == 10 || t == 11 || t == 12) fseek(f, 8, SEEK_CUR);
            else if (t == 0 || t == 1 || t == 7) fseek(f, 1, SEEK_CUR);
            else if (t == 2 || t == 3) fseek(f, 2, SEEK_CUR);
        }
        free(k);
    }
    fclose(f);
    return 0;
}
