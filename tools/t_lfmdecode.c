/* t_lfmdecode.c — decode token ids from LFM2.5's GGUF vocab (byte-level BPE).
 * Usage: t_lfmdecode <model.gguf> <tok1> <tok2> ...
 * Reads vocab directly from GGUF KV (tokenizer.ggml.tokens) + merges. */
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static uint64_t rd_u64(FILE *f) { uint8_t b[8]; if (fread(b,1,8,f)!=8) return 0; uint64_t v=0; for(int i=0;i<8;i++) v|=((uint64_t)b[i])<<(8*i); return v; }
static int rd_u32(FILE *f) { uint8_t b[4]; if (fread(b,1,4,f)!=4) return 0; return (b[3]<<24)|(b[2]<<16)|(b[1]<<8)|b[0]; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <gguf> <tok>...\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    char magic[4]; fread(magic,1,4,f);
    uint32_t version = rd_u32(f);
    uint64_t n_tensors = rd_u64(f);
    uint64_t n_kv = rd_u64(f);
    /* walk KV to find tokenizer.ggml.tokens (type 8 = array of strings) */
    char **tokens = NULL; int n_tok = 0;
    for (uint64_t i = 0; i < n_kv; i++) {
        uint64_t kl = rd_u64(f); char *k = malloc(kl+1); fread(k,1,kl,f); k[kl]=0;
        uint32_t t = rd_u32(f);
        if (!strcmp(k, "tokenizer.ggml.tokens") && t == 9) {
            uint32_t et = rd_u32(f);   /* array element type (8 = string) */
            uint64_t cnt = rd_u64(f);
            if (et != 8) { fprintf(stderr, "tokens array elem type %u != 8\n", et); return 1; }
            tokens = calloc(cnt, sizeof(char*));
            n_tok = (int)cnt;
            for (uint64_t j = 0; j < cnt; j++) {
                uint64_t sl = rd_u64(f);
                tokens[j] = malloc(sl+1);
                if (fread(tokens[j],1,sl,f) != sl) { fprintf(stderr, "read fail tok %llu\n", (unsigned long long)j); return 1; }
                tokens[j][sl]=0;
            }
            break;
        } else {
            /* GGUF v3 KV types: 0=u8 1=i8 2=u16 3=i16 4=u32 5=i32
             * 6=f32 7=bool 8=string 9=array 10=u64 11=i64 12=f64 */
            switch (t) {
                case 0: case 1: case 7: fseek(f,1,SEEK_CUR); break;
                case 2: case 3: fseek(f,2,SEEK_CUR); break;
                case 4: case 5: case 6: fseek(f,4,SEEK_CUR); break;
                case 8: { uint64_t l=rd_u64(f); fseek(f,l,SEEK_CUR); break; }
                case 9: { uint32_t et=rd_u32(f); uint64_t cnt=rd_u64(f);
                          switch (et) {
                            case 0: case 1: case 7: fseek(f,cnt*1,SEEK_CUR); break;
                            case 2: case 3: fseek(f,cnt*2,SEEK_CUR); break;
                            case 4: case 5: case 6: fseek(f,cnt*4,SEEK_CUR); break;
                            case 8: for(uint64_t j=0;j<cnt;j++){uint64_t sl=rd_u64(f); fseek(f,sl,SEEK_CUR);} break;
                            case 10: case 11: fseek(f,cnt*8,SEEK_CUR); break;
                            case 12: fseek(f,cnt*8,SEEK_CUR); break;
                            default: fprintf(stderr,"unknown arr elem type %u\n",et); return 1;
                          } break; }
                case 10: case 11: case 12: fseek(f,8,SEEK_CUR); break;
                default: fprintf(stderr, "unknown KV type %u\n", t); return 1;
            }
        }
        free(k);
    }
    if (!tokens) { fprintf(stderr, "tokens KV not found\n"); return 1; }
    /* decode each arg */
    char out[1<<20]; size_t olen = 0;
    for (int a = 2; a < argc; a++) {
        int id = atoi(argv[a]);
        if (id < 0 || id >= n_tok) { fprintf(stderr, "bad id %d\n", id); continue; }
        size_t sl = strlen(tokens[id]);
        if (olen + sl < sizeof(out)) { memcpy(out+olen, tokens[id], sl); olen += sl; }
    }
    out[olen] = 0;
    printf("DECODED: %s\n", out);
    /* also show each raw token */
    for (int a = 2; a < argc; a++) {
        int id = atoi(argv[a]);
        if (id < 0 || id >= n_tok) continue;
        printf("  %d -> [%s]\n", id, tokens[id]);
    }
    return 0;
}
