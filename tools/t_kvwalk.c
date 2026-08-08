/* t_kvwalk.c — walk a GGUF's KV section, print key + type + size. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static uint64_t rd_u64(FILE *f) { uint8_t b[8]; if (fread(b,1,8,f)!=8) { fprintf(stderr,"EOF at %ld\n",ftell(f)); exit(1);} uint64_t v=0; for(int i=7;i>=0;i--) v=(v<<8)|b[i]; return v; }
static uint32_t rd_u32(FILE *f) { uint8_t b[4]; if (fread(b,1,4,f)!=4) { fprintf(stderr,"EOF4 at %ld\n",ftell(f)); exit(1);} return b[0]|(b[1]<<8)|(b[2]<<16)|(b[3]<<24); }
static uint64_t be_u64(const uint8_t *p) { uint64_t v=0; for(int i=7;i>=0;i--) v=(v<<8)|p[i]; return v; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "rb");
    uint8_t hdr[24]; fread(hdr,1,24,f);
    int64_t n_tensors = (int64_t)be_u64(hdr + 8);
    int64_t n_kv = (int64_t)be_u64(hdr + 16);
    fprintf(stderr, "n_tensors=%lld n_kv=%lld\n", (long long)n_tensors, (long long)n_kv);
    for (int64_t i = 0; i < n_kv; i++) {
        fprintf(stderr, "KV %lld: ", (long long)i);
        uint64_t klen = rd_u64(f);
        char *key = malloc(klen+1);
        if (fread(key, 1, klen, f) != klen) { fprintf(stderr, "key fail\n"); return 1; }
        key[klen]=0;
        int32_t typ = (int32_t)rd_u32(f);
        fprintf(stderr, "type=%d key=%s", typ, key);
        if (typ == 8) { uint64_t sl = rd_u64(f); fprintf(stderr, " str:%llu", (unsigned long long)sl); fseek(f, sl, SEEK_CUR); }
        else if (typ == 9) { int32_t at = (int32_t)rd_u32(f); uint64_t al = rd_u64(f); fprintf(stderr, " arr[t=%d n=%llu]", at, (unsigned long long)al);
            if (at == 8) { for (uint64_t j=0;j<al;j++){ uint64_t sl=rd_u64(f); fseek(f,sl,SEEK_CUR);} }
            else { int es=4; if(at==0||at==1||at==7)es=1; else if(at==2||at==3)es=2; else if(at>=10&&at<=12)es=8; fseek(f,(long)(al*es),SEEK_CUR);} }
        else if (typ == 4 || typ == 6 || typ == 5) { fprintf(stderr, " 4b"); fseek(f,4,SEEK_CUR); }
        else if (typ == 7) { fprintf(stderr, " bool"); fseek(f,1,SEEK_CUR); }
        else if (typ == 10 || typ == 11 || typ == 12) { fprintf(stderr, " 8b"); fseek(f,8,SEEK_CUR); }
        else if (typ <= 3) { int sz = typ<=1?1:2; fprintf(stderr, " %db",sz); fseek(f,sz,SEEK_CUR); }
        else { fprintf(stderr, " UNKNOWN!"); return 1; }
        fprintf(stderr, "\n");
        free(key);
    }
    fprintf(stderr, "KV walk done\n");
    return 0;
}
