#include "wubu_tokenizer.h"
#include "wubu_spawn.h"
#include "gguf_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

// ========== Byte-level encoding helpers ==========

// Qwen3.6 exact byte-to-token mapping from original tokenizer.json
static const int byte_encoder_qwen36[256] = {
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,    0,    1,    2,    3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,
     15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,
     31,   32,   33,   34,   35,   36,   37,   38,   39,   40,   41,   42,   43,   44,   45,   46,
     47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   59,   60,   61,   62,
     63,   64,   65,   66,   67,   68,   69,   70,   71,   72,   73,   74,   75,   76,   77,   78,
     79,   80,   81,   82,   83,   84,   85,   86,   87,   88,   89,   90,   91,   92,   93,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,
     -1,   94,   95,   96,   97,   98,   99,  100,  101,  102,  103,  104,  105,   -1,  106,  107,
    108,  109,  110,  111,  112,  113,  114,  115,  116,  117,  118,  119,  120,  121,  122,  123,
    124,  125,  126,  127,  128,  129,  130,  131,  132,  133,  134,  135,  136,  137,  138,  139,
    140,  141,  142,  143,  144,  145,  146,  147,  148,  149,  150,  151,  152,  153,  154,  155,
    156,  157,  158,  159,  160,  161,  162,  163,  164,  165,  166,  167,  168,  169,  170,  171,
    172,  173,  174,  175,  176,  177,  178,  179,  180,  181,  182,  183,  184,  185,  186,  187
};

static int gpt2_byte_encoder[256] = {0};

static void init_byte_encoder(void) {
    static int initialized = 0;
    if (initialized) return;
    initialized = 1;
    for (int i = 0; i < 256; i++)
        gpt2_byte_encoder[i] = byte_encoder_qwen36[i];
    int highest = 187;
    int remap_idxs[] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,
        32, 127, 128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,
        148,149,150,151,152,153,154,155,156,157,158,159, 160, 173
    };
    for (size_t i = 0; i < sizeof(remap_idxs)/sizeof(int); i++) {
        int b = remap_idxs[i];
        gpt2_byte_encoder[b] = highest + 1 + (int)i;
    }
}

// ========== FNV-1a hash helpers ==========

static uint32_t merge_hash_key(int left_id, int right_id) {
    uint32_t h = 2166136261u;
    h = (h ^ (uint32_t)(left_id & 0xFF)) * 16777619u;
    h = (h ^ (uint32_t)((left_id >> 8) & 0xFF)) * 16777619u;
    h = (h ^ (uint32_t)((left_id >> 16) & 0xFF)) * 16777619u;
    h = (h ^ (uint32_t)((left_id >> 24) & 0xFF)) * 16777619u;
    h = (h ^ (uint32_t)(right_id & 0xFF)) * 16777619u;
    h = (h ^ (uint32_t)((right_id >> 8) & 0xFF)) * 16777619u;
    h = (h ^ (uint32_t)((right_id >> 16) & 0xFF)) * 16777619u;
    h = (h ^ (uint32_t)((right_id >> 24) & 0xFF)) * 16777619u;
    // MurmurHash3 finalizer — avalanches low bits for better table distribution
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}

static int find_merge(const wubu_tokenizer_t *tok, int left, int right, int *merged_id) {
    if (!tok || !tok->merge_hash || tok->merge_hash_size <= 0) return -1;
    uint32_t h = merge_hash_key(left, right);
    int idx = (int)(h & (uint32_t)(tok->merge_hash_size - 1));
    for (int i = 0; i < tok->merge_hash_size; i++) {
        int e = (idx + i) & (tok->merge_hash_size - 1);
        if (!tok->merge_hash[e].valid) return -1;
        if (tok->merge_hash[e].left_id == left && tok->merge_hash[e].right_id == right) {
            *merged_id = tok->merge_hash[e].merged_id;
            return tok->merge_hash[e].priority;
        }
    }
    return -1;
}

static int find_token_by_string(const wubu_tokenizer_t *tok, const uint8_t *bytes, int blen) {
    if (!tok || !tok->vocab_hash || tok->vocab_hash_size <= 0 || blen <= 0) return -1;
    uint32_t h = 2166136261u;
    int n = blen < 8 ? blen : 8;
    for (int i = 0; i < n; i++)
        h = (h ^ (uint32_t)bytes[i]) * 16777619u;
    int idx = (int)(h & (uint32_t)(tok->vocab_hash_size - 1));
    for (int i = 0; i < tok->vocab_hash_size; i++) {
        int e = (idx + i) & (tok->vocab_hash_size - 1);
        if (!tok->vocab_hash[e].valid) return -1;
        int tid = tok->vocab_hash[e].token_id;
        if (tid >= 0 && tid < tok->vocab_size &&
            tok->vocab[tid].byte_len == blen &&
            memcmp(tok->vocab[tid].bytes, bytes, blen) == 0)
            return tid;
    }
    return -1;
}

// ========== Init from binary files (pre-extracted tokenizer data) ==========

// Forward declarations
static void build_byte_token_ids(wubu_tokenizer_t *tok);
static void build_vocab_hash(wubu_tokenizer_t *tok);

bool wubu_tokenizer_init(wubu_tokenizer_t *tok, const char *gguf_path) {
    // Load pre-extracted tokenizer data. The cache files are vocab-keyed
    // (data/vocab_<N>.bin — each model family has its own vocab: Qwen3.5 =
    // 248320, LFM2.5 = 128000, MiniCPM5 = 130560); the GGUF's own tokenizer
    // count picks the right one, with data/vocab.bin as the fallback.
    if (!tok) return false;
    memset(tok, 0, sizeof(*tok));
    init_byte_encoder();
    tok->bos_id = -1; tok->eos_id = -1; tok->pad_id = -1;
    
    char vpath[256], mpath[256];
    int64_t model_vocab = -1;
    if (gguf_path) {
        gguf_ctx *g = gguf_open(gguf_path);
        if (g) {
            model_vocab = gguf_tokenizer_token_count(g);
            gguf_close(g);
        }
    }
    if (model_vocab > 0) {
        snprintf(vpath, sizeof(vpath), "data/vocab_%lld.bin", (long long)model_vocab);
        snprintf(mpath, sizeof(mpath), "data/merges_%lld.bin", (long long)model_vocab);
    } else {
        snprintf(vpath, sizeof(vpath), "data/vocab.bin");
        snprintf(mpath, sizeof(mpath), "data/merges.bin");
    }
    
    // Load vocab
    FILE *f = fopen(vpath, "rb");
    if (!f) {
        /* fall back to the generic cache */
        snprintf(vpath, sizeof(vpath), "data/vocab.bin");
        f = fopen(vpath, "rb");
    }
    if (!f) { fprintf(stderr, "Can't open %s (run python/extract_tokenizer.py first)\n", vpath); return false; }
    
    uint32_t vs;
    if (fread(&vs, 4, 1, f) != 1 || vs == 0) { fclose(f); return false; }
    tok->vocab_size = (int)vs;
    tok->vocab = (wubu_token_t *)calloc(tok->vocab_size, sizeof(wubu_token_t));
    
    for (uint32_t vi = 0; vi < vs; vi++) {
        uint32_t slen;
        if (fread(&slen, 4, 1, f) != 1) { fprintf(stderr,"Truncated vocab at %u\n",vi); break; }
        int max_slen = WUBU_TOKENIZER_MAX_TOKEN_BYTES - 1;
        if (slen > (uint32_t)max_slen) {
            // Token too long: copy what fits, skip the rest
            fread(tok->vocab[vi].bytes, 1, (size_t)max_slen, f);
            tok->vocab[vi].bytes[max_slen] = '\0';
            tok->vocab[vi].byte_len = max_slen;
            fseek(f, (long)(slen - (uint32_t)max_slen), SEEK_CUR);
        } else {
            size_t nrd = fread(tok->vocab[vi].bytes, 1, slen, f);
            tok->vocab[vi].bytes[nrd] = '\0';
            tok->vocab[vi].byte_len = (int)nrd;
        }
        tok->vocab[vi].id = (int)vi;
    }
    fclose(f);
    printf("  Vocab: %d tokens loaded\n", tok->vocab_size);
    
    build_vocab_hash(tok);
    build_byte_token_ids(tok);
    
    // Load merges
    f = fopen(mpath, "rb");
    if (f) {
        uint32_t ms;
        fread(&ms, 4, 1, f);
        if (ms > 0) {
            tok->merges = (wubu_merge_t *)calloc(ms, sizeof(wubu_merge_t));
            
            for (uint32_t mi = 0; mi < ms; mi++) {
                uint32_t slen;
                if (fread(&slen, 4, 1, f) != 1) break;
                if (slen > 512) {
                    /* oversized BPE merge (rare but legal) — skip the entry,
                     * keep the file position consistent (a bare break left
                     * the stream mid-entry and the encode later crashed) */
                    fprintf(stderr, "Skipping oversized merge %u (%u bytes)\n", mi, slen);
                    fseek(f, (long)slen, SEEK_CUR);
                    continue;
                }
                char *merge_str = (char *)malloc(slen + 1);
                fread(merge_str, 1, slen, f);
                merge_str[slen] = '\0';
                
                char *space = strchr(merge_str, ' ');
                if (space) {
                    *space = '\0';
                    int left_id = find_token_by_string(tok, (const uint8_t*)merge_str, (int)(space - merge_str));
                    int right_id = find_token_by_string(tok, (const uint8_t*)(space+1), (int)(slen - (space - merge_str) - 1));
                    
                    /* combined left+right must fit the merge buffer — the
                     * old code clamped mbl AFTER the memcpys, overflowing
                     * mb for long merges (fortify abort). Skip them. */
                    if ((int)(space - merge_str) + (int)(slen - (space - merge_str) - 1) > WUBU_TOKENIZER_MAX_TOKEN_BYTES - 1) {
                        *space = ' ';
                        free(merge_str);
                        continue;
                    }
                    uint8_t mb[WUBU_TOKENIZER_MAX_TOKEN_BYTES];
                    int mbl = (int)(space - merge_str) + (int)(slen - (space - merge_str) - 1);
                    if (mbl > WUBU_TOKENIZER_MAX_TOKEN_BYTES-1) mbl = WUBU_TOKENIZER_MAX_TOKEN_BYTES-1;
                    memcpy(mb, merge_str, space - merge_str);
                    memcpy(mb + (space - merge_str), space + 1, slen - (space - merge_str) - 1);
                    int merged_id = find_token_by_string(tok, mb, mbl);
                    
                    *space = ' '; // restore
                    
                    if (left_id >= 0 && right_id >= 0 && merged_id >= 0) {
                        tok->merges[tok->n_merges].left_id = left_id;
                        tok->merges[tok->n_merges].right_id = right_id;
                        tok->merges[tok->n_merges].new_id = merged_id;
                        tok->merges[tok->n_merges].priority = (int)mi;
                        tok->n_merges++;
                    }
                }
                free(merge_str);
            }
            printf("  Merges: %d loaded (%d resolved)\n", ms, tok->n_merges);
        }
        fclose(f);
    } else {
        printf("  No merges file\n");
    }
    
    // Load special tokens
    f = fopen("data/special_tokens.bin", "rb");
    if (f) {
        int32_t ids[3];
        if (fread(ids, 4, 3, f) == 3) {
            tok->bos_id = ids[0]; tok->eos_id = ids[1]; tok->pad_id = ids[2];
        }
        fclose(f);
    }
    
    // Build merge hash
    if (tok->n_merges > 0) {
        tok->merge_hash_size = 1;
        while (tok->merge_hash_size < tok->n_merges * 2) tok->merge_hash_size *= 2;
        tok->merge_hash = (wubu_merge_hash_entry_t *)calloc(tok->merge_hash_size, sizeof(wubu_merge_hash_entry_t));
        int collisions = 0;
        for (int i = 0; i < tok->n_merges; i++) {
            uint32_t h = merge_hash_key(tok->merges[i].left_id, tok->merges[i].right_id);
            int idx = (int)(h & (uint32_t)(tok->merge_hash_size - 1));
            for (int j = 0; j < tok->merge_hash_size; j++) {
                int e = (idx + j) & (tok->merge_hash_size - 1);
                if (!tok->merge_hash[e].valid) {
                    tok->merge_hash[e].left_id = tok->merges[i].left_id;
                    tok->merge_hash[e].right_id = tok->merges[i].right_id;
                    tok->merge_hash[e].merged_id = tok->merges[i].new_id;
                    tok->merge_hash[e].priority = tok->merges[i].priority;
                    tok->merge_hash[e].valid = 1;
                    if (j > 0) collisions++;
                    break;
                }
            }
        }
        printf("  Merge hash: %d entries in %d slots (%d collisions)\n",
               tok->n_merges, tok->merge_hash_size, collisions);
    }
    
    printf("  BOS=%d, EOS=%d, PAD=%d\n", tok->bos_id, tok->eos_id, tok->pad_id);
    return true;
}

// ========== Init from GGUF-embedded tokenizer KV pairs ==========
// Reads tokenizer.ggml.tokens / tokenizer.ggml.merges straight from the
// .gguf file (self-contained — no data/*.bin extraction step needed).

#include "wubu_gguf_tokenizer.h"

bool wubu_tokenizer_init_from_gguf(wubu_tokenizer_t *tok, const char *gguf_path) {
    if (!tok || !gguf_path) return false;
    memset(tok, 0, sizeof(*tok));
    tok->bos_id = -1; tok->eos_id = -1; tok->pad_id = -1;

    wubu_gguf_tokenizer_data_t d;
    if (!wubu_gguf_tokenizer_extract(gguf_path, &d)) return false;

    /* vocab */
    tok->vocab_size = d.n_tokens;
    tok->vocab = (wubu_token_t *)calloc((size_t)d.n_tokens, sizeof(wubu_token_t));
    if (!tok->vocab) { wubu_gguf_tokenizer_free(&d); return false; }
    for (int i = 0; i < d.n_tokens; i++) {
        const char *s = d.tokens[i];
        size_t l = strlen(s);
        if (l > WUBU_TOKENIZER_MAX_TOKEN_BYTES - 1) l = WUBU_TOKENIZER_MAX_TOKEN_BYTES - 1;
        memcpy(tok->vocab[i].bytes, s, l);
        tok->vocab[i].bytes[l] = '\0';
        tok->vocab[i].byte_len = (int)l;
        tok->vocab[i].id = i;
    }
    build_vocab_hash(tok);
    build_byte_token_ids(tok);
    printf("  Vocab: %d tokens loaded (GGUF)\n", tok->vocab_size);

    /* merges */
    tok->n_merges = 0;
    if (d.merges && d.n_merges > 0) {
        tok->merges = (wubu_merge_t *)calloc((size_t)d.n_merges, sizeof(wubu_merge_t));
        if (!tok->merges) { wubu_gguf_tokenizer_free(&d); return false; }
        for (int mi = 0; mi < d.n_merges; mi++) {
            const char *m = d.merges[mi];
            const char *space = strchr(m, ' ');
            if (!space) continue;
            int left_id = find_token_by_string(tok, (const uint8_t *)m, (int)(space - m));
            int right_id = find_token_by_string(tok, (const uint8_t *)(space + 1),
                                                (int)strlen(space + 1));
            uint8_t mb[WUBU_TOKENIZER_MAX_TOKEN_BYTES];
            int llen = (int)(space - m);
            int rlen = (int)strlen(space + 1);
            int mbl = llen + rlen;
            if (mbl > WUBU_TOKENIZER_MAX_TOKEN_BYTES - 1) mbl = WUBU_TOKENIZER_MAX_TOKEN_BYTES - 1;
            /* Cap each copy so the concatenation fits mb[] exactly —
             * LFM2.5-class byte vocab merges can exceed 256 bytes and the
             * uncapped second memcpy was a stack overflow (glibc fortify). */
            int cl = llen < mbl ? llen : mbl;
            int cr = mbl - cl;
            memcpy(mb, m, (size_t)cl);
            memcpy(mb + cl, space + 1, (size_t)cr);
            int merged_id = find_token_by_string(tok, mb, mbl);
            if (left_id >= 0 && right_id >= 0 && merged_id >= 0) {
                tok->merges[tok->n_merges].left_id = left_id;
                tok->merges[tok->n_merges].right_id = right_id;
                tok->merges[tok->n_merges].new_id = merged_id;
                tok->merges[tok->n_merges].priority = mi;
                tok->n_merges++;
            }
        }
        printf("  Merges: %d loaded (%d resolved, GGUF)\n", d.n_merges, tok->n_merges);
    }

    /* special tokens */
    tok->bos_id = d.bos_id; tok->eos_id = d.eos_id; tok->pad_id = d.pad_id;

    /* merge hash */
    if (tok->n_merges > 0) {
        tok->merge_hash_size = 1;
        while (tok->merge_hash_size < tok->n_merges * 2) tok->merge_hash_size *= 2;
        tok->merge_hash = (wubu_merge_hash_entry_t *)calloc(tok->merge_hash_size, sizeof(wubu_merge_hash_entry_t));
        if (!tok->merge_hash) { wubu_gguf_tokenizer_free(&d); return false; }
        int collisions = 0;
        for (int i = 0; i < tok->n_merges; i++) {
            uint32_t h = merge_hash_key(tok->merges[i].left_id, tok->merges[i].right_id);
            int idx = (int)(h & (uint32_t)(tok->merge_hash_size - 1));
            for (int j = 0; j < tok->merge_hash_size; j++) {
                int e = (idx + j) & (tok->merge_hash_size - 1);
                if (!tok->merge_hash[e].valid) {
                    tok->merge_hash[e].left_id = tok->merges[i].left_id;
                    tok->merge_hash[e].right_id = tok->merges[i].right_id;
                    tok->merge_hash[e].merged_id = tok->merges[i].new_id;
                    tok->merge_hash[e].priority = tok->merges[i].priority;
                    tok->merge_hash[e].valid = 1;
                    if (j > 0) collisions++;
                    break;
                }
            }
        }
        printf("  Merge hash: %d entries in %d slots (%d collisions)\n",
               tok->n_merges, tok->merge_hash_size, collisions);
    }

    printf("  BOS=%d, EOS=%d, PAD=%d (GGUF)\n", tok->bos_id, tok->eos_id, tok->pad_id);
    wubu_gguf_tokenizer_free(&d);
    return true;
}

// ========== Init from text files ==========
static void build_byte_token_ids(wubu_tokenizer_t *tok) {
    // Look up each byte's character in the vocab (matches merge table encoding).
    // Convention 1: the byte AS the codepoint (Latin-1) — matches Llama/Qwen-style
    //   tokenizers whose merges use the literal space and ASCII characters.
    // Convention 2: the GPT-2 byte-level map (byte b -> U+0100+b) — matches
    //   GPT-2-style tokenizers whose space is "Ġ" (U+0120) and whose byte
    //   tokens are the U+0100-U+01FF chars (LFM2.5, etc.).
    // Fallback: the Qwen3.6 byte-token ids for tokenizers with no byte tokens
    //   in the vocab at all.
    for (int i = 0; i < 256; i++) {
        uint8_t utf8[4];
        int ulen;
        int tid = -1;
        if (i < 0x80) {
            utf8[0] = (uint8_t)i;
            ulen = 1;
        } else if (i < 0x800) {
            utf8[0] = 0xC0 | (uint8_t)(i >> 6);
            utf8[1] = 0x80 | (uint8_t)(i & 0x3F);
            ulen = 2;
        } else {
            utf8[0] = 0xE0 | (uint8_t)(i >> 12);
            utf8[1] = 0x80 | (uint8_t)((i >> 6) & 0x3F);
            utf8[2] = 0x80 | (uint8_t)(i & 0x3F);
            ulen = 3;
        }
        tid = find_token_by_string(tok, utf8, ulen);
        if (tid < 0) {
            int cp = 0x100 + i;                 /* GPT-2 byte-level unicode */
            if (cp < 0x800) {
                utf8[0] = 0xC0 | (uint8_t)(cp >> 6);
                utf8[1] = 0x80 | (uint8_t)(cp & 0x3F);
                ulen = 2;
            } else {
                utf8[0] = 0xE0 | (uint8_t)(cp >> 12);
                utf8[1] = 0x80 | (uint8_t)((cp >> 6) & 0x3F);
                utf8[2] = 0x80 | (uint8_t)(cp & 0x3F);
                ulen = 3;
            }
            tid = find_token_by_string(tok, utf8, ulen);
        }
        if (tid < 0) {
            tid = gpt2_byte_encoder[i];  // Qwen3.6-style byte-token fallback
        }
        tok->byte_token_ids[i] = tid;
    }
    // Build decode lookup table and is_byte_token
    tok->is_byte_token = (bool *)calloc((size_t)tok->vocab_size, sizeof(bool));
    for (int i = 0; i < 256; i++) {
        int tid = tok->byte_token_ids[i];
        if (tid >= 0 && tid < tok->vocab_size) {
            tok->is_byte_token[tid] = true;
            // Store the actual text bytes for this byte's token (for decode)
            int blen = tok->vocab[tid].byte_len;
            if (blen > 0 && blen <= 4) {
                memcpy(tok->byte_text_bytes[i], tok->vocab[tid].bytes, blen);
                tok->byte_text_len[i] = blen;
            }
        }
    }
    printf("  byte_token_ids[0xE4]=%d, [0xBD]=%d, [0xA0]=%d\n",
           tok->byte_token_ids[0xE4], tok->byte_token_ids[0xBD], tok->byte_token_ids[0xA0]);
}

static void build_vocab_hash(wubu_tokenizer_t *tok) {
    tok->vocab_hash_size = 1;
    while (tok->vocab_hash_size < tok->vocab_size * 2) tok->vocab_hash_size *= 2;
    tok->vocab_hash = (wubu_vocab_hash_entry_t *)calloc(tok->vocab_hash_size, sizeof(wubu_vocab_hash_entry_t));
    
    for (int i = 0; i < tok->vocab_size; i++) {
        const uint8_t *bytes = (const uint8_t *)tok->vocab[i].bytes;
        int blen = tok->vocab[i].byte_len;
        if (blen <= 0) continue;
        
        uint32_t h = 2166136261u;
        int n = blen < 8 ? blen : 8;
        for (int j = 0; j < n; j++)
            h = (h ^ (uint32_t)bytes[j]) * 16777619u;
        
        int idx = (int)(h & (uint32_t)(tok->vocab_hash_size - 1));
        for (int j = 0; j < tok->vocab_hash_size; j++) {
            int e = (idx + j) & (tok->vocab_hash_size - 1);
            if (!tok->vocab_hash[e].valid) {
                tok->vocab_hash[e].token_id = i;
                tok->vocab_hash[e].valid = 1;
                break;
            }
        }
    }
}

static void build_merge_hash_from_arrays(wubu_tokenizer_t *tok, char **merge_strings, int n_merges) {
    if (!tok || !merge_strings || n_merges <= 0) return;
    
    // First pass: resolve string pairs to token IDs
    tok->merges = (wubu_merge_t *)calloc(n_merges, sizeof(wubu_merge_t));
    tok->n_merges = 0;
    
    for (int mi = 0; mi < n_merges; mi++) {
        char *s = merge_strings[mi];
        char *space = strchr(s, ' ');
        if (!space) continue;
        
        *space = '\0';
        const char *left_str = s;
        const char *right_str = space + 1;
        int left_len = (int)strlen(left_str);
        int right_len = (int)strlen(right_str);
        
        int left_id = find_token_by_string(tok, (const uint8_t *)left_str, left_len);
        int right_id = find_token_by_string(tok, (const uint8_t *)right_str, right_len);
        
        // Concatenated string = merged token
        uint8_t merged_bytes[WUBU_TOKENIZER_MAX_TOKEN_BYTES];
        int mblen = left_len + right_len;
        if (mblen > WUBU_TOKENIZER_MAX_TOKEN_BYTES - 1) mblen = WUBU_TOKENIZER_MAX_TOKEN_BYTES - 1;
        memcpy(merged_bytes, left_str, left_len);
        memcpy(merged_bytes + left_len, right_str, right_len);
        int merged_id = find_token_by_string(tok, merged_bytes, mblen);
        
        *space = ' '; // restore
        
        if (left_id >= 0 && right_id >= 0 && merged_id >= 0) {
            tok->merges[tok->n_merges].left_id = left_id;
            tok->merges[tok->n_merges].right_id = right_id;
            tok->merges[tok->n_merges].new_id = merged_id;
            tok->merges[tok->n_merges].priority = mi;
            tok->n_merges++;
        }
    }
    
    // Build hash table
    tok->merge_hash_size = 1;
    while (tok->merge_hash_size < tok->n_merges * 2) tok->merge_hash_size *= 2;
    tok->merge_hash = (wubu_merge_hash_entry_t *)calloc(tok->merge_hash_size, sizeof(wubu_merge_hash_entry_t));
    
    int collisions = 0;
    for (int i = 0; i < tok->n_merges; i++) {
        uint32_t h = merge_hash_key(tok->merges[i].left_id, tok->merges[i].right_id);
        int idx = (int)(h & (uint32_t)(tok->merge_hash_size - 1));
        int placed = 0;
        for (int j = 0; j < tok->merge_hash_size; j++) {
            int e = (idx + j) & (tok->merge_hash_size - 1);
            if (!tok->merge_hash[e].valid) {
                tok->merge_hash[e].left_id = tok->merges[i].left_id;
                tok->merge_hash[e].right_id = tok->merges[i].right_id;
                tok->merge_hash[e].merged_id = tok->merges[i].new_id;
                tok->merge_hash[e].priority = tok->merges[i].priority;
                tok->merge_hash[e].valid = 1;
                if (j > 0) collisions++;
                placed = 1;
                break;
            }
        }
    }
    printf("  Merge hash: %d entries in %d slots (%d collisions)\n",
           tok->n_merges, tok->merge_hash_size, collisions);
}

// ========== Init from text files ==========

bool wubu_tokenizer_init_from_files(wubu_tokenizer_t *tok,
                                     const char *vocab_path,
                                     const char *merges_path,
                                     int bos_id,
                                     int eos_id,
                                     int pad_id) {
    if (!tok || !vocab_path || !merges_path) return false;
    memset(tok, 0, sizeof(*tok));
    init_byte_encoder();
    tok->bos_id = bos_id; tok->eos_id = eos_id; tok->pad_id = pad_id;
    
    // Count lines
    int vs = 0; char buf[4096];
    FILE *f = fopen(vocab_path, "r");
    if (!f) return false;
    while (fgets(buf,sizeof(buf),f)) vs++;
    fclose(f);
    
    int ms = 0;
    f = fopen(merges_path, "r");
    if (!f) return false;
    while (fgets(buf,sizeof(buf),f)) ms++;
    fclose(f);
    
    tok->vocab = (wubu_token_t *)calloc(vs, sizeof(wubu_token_t));
    tok->vocab_size = vs;
    tok->merges = (wubu_merge_t *)calloc(ms, sizeof(wubu_merge_t));
    tok->n_merges = ms;
    
    // Read vocab
    f = fopen(vocab_path, "r");
    int idx = 0;
    while (fgets(buf,sizeof(buf),f) && idx < vs) {
        char *sep = strchr(buf, ' ');
        if (sep) {
            *sep = '\0'; idx = atoi(buf);
            char *str = sep+1; int slen = (int)strlen(str);
            if (slen > 0 && str[slen-1]=='\n') str[--slen]='\0';
            if (slen > WUBU_TOKENIZER_MAX_TOKEN_BYTES-1) slen = WUBU_TOKENIZER_MAX_TOKEN_BYTES-1;
            memcpy(tok->vocab[idx].bytes, str, slen);
            tok->vocab[idx].bytes[slen] = '\0';
            tok->vocab[idx].byte_len = slen; tok->vocab[idx].id = idx;
        }
    }
    fclose(f);
    
    // Read merges
    f = fopen(merges_path, "r");
    int mi = 0;
    while (fgets(buf,sizeof(buf),f) && mi < ms) {
        int left, right;
        if (sscanf(buf, "%d %d", &left, &right) == 2) {
            if (left < vs && right < vs) {
                uint8_t mb[WUBU_TOKENIZER_MAX_TOKEN_BYTES];
                int mbl = tok->vocab[left].byte_len + tok->vocab[right].byte_len;
                if (mbl > WUBU_TOKENIZER_MAX_TOKEN_BYTES-1) mbl = WUBU_TOKENIZER_MAX_TOKEN_BYTES-1;
                memcpy(mb, tok->vocab[left].bytes, tok->vocab[left].byte_len);
                memcpy(mb+tok->vocab[left].byte_len, tok->vocab[right].bytes, tok->vocab[right].byte_len);
                int mid = find_token_by_string(tok, mb, mbl);
                tok->merges[mi].left_id = left; tok->merges[mi].right_id = right;
                tok->merges[mi].new_id = mid>=0 ? mid : left;
                tok->merges[mi].priority = mi; mi++;
            }
        }
    }
    fclose(f);
    tok->n_merges = mi;
    
    build_vocab_hash(tok);
    build_byte_token_ids(tok);
    
    // Build merge hash
    tok->merge_hash_size = 1;
    while (tok->merge_hash_size < tok->n_merges * 2) tok->merge_hash_size *= 2;
    tok->merge_hash = (wubu_merge_hash_entry_t *)calloc(tok->merge_hash_size, sizeof(wubu_merge_hash_entry_t));
    for (int i = 0; i < tok->n_merges; i++) {
        uint32_t h = merge_hash_key(tok->merges[i].left_id, tok->merges[i].right_id);
        int idx = (int)(h & (uint32_t)(tok->merge_hash_size - 1));
        for (int j = 0; j < tok->merge_hash_size; j++) {
            int e = (idx+j) & (tok->merge_hash_size-1);
            if (!tok->merge_hash[e].valid) {
                tok->merge_hash[e].left_id = tok->merges[i].left_id;
                tok->merge_hash[e].right_id = tok->merges[i].right_id;
                tok->merge_hash[e].merged_id = tok->merges[i].new_id;
                tok->merge_hash[e].priority = tok->merges[i].priority;
                tok->merge_hash[e].valid = 1; break;
            }
        }
    }
    return true;
}

void wubu_tokenizer_free(wubu_tokenizer_t *tok) {
    if (tok) {
        free(tok->vocab); free(tok->merges);
        free(tok->merge_hash); free(tok->vocab_hash);
        free(tok->is_byte_token);
        memset(tok, 0, sizeof(*tok));
    }
}

// ========== Encode ==========

int wubu_tokenizer_encode(wubu_tokenizer_t *tok,
                          const char *text,
                          int *output_ids,
                          int max_output_ids) {
    if (!tok || !text || !output_ids) return -1;
    int text_len = (int)strlen(text);
    if (text_len <= 0) return 0;
    
    // Convert text to byte tokens
    int *byte_tokens = (int *)malloc((size_t)text_len * sizeof(int));
    int n_bytes = 0;
    for (int i = 0; i < text_len; i++) {
        int tid = tok->byte_token_ids[(uint8_t)text[i]];
        if (tid >= 0) byte_tokens[n_bytes++] = tid;
    }
    if (n_bytes <= 0) { free(byte_tokens); return 0; }
    
    int *work_ids = (int *)malloc((size_t)n_bytes * 2 * sizeof(int));
    char *work_merged = (char *)malloc((size_t)n_bytes * 2);
    int work_len = n_bytes;
    for (int i = 0; i < n_bytes; i++) work_ids[i] = byte_tokens[i];
    memset(work_merged, 0, (size_t)work_len);
    
    int changed = 1;
    while (changed && work_len > 1) {
        changed = 0;
        int best_prio = -1, best_pos = -1, best_right = -1, best_mid = -1;
        
        for (int i = 0; i < work_len - 1; i++) {
            if (work_merged[i]) continue;
            int j = i + 1;
            while (j < work_len && work_merged[j]) j++;
            if (j >= work_len) break;
            
            int merged_id;
            int prio = find_merge(tok, work_ids[i], work_ids[j], &merged_id);
            if (prio >= 0 && (best_prio < 0 || prio < best_prio)) {
                best_prio = prio; best_pos = i; best_right = j; best_mid = merged_id;
            }
        }
        
        if (best_pos >= 0 && best_mid >= 0) {
            work_merged[best_right] = 1;
            work_ids[best_pos] = best_mid;
            changed = 1;
        }
    }
    
    int total = 0;
    for (int i = 0; i < work_len; i++)
        if (!work_merged[i] && total < max_output_ids)
            output_ids[total++] = work_ids[i];
    
    free(byte_tokens); free(work_ids); free(work_merged);
    return total;
}

// ========== Decode (with GPT-2 byte encoding fix) ==========

int wubu_tokenizer_decode(wubu_tokenizer_t *tok,
                          const int *input_ids,
                          int n_ids,
                          char *output_text,
                          int max_output_chars) {
    if (!input_ids || !output_text || max_output_chars <= 0) return -1;
    
    int total = 0;
    for (int i = 0; i < n_ids && total < max_output_chars - 1; i++) {
        int id = input_ids[i];
        if (id >= 0 && id < tok->vocab_size) {
            // For byte-level tokens: find which byte this token represents.
            // The byte must match EXACTLY (token text == the byte's canonical
            // text); a token like vocab[0] (a 20-byte "Ċ"x10 string) can be
            // wrongly flagged via the Qwen3.6 fallback ids — without the
            // exact-match check the first byte whose fallback id equals the
            // token (e.g. 0x7F) hijacks the decode.
            if (tok->is_byte_token && tok->is_byte_token[id]) {
                int byte_val = -1;
                int blen = tok->vocab[id].byte_len;
                for (int b = 0; b < 256; b++) {
                    if (tok->byte_token_ids[b] == id &&
                        tok->byte_text_len[b] == blen &&
                        memcmp(tok->byte_text_bytes[b], tok->vocab[id].bytes, blen) == 0) {
                        byte_val = b;
                        break;
                    }
                }
                if (byte_val >= 0) {
                    output_text[total++] = (char)(uint8_t)byte_val;
                    continue;
                }
            }
            
            // For non-byte tokens: scan for known byte token texts
            const uint8_t *bytes = (const uint8_t *)tok->vocab[id].bytes;
            int blen = tok->vocab[id].byte_len;
            for (int j = 0; j < blen && total < max_output_chars - 1; ) {
                // Find the LONGEST matching byte-level token text at this position
                int best_bv = -1, best_len = 0;
                for (int bv = 0; bv < 256; bv++) {
                    int tlen = tok->byte_text_len[bv];
                    if (tlen > best_len && j + tlen <= blen) {
                        if (memcmp(&bytes[j], tok->byte_text_bytes[bv], tlen) == 0) {
                            best_bv = bv;
                            best_len = tlen;
                        }
                    }
                }
                if (best_bv >= 0) {
                    output_text[total++] = (char)(uint8_t)best_bv;
                    j += best_len;
                } else {
                    output_text[total++] = (char)bytes[j];
                    j++;
                }
            }
        }
    }
    output_text[total] = '\0';
    return total;
}

// ========== Python bridge (fallback) ==========

int wubu_tokenizer_encode_python(wubu_tokenizer_t *tok,
                                 const char *text,
                                 int *output_ids,
                                 int max_output_ids) {
    char script_path[] = "/tmp/wubu_tokenize_XXXXXX";
    int fd = mkstemp(script_path);
    if (fd < 0) return -1;
    
    char script[66000];
    int slen = snprintf(script, sizeof(script),
        "import sys, json\n"
        "from transformers import AutoTokenizer\n"
        "t = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-35B-A3B', trust_remote_code=True)\n"
        "text = sys.argv[1]\n"
        "ids = t.encode(text)\n"
        "print(' '.join(str(i) for i in ids))\n");
    write(fd, script, slen);
    close(fd);
    
    char *py_argv[] = {
        "python3", script_path, NULL
    };

    int rc = -1;
    char result[65536];
    int captured = wubu_spawn_capture("python3", py_argv, result, sizeof(result), &rc);
    unlink(script_path);
    if (captured < 0 || rc != 0) return -1;
    
    int n = 0;
    char *token = strtok(result, " \n");
    while (token && n < max_output_ids) {
        output_ids[n++] = atoi(token);
        token = strtok(NULL, " \n");
    }
    return n;
}
