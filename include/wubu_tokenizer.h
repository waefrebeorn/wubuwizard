#ifndef WUBU_TOKENIZER_H
#define WUBU_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum values
#define WUBU_TOKENIZER_MAX_VOCAB 248320
#define WUBU_TOKENIZER_MAX_MERGES 248320
#define WUBU_TOKENIZER_MAX_SPECIAL 32
#define WUBU_TOKENIZER_MAX_TOKEN_BYTES 256
#define WUBU_TOKENIZER_MAX_INPUT_CHARS 65536
#define WUBU_TOKENIZER_MAX_PIECES 65536

// Token type
typedef struct {
    int id;
    char bytes[WUBU_TOKENIZER_MAX_TOKEN_BYTES];
    int byte_len;
    bool is_special;
} wubu_token_t;

// Merge rule: two tokens are merged into one
typedef struct {
    int left_id;
    int right_id;
    int new_id;
    int priority;  // lower = higher priority (order in merge file)
} wubu_merge_t;

// Hash table entry for O(1) merge lookup
// Key: (left_id, right_id) pair → merge info
typedef struct {
    int left_id;
    int right_id;
    int merged_id;
    int priority;
    int valid;  // 1 if entry is occupied
} wubu_merge_hash_entry_t;

// Hash table for O(1) vocab string lookup (for decoding)
typedef struct {
    int token_id;
    int valid;
} wubu_vocab_hash_entry_t;

// Tokenizer context
typedef struct {
    // Vocab: id -> token string (heap-allocated)
    wubu_token_t *vocab;  // [vocab_size]
    wubu_merge_t *merges; // [n_merges] — still needed for iteration
    int vocab_size;
    int n_merges;
    
    // Hash table for merge lookup: key = (left_id << 16) | right_id
    // Size: next power of 2 > n_merges * 2
    wubu_merge_hash_entry_t *merge_hash;
    int merge_hash_size;
    
    // Hash table for vocab string → token_id (for merged token lookup)
    // Simple: hash of first few bytes + length
    wubu_vocab_hash_entry_t *vocab_hash;
    int vocab_hash_size;
    
    // Special tokens (by name and id)
    int n_special;
    int special_ids[WUBU_TOKENIZER_MAX_SPECIAL];
    char special_names[WUBU_TOKENIZER_MAX_SPECIAL][64];
    
    // BOS/EOS/PAD
    int bos_id;
    int eos_id;
    int pad_id;
    
    // Byte-to-token lookup for bytes 0-255
    int byte_token_ids[256];  // maps byte -> token id
    bool *is_byte_token;      // token ID -> bool (allocated at init)
    uint8_t byte_text_bytes[256][4];  // raw text bytes for each byte value (for decode)
    int byte_text_len[256];           // length of text bytes for each byte value
} wubu_tokenizer_t;

// Initialize tokenizer from pre-extracted vocab/merges text files
bool wubu_tokenizer_init_from_files(wubu_tokenizer_t *tok,
                                     const char *vocab_path,
                                     const char *merges_path,
                                     int bos_id,
                                     int eos_id,
                                     int pad_id);

// Initialize tokenizer from GGUF file
bool wubu_tokenizer_init(wubu_tokenizer_t *tok, const char *gguf_path);

// Initialize tokenizer from the GGUF's own tokenizer.ggml.* KV pairs
// (self-contained; no data/*.bin extraction needed)
bool wubu_tokenizer_init_from_gguf(wubu_tokenizer_t *tok, const char *gguf_path);

// Free tokenizer resources
void wubu_tokenizer_free(wubu_tokenizer_t *tok);

// Encode text to token IDs
// Returns number of tokens written, or -1 on error
int wubu_tokenizer_encode(wubu_tokenizer_t *tok,
                          const char *text,
                          int *output_ids,
                          int max_output_ids);

// Decode token IDs back to text
// Returns number of characters written, or -1 on error
int wubu_tokenizer_decode(wubu_tokenizer_t *tok,
                          const int *input_ids,
                          int n_ids,
                          char *output_text,
                          int max_output_chars);

// Quick encode using Python subprocess (fallback)
int wubu_tokenizer_encode_python(wubu_tokenizer_t *tok,
                                 const char *text,
                                 int *output_ids,
                                 int max_output_ids);

#ifdef __cplusplus
}
#endif

#endif // WUBU_TOKENIZER_H
