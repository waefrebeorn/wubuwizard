#ifndef MTP_Q8_CACHE_H
#define MTP_Q8_CACHE_H

#include "gguf_reader.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Raw-quant cache: stores original IQ2_XXS/IQ3_XXS bytes per expert in RAM.
// No dequant/requant — keeps the native quant format. Uses original vec_dot path.
// Key = expert_index, Value = 3 raw quantized blobs + type info.
// Benefits vs Q8_K cache: no ~3% acceptance loss, ~1/4 memory, no dequant overhead.
#define MTP_IQ_CACHE_SLOTS 16  // 8 routed + 1 shared + 7 spare (extra headroom)

// Per-expert raw quantized weight data (variable-length, capped at max possible)
#define MTP_IQ_MAX_EXP_BYTES 524288  // > max needed (IQ2: 270KB, IQ3: 401KB, IQ4: 545KB)

typedef struct {
    int   key;               // expert index (-1 = empty)
    int   lru_tick;           // last access tick
    int   gate_type;          // GGML type for gate
    int   up_type;            // GGML type for up
    int   down_type;          // GGML type for down
    int   gate_bytes;         // actual bytes stored for gate
    int   up_bytes;           // actual bytes stored for up
    int   down_bytes;         // actual bytes stored for down
    uint8_t gate_q[MTP_IQ_MAX_EXP_BYTES];
    uint8_t up_q[MTP_IQ_MAX_EXP_BYTES];
    uint8_t down_q[MTP_IQ_MAX_EXP_BYTES];
} mtp_iq_cache_slot_t;

typedef struct {
    mtp_iq_cache_slot_t slots[MTP_IQ_CACHE_SLOTS];
    int tick;
} mtp_iq_cache_t;

// Initialize cache
static inline void mtp_iq_cache_init(mtp_iq_cache_t *cache) {
    memset(cache, 0, sizeof(*cache));
    for (int i = 0; i < MTP_IQ_CACHE_SLOTS; i++)
        cache->slots[i].key = -1;
    cache->tick = 0;
}

// Find slot for given key, or evict LRU
// was_miss: set to true if a new slot was allocated (caller must fill)
static inline mtp_iq_cache_slot_t *mtp_iq_cache_get_slot(mtp_iq_cache_t *cache, int key, bool *was_miss) {
    cache->tick++;
    
    // Check for existing entry
    for (int i = 0; i < MTP_IQ_CACHE_SLOTS; i++) {
        if (cache->slots[i].key == key) {
            cache->slots[i].lru_tick = cache->tick;
            *was_miss = false;
            return &cache->slots[i];
        }
    }
    
    // Find LRU slot
    int lru_idx = 0;
    int min_tick = cache->slots[0].lru_tick;
    for (int i = 1; i < MTP_IQ_CACHE_SLOTS; i++) {
        if (cache->slots[i].lru_tick < min_tick) {
            min_tick = cache->slots[i].lru_tick;
            lru_idx = i;
        }
    }
    
    mtp_iq_cache_slot_t *slot = &cache->slots[lru_idx];
    slot->key = key;
    slot->lru_tick = cache->tick;
    *was_miss = true;
    return slot;
}

// Fill slot with raw quantized bytes (memcpy from blob — no dequant/requant)
// Returns 1 on success, 0 on failure
static inline int mtp_iq_cache_fill(mtp_iq_cache_slot_t *slot,
                                     const moe_weights_t *moe,
                                     int e) {
    if (!moe->ffn_gate_exps_q || !moe->ffn_up_exps_q || !moe->ffn_down_exps_q)
        return 0;
    
    // Compute bytes per expert for each quant type
    int64_t gate_bytes = gguf_raw_size(moe->ffn_gate_exps_q_type, (int64_t)D_MODEL * D_FF);
    int64_t up_bytes   = gguf_raw_size(moe->ffn_up_exps_q_type,   (int64_t)D_MODEL * D_FF);
    int64_t down_bytes = gguf_raw_size(moe->ffn_down_exps_q_type, (int64_t)D_FF * D_MODEL);
    
    if (gate_bytes <= 0 || up_bytes <= 0 || down_bytes <= 0) return 0;
    if (gate_bytes > MTP_IQ_MAX_EXP_BYTES || up_bytes > MTP_IQ_MAX_EXP_BYTES || down_bytes > MTP_IQ_MAX_EXP_BYTES)
        return 0;
    
    // Memcpy raw quantized bytes from blob
    const uint8_t *src_gate = moe->ffn_gate_exps_q + (int64_t)e * gate_bytes;
    const uint8_t *src_up   = moe->ffn_up_exps_q   + (int64_t)e * up_bytes;
    const uint8_t *src_down = moe->ffn_down_exps_q + (int64_t)e * down_bytes;
    
    memcpy(slot->gate_q, src_gate, (size_t)gate_bytes);
    memcpy(slot->up_q,   src_up,   (size_t)up_bytes);
    memcpy(slot->down_q, src_down, (size_t)down_bytes);
    
    slot->gate_type = moe->ffn_gate_exps_q_type;
    slot->up_type   = moe->ffn_up_exps_q_type;
    slot->down_type = moe->ffn_down_exps_q_type;
    slot->gate_bytes = (int)gate_bytes;
    slot->up_bytes   = (int)up_bytes;
    slot->down_bytes = (int)down_bytes;
    
    return 1;
}

#endif // MTP_Q8_CACHE_H
