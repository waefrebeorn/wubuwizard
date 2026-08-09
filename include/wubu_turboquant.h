// Q2_0 and TurboQuant+/RotorQuant types for KV cache

#ifndef WUBU_TURBOQUANT_H
#define WUBU_TURBOQUANT_H

#include <stdint.h>
#include <cuda_fp16.h>

#define KV_BLOCK_SIZE 16
#define TILE_SIZE 64  // 4 blocks = 64 tokens per tile

// ================================================================
// Q4_0 Block (existing - for K cache)
// 32 elements, fp16 scale, 4-bit signed [-8, 7]
// 18 bytes per 32 elements
// ================================================================
typedef struct {
    uint16_t d;      // fp16 scale
    uint8_t qs[16];  // 32 × 4-bit nibbles
} block_q4_0_t;

// Q4_0 quantization (symmetric signed)
static inline void quantize_q4_0_block(const float* x, block_q4_0_t* b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) amax = ax;
    }
    if (amax == 0.0f) {
        b->d = 0;
        memset(b->qs, 0, 16);
        return;
    }
    const float d = amax / 7.0f;  // [-7, 7] → [1, 15]
    const float id = 1.0f / d;
    b->d = __float2half_rn(d);
    for (int i = 0; i < 32; i++) {
        int q = (int)(x[i] * id + 8.0f);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        b->qs[i / 2] |= (uint8_t)(q << (4 * (i % 2)));
    }
}

static inline void dequantize_q4_0_block(const block_q4_0_t* b, float* x) {
    const float d = __half2float(b->d);
    for (int i = 0; i < 32; i++) {
        int q = (b->qs[i / 2] >> (4 * (i % 2))) & 0xF;
        x[i] = ((float)q - 8.0f) * d;
    }
}

// ================================================================
// Q2_0 Block (new - for V cache)
// 32 elements, fp16 scale, 2-bit signed [-2, 1]
// 10 bytes per 32 elements (5.12:1 compression vs FP16)
// ================================================================
typedef struct {
    uint16_t d;      // fp16 scale
    uint8_t qs[8];   // 32 × 2-bit values (4 per byte)
} block_q2_0_t;

static inline void quantize_q2_0_block(const float* x, block_q2_0_t* b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float ax = fabsf(x[i]);
        if (ax > amax) amax = ax;
    }
    if (amax == 0.0f) {
        b->d = 0;
        memset(b->qs, 0, 8);
        return;
    }
    const float d = amax / 1.0f;  // [-1, 1] → [1, 2]
    const float id = 1.0f / d;
    b->d = __float2half_rn(d);
    for (int i = 0; i < 32; i++) {
        int q = (int)(x[i] * id + 2.0f);  // bias = 2
        if (q < 0) q = 0;
        if (q > 3) q = 3;
        b->qs[i / 4] |= (uint8_t)(q << (2 * (i % 4)));
    }
}

static inline void dequantize_q2_0_block(const block_q2_0_t* b, float* x) {
    const float d = __half2float(b->d);
    for (int i = 0; i < 32; i++) {
        int q = (b->qs[i / 4] >> (2 * (i % 4))) & 0x3;
        x[i] = ((float)q - 2.0f) * d;
    }
}

// TurboQuant: extract 2 elements at once for V dequant

#if defined(__CUDACC__)
__device__ __forceinline__ float2 dequant_q2_0_pair(const block_q2_0_t* b, int idx) {
    int q0 = (b->qs[idx / 4] >> (2 * (idx % 4))) & 0x3;
    int q1 = (b->qs[(idx+1) / 4] >> (2 * ((idx+1) % 4))) & 0x3;
    float scale = __half2float(b->d);
    return make_float2((q0 - 2.0f) * scale, (q1 - 2.0f) * scale);
}

// ================================================================
// Walsh-Hadamard Transform (in-place, for TurboQuant pre-rotation)
// ================================================================
__device__ __forceinline__ void wht32(float* x) {
    // In-place WHT for 32 elements - 5 stages
    // Can fuse with load/store for efficiency
    #pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
        for (int i = threadIdx.x; i < 32; i += blockDim.x * 2) {
            int base = (i / (stride * 2)) * stride * 2;
            int j = base + (i % stride);
            if (j + stride < 32) {
                float u = x[j];
                float v = x[j + stride];
                x[j] = u + v;
                x[j + stride] = u - v;
            }
        }
        __syncthreads();
    }
}

// ================================================================
// RotorQuant: Givens Rotation Tables (pre-computed per layer)
// ================================================================
// 2×2 Givens for K cache (PlanarQuant)
// Per 32-element block: 16 independent 2×2 rotations
// Pre-compute cos/sin per layer per pair

#define ROTOR_KV_HEADS 2
#define ROTOR_HEAD_DIM 256
#define ROTOR_PAIRS_PER_BLOCK 16  // 32 elements / 2

typedef struct {
    float cos[ROTOR_KV_HEADS][ROTOR_HEAD_DIM / 2];  // [2][128]
    float sin[ROTOR_KV_HEADS][ROTOR_HEAD_DIM / 2];
    // For quaternion V: 4 params per 4-element group
    float quat[ROTOR_KV_HEADS][ROTOR_HEAD_DIM / 4][4];  // [2][64][4]
} rotor_tables_t;

// ================================================================
// Framebuffer Tile Structure
// ================================================================
typedef struct {
    int tile_id;              // 0..8191 for 512k/64
    int layer;                // 0..39
    int kv_head;              // 0..1 (2 KV heads)
    int block_start;          // First block in tile (4 blocks = 64 tokens)
    void* k_tile_ptr;         // Contiguous K tiles in pool
    void* v_tile_ptr;         // Contiguous V tiles in pool
    bool is_resident;         // In VRAM or swapped
    int last_access_frame;    // LRU eviction
} kv_tile_t;

#define MAX_TILES (524288 / 64)  // 8192 tiles for 512k context
#define TILES_PER_WINDOW (8192 / 64)  // 128 tiles for 8k window

#endif /* __CUDACC__ — tile_manager below is CPU */

typedef struct {
    kv_tile_t tiles[MAX_TILES];
    int free_list[MAX_TILES];
    int free_count;
    int free_capacity;
    
    // Per-sequence tile tables
    int** tile_tables;      // [batch][max_tiles_per_seq]
    int* tile_table_sizes;
    int batch_capacity;
    
    // GPU tile pools
    void* d_k_tile_pool;
    void* d_v_tile_pool;
    size_t tile_pool_bytes;
    
    // Frame planning
    int current_frame;
    int window_tiles;
} tile_manager_t;

// Initialize tile manager
tile_manager_t* tile_manager_init(int max_ctx, int block_size, int n_layers, int n_kv_heads);
void tile_manager_free(tile_manager_t* mgr);
int tile_manager_alloc_tile(tile_manager_t* mgr, int layer, int kv_head);
void tile_manager_free_tile(tile_manager_t* mgr, int tile_id);
void tile_manager_plan_frame(tile_manager_t* mgr, int req_id, int current_pos, int window_size);

#endif // WUBU_TURBOQUANT_H