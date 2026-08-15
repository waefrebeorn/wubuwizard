/*
 * wubu_ssd_moe.c -- SSD-paged MoE slot-bank (ds4-ssd technique).
 * See include/wubu_ssd_moe.h. Self-contained; C11; opaque ctx.
 *
 * Slot-bank design (mirrors Anemll ds4-ssd / Apple "LLM in a flash"):
 *   - Dense / shared tensors stay resident in RAM (loaded by the caller).
 *   - Routed MoE expert weights are PAGED ON DEMAND FROM THE SOURCE
 *     CHECKPOINT SHARDS (the same *.safetensors the model already loads),
 *     NEVER a redundant copy. The checkpoint is already mmap'd / readable
 *     via wubu_shard_raw, so an expert's BF16 gate/up/down tensors are read
 *     straight from disk through the shard index — no 256 GB sidecar, no
 *     extra disk footprint, no duplicate of weights we already have.
 *   - A fixed POOL of expert "slots" per layer is kept resident in RAM.
 *   - On a router miss, the selected expert is paged in (BF16 -> F32) and
 *     the least-recently-used slot is evicted. The OS page cache absorbs
 *     repeated reads, so within a decode pass most experts are already warm.
 */
#include "wubu_ssd_moe.h"
#include "wubu_safetensors_shard.h"
#include "safetensors_reader.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#define SSD_MOE_MAX_LAYERS 256
#define SSD_MOE_MAX_SLOTS  256

/* BF16 <-> F32: use wubu_fp16.h */

typedef struct {
    int      expert;      /* which expert currently occupies this slot, or -1 */
    long     lru;         /* monotonic timestamp; higher = more recently used */
    float   *gate;        /* [d_model*d_ff] F32 resident */
    float   *up;
    float   *down;
} ssd_slot_t;

typedef struct {
    ssd_slot_t slots[SSD_MOE_MAX_SLOTS];
    int        n_slots;   /* = slot_bank */
} ssd_layer_t;

struct wubu_ssd_moe {
    wubu_shard_ctx_t *sc;  /* source checkpoint shards (already on disk) */
    ssd_layer_t layers[SSD_MOE_MAX_LAYERS];
    int n_layers;
    int n_experts;
    int d_model;
    int d_ff;
    int dff_dm;            /* d_ff * d_model (elems per expert matrix) */
    int slot_bank;
    long   stat_pageins;
    long   stat_hits;
    long long stat_bytes;
    long   lru_clock;
    pthread_mutex_t lock;   /* serializes slot-bank page-ins (LRU eviction
                               is NOT safe under the forward's OpenMP threads) */
};

/* Read one expert matrix (BF16) straight from the source shard and dequant
 * to F32 into `out` (elems = d_ff*d_model). Returns 0 on ok, -1 on miss. */
static int read_expert_mat(wubu_ssd_moe_t *m, int layer, int expert,
                           const char *proj, float *out) {
    char nm[256];
    snprintf(nm, sizeof(nm),
             "model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight",
             layer, expert, proj);
    int dt = 0; int64_t row = 0;
    const uint8_t *raw = wubu_shard_raw(m->sc, nm, &dt, &row);
    if (!raw) return -1;
    /* Expert gate/up/down are [d_ff, d_model] BF16. */
    const uint16_t *b = (const uint16_t *)raw;
    int64_t n = m->dff_dm;
    if (dt == ST_DTYPE_F32) {
        const float *f = (const float *)raw;
        for (int64_t i = 0; i < n; i++) out[i] = f[i];
    } else { /* BF16 (most common) or F16 -> use bf16 path for both */
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(b[i]);
    }
    return 0;
}

/* Open the slot-bank over an EXISTING checkpoint directory (no sidecar copy).
 * slot_bank = resident expert slots kept per layer (ds4-ssd --moe-slot-bank). */
wubu_ssd_moe_t *wubu_ssd_moe_open(const char *checkpoint_dir, int slot_bank) {
    return wubu_ssd_moe_open_from_shards(wubu_shard_open(checkpoint_dir), slot_bank);
}

wubu_ssd_moe_t *wubu_ssd_moe_open_from_shards(wubu_shard_ctx_t *sc, int slot_bank) {
    if (!sc) return NULL;
    wubu_ssd_moe_t *m = (wubu_ssd_moe_t *)calloc(1, sizeof(*m));
    if (!m) { wubu_shard_close(sc); return NULL; }
    m->sc = sc;
    m->slot_bank = slot_bank < 1 ? 1 : (slot_bank > SSD_MOE_MAX_SLOTS ? SSD_MOE_MAX_SLOTS : slot_bank);

    /* Derive dims from the real checkpoint tensors (no manifest needed). */
    int D = wubu_shard_dimof(sc,
              "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", 1);
    if (D <= 0) D = wubu_shard_dimof(sc,
              "model.language_model.layers.0.mlp.experts.0.up_proj.weight", 1);
    int dff = wubu_shard_dimof(sc,
              "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", 0);
    if (dff <= 0) dff = D * 4;

    int nE = 0;
    for (int e = 0; e < 512; e++) {
        char nm[256];
        snprintf(nm, sizeof(nm),
                 "model.language_model.layers.0.mlp.experts.%d.gate_proj.weight", e);
        if (!wubu_shard_has(sc, nm)) break;
        nE = e + 1;
    }
    int nL = 0;
    for (int l = 0; l < 256; l++) {
        char nm[256];
        snprintf(nm, sizeof(nm),
                 "model.language_model.layers.%d.mlp.experts.0.gate_proj.weight", l);
        if (!wubu_shard_has(sc, nm)) break;
        nL = l + 1;
    }
    if (D <= 0 || dff <= 0 || nE <= 0 || nL <= 0) { wubu_ssd_moe_close(m); return NULL; }

    m->n_layers = nL; m->n_experts = nE; m->d_model = D; m->d_ff = dff;
    m->dff_dm = D * dff;
    pthread_mutex_init(&m->lock, NULL);

    for (int l = 0; l < m->n_layers && l < SSD_MOE_MAX_LAYERS; l++) {
        ssd_layer_t *sl = &m->layers[l];
        sl->n_slots = m->slot_bank;
        for (int s = 0; s < sl->n_slots; s++) {
            sl->slots[s].expert = -1;
            sl->slots[s].lru = 0;
            sl->slots[s].gate = (float *)malloc((size_t)m->dff_dm * sizeof(float));
            sl->slots[s].up   = (float *)malloc((size_t)m->dff_dm * sizeof(float));
            sl->slots[s].down = (float *)malloc((size_t)m->dff_dm * sizeof(float));
            if (!sl->slots[s].gate || !sl->slots[s].up || !sl->slots[s].down) {
                wubu_ssd_moe_close(m); return NULL;
            }
        }
    }
    return m;
}

int wubu_ssd_moe_get(wubu_ssd_moe_t *m, int layer, int expert, float *out[3]) {
    if (!m || layer < 0 || layer >= m->n_layers || expert < 0 || expert >= m->n_experts)
        return -1;
    ssd_layer_t *sl = &m->layers[layer];

    /* Slot search (under lock — LRU + slot pointers are shared state). */
    pthread_mutex_lock(&m->lock);

    /* Slot search. */
    int hit = -1;
    for (int s = 0; s < sl->n_slots; s++)
        if (sl->slots[s].expert == expert) { hit = s; break; }
    if (hit >= 0) {
        sl->slots[hit].lru = ++m->lru_clock;
        out[0] = sl->slots[hit].gate; out[1] = sl->slots[hit].up; out[2] = sl->slots[hit].down;
        m->stat_hits++;
        pthread_mutex_unlock(&m->lock);
        return 1; /* hit */
    }

    /* Miss: choose LRU slot (or first empty). */
    int victim = 0; long best = sl->slots[0].lru;
    for (int s = 1; s < sl->n_slots; s++)
        if (sl->slots[s].lru < best) { best = sl->slots[s].lru; victim = s; }

    /* Page expert's three matrices from the source checkpoint shards. */
    if (read_expert_mat(m, layer, expert, "gate", sl->slots[victim].gate) != 0 ||
        read_expert_mat(m, layer, expert, "up",   sl->slots[victim].up)   != 0 ||
        read_expert_mat(m, layer, expert, "down", sl->slots[victim].down) != 0) {
        pthread_mutex_unlock(&m->lock);
        return -1;
    }
    m->stat_bytes += (long long)(m->dff_dm * 3 * (m->dff_dm > 0 ? 2 : 2));
    m->stat_pageins++;

    sl->slots[victim].expert = expert;
    sl->slots[victim].lru = ++m->lru_clock;
    out[0] = sl->slots[victim].gate; out[1] = sl->slots[victim].up; out[2] = sl->slots[victim].down;
    pthread_mutex_unlock(&m->lock);
    return 0; /* page-in */
}

int  wubu_ssd_moe_n_layers(const wubu_ssd_moe_t *m){ return m?m->n_layers:0; }
int  wubu_ssd_moe_n_experts(const wubu_ssd_moe_t *m){ return m?m->n_experts:0; }
int  wubu_ssd_moe_d_model(const wubu_ssd_moe_t *m){ return m?m->d_model:0; }
int  wubu_ssd_moe_d_ff(const wubu_ssd_moe_t *m){ return m?m->d_ff:0; }

void wubu_ssd_moe_stats(const wubu_ssd_moe_t *m, long *pageins, long *hits, long long *bytes_read){
    if(pageins)*pageins=m?m->stat_pageins:0;
    if(hits)*hits=m?m->stat_hits:0;
    if(bytes_read)*bytes_read=m?m->stat_bytes:0;
}

void wubu_ssd_moe_close(wubu_ssd_moe_t *m){
    if(!m) return;
    for(int l=0;l<m->n_layers && l<SSD_MOE_MAX_LAYERS;l++){
        ssd_layer_t *sl=&m->layers[l];
        for(int s=0;s<sl->n_slots;s++){ free(sl->slots[s].gate); free(sl->slots[s].up); free(sl->slots[s].down); }
    }
    if (m->sc) wubu_shard_close(m->sc);
    pthread_mutex_destroy(&m->lock);
    free(m);
}

/* ---- Packer (REMOVED: source shards ARE the backing store) ---- */
/* The old packer duplicated every expert weight into a 256 GB sidecar that
 * was 100% redundant with the checkpoint we already load. Paging now reads
 * straight from wubu_shard_raw(), so the packer + manifest are gone. These
 * symbols are retained as no-ops so any straggler build reference compiles. */
void wubu_ssd_moe_pack_layer(const char *sidecar_dir, int layer,
                             int n_experts, int d_model, int d_ff,
                             const float *gate, const float *up, const float *down){ (void)sidecar_dir;(void)layer;(void)n_experts;(void)d_model;(void)d_ff;(void)gate;(void)up;(void)down; }
void wubu_ssd_moe_write_manifest(const char *sidecar_dir, int n_layers,
                                 int n_experts, int d_model, int d_ff,
                                 int n_active, int slot_bank){ (void)sidecar_dir;(void)n_layers;(void)n_experts;(void)d_model;(void)d_ff;(void)n_active;(void)slot_bank; }
