/* wubu_model_arch.h — architecture registry for GGUF models.
 *
 * A model's `general.architecture` metadata is the primary dispatch key;
 * tensor naming and shapes are only fallbacks. Unknown architectures still
 * get a generic transformer/SSM/MoE description instead of being forced
 * through one hard-coded Gemma/Qwen path.
 */
#ifndef WUBU_MODEL_ARCH_H
#define WUBU_MODEL_ARCH_H

#include <stdbool.h>
#include <stdio.h>
#include "gguf_reader.h"

typedef enum {
    WUBU_ARCH_UNKNOWN = 0,
    WUBU_ARCH_LLAMA,
    WUBU_ARCH_GEMMA,
    WUBU_ARCH_GEMMA4,
    WUBU_ARCH_QWEN,
    WUBU_ARCH_QWEN_MOE,
    WUBU_ARCH_DEEPSEEK,
    WUBU_ARCH_MISTRAL,
    WUBU_ARCH_PHI,
    WUBU_ARCH_LFM2,
    WUBU_ARCH_COUNT
} wubu_arch_id_t;

typedef enum {
    WUBU_LAYOUT_BLK = 0,       /* blk.N.*            */
    WUBU_LAYOUT_MODEL_LAYERS,  /* model.layers.N.*   */
    WUBU_LAYOUT_AUTO
} wubu_tensor_layout_t;

typedef enum {
    WUBU_FAMILY_TRANSFORMER = 0,
    WUBU_FAMILY_HYBRID_SSM,
    WUBU_FAMILY_MOE
} wubu_family_t;

/* A registered architecture handler: metadata name -> kernel family. */
typedef struct {
    const char *gguf_arch;
    wubu_arch_id_t id;
    wubu_family_t family;
    wubu_tensor_layout_t layout;
    bool rope_neox;
    bool rmsnorm;
    bool sliding_window;
} wubu_arch_entry_t;

typedef enum {
    WUBU_RUNNER_UNKNOWN = 0,
    WUBU_RUNNER_GENERIC_TRANSFORMER,
    WUBU_RUNNER_GEMMA4,
    WUBU_RUNNER_MOE,
    WUBU_RUNNER_SSM
} wubu_runner_t;

/* Concrete geometry extracted from one model file. */
typedef struct {
    wubu_arch_id_t id;
    char arch_name[64];
    wubu_family_t family;
    wubu_tensor_layout_t layout;
    wubu_runner_t runner;
    int n_layers;
    int d_model;
    int d_ff;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int rotary_dim;
    int vocab_size;
    int sliding_window;
    float rope_base;
    bool metadata_complete; /* all core dims came from GGUF metadata */
    bool supported;         /* a registered architecture handler exists */
} wubu_arch_spec_t;

const wubu_arch_entry_t *wubu_arch_registry(void);
int wubu_arch_registry_count(void);
const wubu_arch_entry_t *wubu_arch_lookup(const char *gguf_arch);
const char *wubu_arch_id_name(wubu_arch_id_t id);
const char *wubu_family_name(wubu_family_t family);
const char *wubu_layout_name(wubu_tensor_layout_t layout);
const char *wubu_runner_name(wubu_runner_t runner);

/* Probe a GGUF file. Returns 1 if a usable spec was produced (even for an
 * unknown architecture), 0 if the file cannot be read at all. */
int wubu_arch_probe(gguf_ctx *ctx, wubu_arch_spec_t *out);
void wubu_arch_print(const wubu_arch_spec_t *spec, FILE *fp);

#endif /* WUBU_MODEL_ARCH_H */
