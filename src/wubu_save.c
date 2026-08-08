/*
 * wubu_save.c -- save WuBu checkpoints as REAL safetensors.
 *
 * The DA pass found: we could read safetensors but never write them --
 * every trained checkpoint was a private .st dump no standard tooling
 * could open. This module exports the trained seed in the exact
 * released layout (137 tensors, same names), so HF tooling, the bigger
 * brother (Qwen), and any future framework can load our fine-tunes.
 */
#include "wubu.h"
#include "safetensors_writer.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* strdup is POSIX, not C11 -- implicitly declared under -std=c11 it
 * returns int garbage (the same bug class as the tokenizer crash).
 * Local implementation, freestanding-safe. */
static char *local_strdup(const char *s)
{
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *out = (char *)malloc(n);
    if (!out) return NULL;
    memcpy(out, s, n);
    return out;
}

int wubu_save_safetensors(const wubu_model_t *m, const char *path)
{
    if (!m || !path) return -1;
    /* 1 embedding + 1 final_norm + 12 blocks * 11 + 3 selectors = 137 */
    enum { MAX_T = 137 };
    st_writer_tensor_t t[MAX_T];
    int n = 0;
    t[n].name = "embedding.weight";
    t[n].data = m->embedding;
    t[n].n_elems = 16384 * 448;
    t[n].dims[0] = 16384; t[n].dims[1] = 448; t[n].n_dims = 2;
    n++;
    t[n].name = "final_norm.weight";
    t[n].data = m->final_norm;
    t[n].n_elems = 448;
    t[n].dims[0] = 448; t[n].n_dims = 1;
    n++;
    char name[128];
    for (int i = 0; i < m->n_layers && n < MAX_T; i++) {
        const wubu_block_t *b = &m->blocks[i];
        struct { const char *suffix; const float *data; int64_t r, c; } w[11] = {
            { "attn.q_proj.weight", b->q_proj, 448, 448 },
            { "attn.k_proj.weight", b->k_proj, 64, 448 },
            { "attn.v_proj.weight", b->v_proj, 64, 448 },
            { "attn.o_proj.weight", b->o_proj, 448, 448 },
            { "attn.g_proj.weight", b->g_proj, 448, 448 },
            { "attn.q_norm.weight", b->q_norm, 64, 1 },
            { "attn.k_norm.weight", b->k_norm, 64, 1 },
            { "attn_norm.weight", b->attn_norm, 448, 1 },
            { "ffn.gate_up.weight", b->gate_up, 2456, 448 },
            { "ffn.down.weight", b->down, 448, 1228 },
            { "ffn_norm.weight", b->ffn_norm, 448, 1 },
        };
        for (int j = 0; j < 11 && n < MAX_T; j++) {
            snprintf(name, sizeof(name), "layers.%d.%s", i, w[j].suffix);
            t[n].name = local_strdup(name);
            if (!t[n].name) return -1;
            t[n].data = w[j].data;
            t[n].n_elems = w[j].r * (w[j].c > 1 ? w[j].c : 1);
            if (w[j].c > 1) { t[n].dims[0] = w[j].r; t[n].dims[1] = w[j].c; t[n].n_dims = 2; }
            else { t[n].dims[0] = w[j].r; t[n].n_dims = 1; }
            n++;
        }
    }
    for (int i = 0; i < WUBU_SELECTORS && n < MAX_T; i++) {
        snprintf(name, sizeof(name), "selectors.%d.score.weight", i);
        t[n].name = local_strdup(name);
        if (!t[n].name) return -1;
        t[n].data = m->selectors[i];
        t[n].n_elems = 448;
        t[n].dims[0] = 448; t[n].n_dims = 1;
        n++;
    }
    int rc = st_write_f32(path, t, n);
    for (int i = 0; i < n; i++) {
        if (strcmp(t[i].name, "embedding.weight") != 0 &&
            strcmp(t[i].name, "final_norm.weight") != 0)
            free((void *)t[i].name);
    }
    return rc;
}
