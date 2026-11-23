/*
 * wubu_lora_slots.c -- EG-17 multi-adapter LoRA slots. Pure C11.
 */
#include "wubu_lora_slots.h"
#include "wubu_lora.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    wubu_lora_t *adapter;
    int in_f, out_f;
    int used;
} slot_t;

struct wubu_lora_slots {
    slot_t *slots;
    int capacity, count;
};

wubu_lora_slots_t *wubu_lora_slots_create(int capacity) {
    wubu_lora_slots_t *s;
    if (capacity <= 0) return NULL;
    s = (wubu_lora_slots_t *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->slots = (slot_t *)calloc((size_t)capacity, sizeof(slot_t));
    if (!s->slots) { free(s); return NULL; }
    s->capacity = capacity;
    return s;
}

void wubu_lora_slots_free(wubu_lora_slots_t *s) {
    int i;
    if (!s) return;
    for (i = 0; i < s->capacity; i++) wubu_lora_free(s->slots[i].adapter);
    free(s->slots);
    free(s);
}

int wubu_lora_slots_add(wubu_lora_slots_t *s, int id, int rank, float alpha,
                        int in_f, int out_f, const float *A, const float *B) {
    wubu_lora_t *l;
    if (!s || id < 0 || id >= s->capacity || s->slots[id].used ||
        in_f <= 0 || out_f <= 0 || !A || !B) return -1;
    l = wubu_lora_create(rank, alpha, in_f, out_f);
    if (!l) return -1;
    if (wubu_lora_load_f32(l, A, B) != 0) {
        wubu_lora_free(l);
        return -1;
    }
    s->slots[id].adapter = l;
    s->slots[id].in_f = in_f;
    s->slots[id].out_f = out_f;
    s->slots[id].used = 1;
    s->count++;
    return 0;
}

int wubu_lora_slots_remove(wubu_lora_slots_t *s, int id) {
    if (!s || id < 0 || id >= s->capacity || !s->slots[id].used) return -1;
    wubu_lora_free(s->slots[id].adapter);
    memset(&s->slots[id], 0, sizeof(s->slots[id]));
    s->count--;
    return 0;
}

int wubu_lora_slots_count(const wubu_lora_slots_t *s) {
    return s ? s->count : 0;
}

int wubu_lora_slots_apply_grouped(const wubu_lora_slots_t *s,
                                  const int *ids, int n_ids,
                                  const float *x, float *out, int out_f) {
    int i, j, in_f = -1;
    float *tmp;
    if (!s || !x || !out || n_ids < 0 || (n_ids > 0 && !ids)) return -1;
    for (i = 0; i < out_f; i++) out[i] = 0.0f;   /* per-request isolation */
    if (n_ids == 0) return 0;

    for (i = 0; i < n_ids; i++) {
        const slot_t *sl;
        int id = ids[i];
        if (id < 0 || id >= s->capacity) return -1;
        sl = &s->slots[id];
        if (!sl->used) return -1;
        for (j = 0; j < i; j++)
            if (ids[j] == id) return -1;          /* no double-count */
        if (in_f < 0) in_f = sl->in_f;
        if (sl->in_f != in_f || sl->out_f != out_f) return -1;
    }
    tmp = (float *)malloc((size_t)out_f * sizeof(float));
    if (!tmp) return -1;
    for (i = 0; i < n_ids; i++) {
        const slot_t *sl = &s->slots[ids[i]];
        if (wubu_lora_forward(sl->adapter, x, tmp) != 0) {
            free(tmp);
            return -1;
        }
        for (j = 0; j < out_f; j++) out[j] += tmp[j];
    }
    free(tmp);
    return 0;
}
