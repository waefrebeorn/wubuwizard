/*
 * wubu_skill_train.c — SKILL → TRAIN FEEDBACK (see the header).
 * Experience → weights, not only experience → hive cells.
 */
#include "wubu_skill_train.h"

#include <stdio.h>
#include <string.h>

int wubu_train_stream_init(wubu_train_stream_t *st)
{
    if (!st) return -1;
    memset(st, 0, sizeof(*st));
    return 0;
}

int wubu_train_stream_push_pair(wubu_train_stream_t *st,
                                uint32_t skill_version, uint16_t goal,
                                float win, float lose, uint8_t cell_idx,
                                uint64_t batch)
{
    if (!st || st->n >= 64) return -1;
    wubu_train_stream_item_t *it = &st->items[st->n++];
    memset(it, 0, sizeof(*it));
    it->skill_version = skill_version;
    it->goal_token = goal;
    it->win_value = win;
    it->lose_value = lose;
    it->kind = 0;   /* preference pair */
    it->cell_idx = cell_idx;
    it->batch = batch;
    return 0;
}

int wubu_train_stream_push_sft(wubu_train_stream_t *st,
                               uint32_t skill_version, uint16_t goal,
                               float value, uint8_t cell_idx,
                               uint64_t batch)
{
    if (!st || st->n >= 64) return -1;
    wubu_train_stream_item_t *it = &st->items[st->n++];
    memset(it, 0, sizeof(*it));
    it->skill_version = skill_version;
    it->goal_token = goal;
    it->win_value = value;
    it->kind = 1;   /* SFT segment */
    it->cell_idx = cell_idx;
    it->batch = batch;
    return 0;
}

int wubu_train_stream_drain(wubu_train_stream_t *st,
                            wubu_train_stream_item_t *out, int cap)
{
    if (!st || !out || cap <= 0) return 0;
    int k = 0;
    for (int i = 0; i < st->n && k < cap; i++) {
        out[k++] = st->items[i];
    }
    st->n_drained += (uint64_t)k;
    st->n = 0;   /* drained */
    return k;
}

void wubu_train_stream_stats(const wubu_train_stream_t *st, char *buf, size_t cap)
{
    if (!st || !buf || cap == 0) return;
    int pairs = 0, sfts = 0;
    for (int i = 0; i < st->n; i++)
        if (st->items[i].kind == 0) pairs++; else sfts++;
    snprintf(buf, cap,
             "pending=%d pairs=%d sft=%d drained=%llu",
             st->n, pairs, sfts,
             (unsigned long long)st->n_drained);
}
