/*
 * wubu_skill_train.h — SKILL → TRAIN FEEDBACK (AN47 #5: "accepted
 * skills and high-value traj cells become an explicit fine-tune /
 * preference stream into the next train window. Close experience →
 * weights, not only experience → hive cells."). C11.
 *
 * The bridge: the hive's accepted skills + high-value traj cells
 * become a TRAIN STREAM the trainer consumes in the next window —
 * the preference pairs (win vs lose lineages) become the RLHF-style
 * gradient signal, the skill patterns become the SFT segments.
 *
 * Pure C11, opaque. The stream is a bounded buffer the trainer polls.
 */
#ifndef WUBU_SKILL_TRAIN_H
#define WUBU_SKILL_TRAIN_H

#include <stdint.h>
#include <stddef.h>

/* one train-stream item: a preference pair or an SFT segment */
typedef struct {
    uint32_t   skill_version;  /* the skill that produced it (0 = traj) */
    uint16_t   goal_token;     /* the goal domain */
    float      win_value;      /* the preferred outcome value */
    float      lose_value;     /* the dispreferred outcome value */
    uint8_t    kind;           /* 0 = preference pair, 1 = SFT seg */
    uint8_t    cell_idx;
    uint64_t   batch;
} wubu_train_stream_item_t;

/* the train-stream state */
typedef struct {
    wubu_train_stream_item_t items[64];  /* the bounded stream */
    int  n;
    uint64_t n_drained;
} wubu_train_stream_t;

/* ST1: init. */
int wubu_train_stream_init(wubu_train_stream_t *st);

/* ST2: push a preference pair from a skill's win/lose lineages (the
 * RLHF-style signal the next window trains on). */
int wubu_train_stream_push_pair(wubu_train_stream_t *st,
                                uint32_t skill_version, uint16_t goal,
                                float win, float lose, uint8_t cell_idx,
                                uint64_t batch);

/* ST3: push an SFT segment from an accepted skill's pattern (the
 * fine-tune signal: the skill's traj becomes a train datum). */
int wubu_train_stream_push_sft(wubu_train_stream_t *st,
                               uint32_t skill_version, uint16_t goal,
                               float value, uint8_t cell_idx,
                               uint64_t batch);

/* ST4: DRAIN the stream into the trainer's preference buffer (the
 * next window consumes it). Returns the count drained. */
int wubu_train_stream_drain(wubu_train_stream_t *st,
                            wubu_train_stream_item_t *out, int cap);

/* ST5: the stream stats. */
void wubu_train_stream_stats(const wubu_train_stream_t *st, char *buf, size_t cap);

#endif
