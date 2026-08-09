/*
 * wubu_harness_file.h — LIVE-FILE HARNESS TASKS (AN47 #3: "replace
 * or extend synthetic scorers with real user-space tasks — chunked
 * text, WAV→canvas, codec round-trip — so traj cells and skills form
 * from actual file work, not only fixed toy tasks"). C11.
 *
 * The tasks score on REAL file outcomes:
 *   - the TEXT-CHUNK task: ingest a real file via userfs, split it
 *     into chunks, rejoin — the score is the round-trip integrity
 *   - the CODEC task: PCM -> Kodak image -> PCM (the universal codec
 *     round-trip) — the score is the reconstruction correlation
 *   - the CANVAS task: the codec's image written into the canvas —
 *     the score is the canvas occupancy/integrity
 *
 * The outcomes are TRAJ CELLS (the same as the synthetic harness) —
 * but now the colony's skills form from ACTUAL file work.
 */
#ifndef WUBU_HARNESS_FILE_H
#define WUBU_HARNESS_FILE_H

#include <stdint.h>
#include <stddef.h>

#include "wubu_hive.h"

/* one live-file task result */
typedef struct {
    uint16_t goal_token;     /* the goal this task served */
    uint8_t  kind;           /* 0=text-chunk 1=codec 2=canvas */
    float    integrity;      /* 0..1 the round-trip integrity */
    float    cost;           /* normalized cost */
    int      ok;             /* 1 = passed the floor */
    uint64_t batch;
} wubu_file_task_result_t;

/* the live-file harness state */
typedef struct {
    wubu_hive_t *tissue;      /* the traj cells land here */
    uint64_t n_passed, n_failed;
    float    score;           /* the running live-file score */
} wubu_file_harness_t;

/* F1: init. */
int wubu_file_harness_init(wubu_file_harness_t *fh, wubu_hive_t *tissue);

/* F2: the TEXT-CHUNK task — ingest a real text file (via the caller's
 * buffer), split into chunks + rejoin, score the round-trip. The
 * caller provides the file bytes + the chunk size. Returns 1 = pass. */
int wubu_file_task_text(wubu_file_harness_t *fh, const char *data,
                        size_t n, size_t chunk, uint16_t goal,
                        float cost, uint64_t batch);

/* F3: the CODEC task — PCM -> image -> PCM round-trip. The caller
 * provides the PCM (the harness runs the codec both ways + scores
 * the reconstruction). Returns 1 = pass. */
int wubu_file_task_codec(wubu_file_harness_t *fh, const float *pcm,
                         int n_samples, uint16_t goal, float cost,
                         uint64_t batch);

/* F4: the live-file score (the first-class signal the metadiag
 * reads — actual file work, not toy tasks). */
float wubu_file_harness_score(const wubu_file_harness_t *fh);

/* F5: the stats. */
void wubu_file_harness_stats(const wubu_file_harness_t *fh, char *buf, size_t cap);

#endif
