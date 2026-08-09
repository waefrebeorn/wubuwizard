/*
 * test_skill_train.c — the SKILL → TRAIN FEEDBACK gate (AN47 #5:
 * "close experience → weights, not only experience → hive cells").
 *
 * Asserts:
 *   1. accepted skills push PREFERENCE PAIRS into the train stream
 *   2. high-value traj cells push SFT segments
 *   3. the trainer DRAINS the stream (the next window consumes it)
 */
#include <stdio.h>
#include <string.h>

#include "wubu_skill_train.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_skill_train (experience -> weights) ===\n");

    wubu_train_stream_t st;
    if (wubu_train_stream_init(&st) != 0) FAIL("stream init");

    /* 1. accepted skills -> preference pairs (win vs lose lineages) */
    wubu_train_stream_push_pair(&st, 1, 42, 0.9f, 0.6f, 0, 1);
    wubu_train_stream_push_pair(&st, 2, 43, 0.8f, 0.5f, 1, 2);
    if (st.n != 2) FAIL("pairs not pushed (%d)", st.n);
    printf("  2 preference pairs from the accepted skills\n");

    /* 2. high-value traj cells -> SFT segments */
    wubu_train_stream_push_sft(&st, 3, 21, 0.95f, 2, 3);
    wubu_train_stream_push_sft(&st, 3, 22, 0.85f, 2, 4);
    if (st.n != 4) FAIL("SFT segments not pushed (%d)", st.n);
    printf("  2 SFT segments from the high-value traj cells\n");

    /* 3. the trainer drains the stream (the next window consumes it) */
    wubu_train_stream_item_t out[8];
    int k = wubu_train_stream_drain(&st, out, 8);
    printf("  drained %d items (next window's preference/SFT stream)\n", k);
    if (k != 4) FAIL("drain wrong (%d)", k);
    if (out[0].kind != 0 || out[2].kind != 1) FAIL("item kinds wrong");
    if (st.n != 0) FAIL("stream not empty after drain");
    if (st.n_drained != 4) FAIL("drain counter wrong");

    char stats[256];
    wubu_train_stream_stats(&st, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    printf("=== ALL SKILL-TRAIN TESTS PASSED (experience becomes weights) ===\n");
    return 0;
}
