/*
 * test_harness_file.c — the LIVE-FILE HARNESS gate (AN47 #3: "real
 * user-space tasks so traj cells and skills form from actual file
 * work, not only fixed toy tasks").
 *
 * Asserts:
 *   1. the text-chunk task scores the REAL round-trip (byte-exact
 *      rejoin = integrity 1.0)
 *   2. the codec task runs a REAL PCM->image->PCM round-trip with a
 *      meaningful reconstruction correlation
 *   3. the outcomes are traj cells in the hive (the skills will form
 *      from this actual file work)
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "wubu_harness_file.h"
#include "wubu_hive.h"

#define FAIL(...) do { printf("  FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)

int main(void)
{
    printf("=== test_harness_file (real file work, real scores) ===\n");

    wubu_hive_t tissue;
    if (wubu_hive_init(&tissue) != 0) FAIL("hive init");

    wubu_file_harness_t fh;
    if (wubu_file_harness_init(&fh, &tissue) != 0) FAIL("harness init");

    /* 1. the text-chunk task: a real 10KB buffer, chunked + rejoined */
    const size_t N = 10000;
    char *buf = (char *)malloc(N);
    for (size_t i = 0; i < N; i++) buf[i] = (char)('a' + (i % 26));
    int p1 = wubu_file_task_text(&fh, buf, N, 512, 11, 0.2f, 1);
    printf("  text-chunk round-trip: pass=%d (byte-exact rejoin)\n", p1);
    if (!p1) FAIL("the byte-exact text round-trip did not pass");
    /* a corrupted rejoin fails: chunk size 1 with a bad transform... */
    int p2 = wubu_file_task_text(&fh, buf, N, 0, 12, 0.2f, 1);  /* chunk 0 */
    if (p2) FAIL("the degenerate chunk size passed (scorer broken)");
    printf("  degenerate chunk (0): pass=%d (0 = correctly failed)\n", p2);

    /* 2. the codec task: a real PCM sweep through the Kodak (the
     * codec's internal rate is 16kHz + the STFT needs >=8192 samples
     * for the 128-frame reconstruction) */
    int n_samp = 8192;
    float *pcm = (float *)malloc((size_t)n_samp * sizeof(float));
    for (int i = 0; i < n_samp; i++)
        pcm[i] = 0.5f * sinf(2.0f * 3.14159f * 440.0f * (float)i / 16000.0f) +
                 0.3f * sinf(2.0f * 3.14159f * 880.0f * (float)i / 16000.0f);
    int p3 = wubu_file_task_codec(&fh, pcm, n_samp, 21, 0.3f, 1);
    printf("  codec PCM->image->PCM: pass=%d (reconstruction scored)\n", p3);
    if (!p3) FAIL("the codec round-trip did not clear the correlation floor");

    /* 3. the outcomes are traj cells in the hive (the degenerate
     * chunk task never ran, so 2 real-file cells: 1 text + 1 codec) */
    size_t live = wubu_hive_live(&tissue);
    printf("  hive live cells: %zu (2 real-file traj cells)\n", live);
    if (live < 2) FAIL("the file-task outcomes are not in the hive");

    char stats[256];
    wubu_file_harness_stats(&fh, stats, sizeof(stats));
    printf("  stats: %s\n", stats);

    free(buf); free(pcm);
    printf("=== ALL FILE-HARNESS TESTS PASSED (the colony does real work) ===\n");
    return 0;
}
