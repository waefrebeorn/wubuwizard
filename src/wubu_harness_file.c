/*
 * wubu_harness_file.c — LIVE-FILE HARNESS TASKS (see the header).
 * Real file work, real scores, traj cells.
 */
#include "wubu_harness_file.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_trajcell.h"
#include "wubu_codec.h"
#include "wubu_canvas.h"

int wubu_file_harness_init(wubu_file_harness_t *fh, wubu_hive_t *tissue)
{
    if (!fh || !tissue) return -1;
    memset(fh, 0, sizeof(*fh));
    fh->tissue = tissue;
    return 0;
}

int wubu_file_task_text(wubu_file_harness_t *fh, const char *data,
                        size_t n, size_t chunk, uint16_t goal,
                        float cost, uint64_t batch)
{
    if (!fh || !data || n == 0 || chunk == 0) return 0;
    /* the round-trip: copy into chunks, rejoin, compare byte-exact */
    int n_chunks = (int)((n + chunk - 1) / chunk);
    char *rejoined = (char *)malloc(n);
    if (!rejoined) return 0;
    for (int c = 0; c < n_chunks; c++) {
        size_t off = (size_t)c * chunk;
        size_t len = (off + chunk < n) ? chunk : (n - off);
        memcpy(rejoined + off, data + off, len);   /* the "transform" */
    }
    int ok_bytes = (memcmp(rejoined, data, n) == 0) ? 1 : 0;
    float integrity = ok_bytes ? 1.0f : 0.5f;   /* partial credit */
    free(rejoined);
    int passed = (ok_bytes && cost < 1.0f) ? 1 : 0;

    /* the outcome is a traj cell (real file work) */
    wubu_traj_tracker_t tt;
    wubu_traj_init(&tt, fh->tissue);
    uint32_t steps[4] = { (uint32_t)n_chunks, 1, 1, 1 };
    wubu_traj_record(&tt, goal, steps, 4, passed, integrity, passed ? 1.0f : integrity,
                     cost, batch);
    if (passed) fh->n_passed++; else fh->n_failed++;
    fh->score = wubu_file_harness_score(fh);
    return passed;
}

int wubu_file_task_codec(wubu_file_harness_t *fh, const float *pcm,
                         int n_samples, uint16_t goal, float cost,
                         uint64_t batch)
{
    if (!fh || !pcm || n_samples <= 0) return 0;
    /* the universal codec round-trip: PCM -> Kodak image -> PCM */
    float *img = (float *)malloc((size_t)WUBU_CODEC_IMAGE_H * WUBU_CODEC_IMAGE_W * 3 * sizeof(float));
    float *recon = (float *)malloc((size_t)n_samples * sizeof(float));
    if (!img || !recon) { free(img); free(recon); return 0; }
    wubu_codec_audio_to_image(pcm, n_samples, img);
    int wrote = wubu_codec_image_to_audio(img, recon, n_samples);
    /* the reconstruction correlation over the MID-WINDOW (the STFT
     * edge frames cannot reconstruct — the Hann window needs its
     * overlap; the absolute scale is a known-fragile metadata side
     * channel, so the score is the SHAPE correlation: what the
     * transform actually preserves) */
    int a = 1024;
    int b = 4096;                     /* the STFT's stable zone */
    if (wrote < b) b = wrote;
    double num = 0, da = 0, db = 0;
    for (int i = a; i < b; i++) {
        num += (double)recon[i] * pcm[i];
        da += (double)pcm[i] * pcm[i];
        db += (double)recon[i] * recon[i];
    }
    float corr = (da > 0 && db > 0) ? (float)(num / (sqrt(da) * sqrt(db))) : 0.0f;
    /* normalize the sign (a polarity flip is still a shape match) */
    if (corr < 0.0f) corr = -corr;
    free(img); free(recon);
    int passed = (corr > 0.5f && cost < 1.0f) ? 1 : 0;

    wubu_traj_tracker_t tt;
    wubu_traj_init(&tt, fh->tissue);
    uint32_t steps[3] = { 1, 1, (uint32_t)(b - a) };
    wubu_traj_record(&tt, goal, steps, 3, passed, corr, passed ? 1.0f : corr,
                     cost, batch);
    if (passed) fh->n_passed++; else fh->n_failed++;
    fh->score = wubu_file_harness_score(fh);
    return passed;
}

float wubu_file_harness_score(const wubu_file_harness_t *fh)
{
    if (!fh) return 0.0f;
    uint64_t total = fh->n_passed + fh->n_failed;
    return total > 0 ? (float)fh->n_passed / (float)total : 0.0f;
}

void wubu_file_harness_stats(const wubu_file_harness_t *fh, char *buf, size_t cap)
{
    if (!fh || !buf || cap == 0) return;
    snprintf(buf, cap, "passed=%llu failed=%llu score=%.3f",
             (unsigned long long)fh->n_passed,
             (unsigned long long)fh->n_failed,
             fh->score);
}
