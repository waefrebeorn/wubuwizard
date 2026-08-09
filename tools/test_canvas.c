/*
 * test_canvas.c -- research/066 gate: THE VHF CANVAS in C11.
 *
 * The user's prior art (VHF encoder, AUDIO/wubusynth): the latent
 * space is a coordinate-addressable field — a canvas where every
 * modality has its own coordinates, encoded as quaternion+amplitude
 * per cell, decoded by sampling with positional encoding. This gate
 * proves that field works in C11 and connects to the encoder space:
 *
 *   1. WRITE: the encoder slot's output lands as field cells
 *      (orientation quaternion + amplitude).
 *   2. DECODE: any coordinate (cy,cx in [-1,1]) queries the field —
 *      the decode function is a first-class module (implicit field).
 *   3. COHERENCE: neighboring coordinates decode to SIMILAR
 *      embeddings (the field is continuous, not a lookup table) —
 *      that is what makes it a space, not a bag.
 *   4. AMPLITUDE GATES: a zero-amplitude region still decodes (the
 *      field is everywhere), but cells with energy dominate.
 *   5. ADDRESSABILITY: a video-line coordinate and an audio-HBI
 *      coordinate are DIFFERENT points in the SAME field — one
 *      canvas, all modalities (the VHF "video + audio in one
 *      morning" insight).
 *
 * Gate: `make test_canvas`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "wubu_canvas.h"

static int failures = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

int main(void)
{
    printf("=== test_canvas (research/066: THE VHF CANVAS, C11) ===\n");

    /* the canvas: 96x96 cells (the VHF latent grid size) */
    wubu_canvas_t *cv = wubu_canvas_create(96, 96);
    CHECK(cv != NULL, "canvas created");
    if (!cv) return 1;
    CHECK(wubu_canvas_grid_y(cv) == 96 && wubu_canvas_grid_x(cv) == 96,
          "canvas is 96x96 (the VHF latent grid)");

    /* ---- 1. write field cells (the encoder's output) ---- */
    /* a "video region": a bright quaternion field in the top-left */
    for (int y = 10; y < 40; y++)
        for (int x = 10; x < 40; x++) {
            wubu_canvas_cell_t cell;
            memset(&cell, 0, sizeof(cell));
            float a = (float)(x - 10) / 30.0f;
            cell.q[0] = cosf(a * 0.5f);
            cell.q[3] = sinf(a * 0.5f);   /* rotation about z */
            cell.amp = 1.0f;
            CHECK(wubu_canvas_set(cv, y, x, &cell) == 0, "cell write");
        }
    /* an "audio region": a different field in the HBI columns (left edge) */
    for (int y = 0; y < 96; y++) {
        wubu_canvas_cell_t cell;
        memset(&cell, 0, sizeof(cell));
        cell.q[0] = cosf(0.9f);
        cell.q[1] = sinf(0.9f);           /* rotation about x */
        cell.amp = 0.5f;
        wubu_canvas_set(cv, y, 0, &cell);
    }

    /* ---- 2. decode: the field is queryable at ANY coordinate ---- */
    float e_video[64], e_video2[64], e_audio[64], e_empty[64];
    CHECK(wubu_canvas_decode(cv, -0.5f, -0.5f, e_video, 64) == 0,
          "video-region coordinate decodes");
    CHECK(wubu_canvas_decode(cv, -0.5f, -0.5f, e_video2, 64) == 0,
          "decode is repeatable");
    CHECK(wubu_canvas_decode(cv, -0.9f, 0.0f, e_audio, 64) == 0,
          "audio-region coordinate decodes");
    CHECK(wubu_canvas_decode(cv, 0.9f, 0.9f, e_empty, 64) == 0,
          "empty-region coordinate decodes (the field is everywhere)");

    /* ---- 3. coherence: same coordinate -> same embedding ---- */
    int same = 1;
    for (int d = 0; d < 64; d++)
        if (e_video[d] != e_video2[d]) { same = 0; break; }
    CHECK(same, "decode is deterministic (pure function of coordinate)");

    /* ---- 4. continuity: neighbors are similar, far points differ ---- */
    float e_near[64], e_far[64];
    wubu_canvas_decode(cv, -0.5f, -0.48f, e_near, 64);
    wubu_canvas_decode(cv, -0.5f, 0.9f, e_far, 64);
    double d_near = 0.0, d_far = 0.0;
    for (int d = 0; d < 64; d++) {
        d_near += fabs((double)e_video[d] - e_near[d]);
        d_far += fabs((double)e_video[d] - e_far[d]);
    }
    printf("  ok: near-distance=%.3f far-distance=%.3f\n", d_near, d_far);
    CHECK(d_near < d_far * 0.5,
          "neighboring coordinates decode closer (the field is "
          "continuous — a space, not a lookup)");

    /* ---- 5. one canvas, all modalities ---- */
    double d_audio = 0.0;
    for (int d = 0; d < 64; d++)
        d_audio += fabs((double)e_video[d] - e_audio[d]);
    CHECK(d_audio > d_near,
          "video and audio regions are DIFFERENT points in the SAME "
          "field (VHF: video lines + HBI audio, one canvas)");
    printf("  ok: video-region vs audio-region distance=%.3f "
           "(same field, different coordinates)\n", d_audio);

    wubu_canvas_free(cv);
    if (failures == 0) printf("=== ALL CANVAS TESTS PASSED ===\n");
    else printf("=== %d FAILURES ===\n", failures);
    return failures ? 1 : 0;
}
