/* wubu_canvas.c -- THE VHF CANVAS (research/066): the latent space as
 * a coordinate-addressable field, in C11.
 *
 * The user's prior art (VHF tool, AUDIO/wubusynth): a canvas where
 * video lives in the visible lines and audio in the HBI columns —
 * every modality at its own coordinates, encoded as quaternion +
 * amplitude per cell, decoded by sampling with positional encoding.
 * This is that field, made C11 and connected to the encoder space.
 *
 * C11.
 */
#include "wubu_canvas.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

struct wubu_canvas {
    int gy, gx;
    wubu_canvas_cell_t *cells;   /* gy*gx, row-major */
};

wubu_canvas_t *wubu_canvas_create(int grid_y, int grid_x) {
    if (grid_y < 1 || grid_x < 1 || grid_y > 4096 || grid_x > 4096)
        return NULL;
    wubu_canvas_t *c = (wubu_canvas_t *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->cells = (wubu_canvas_cell_t *)calloc((size_t)grid_y * grid_x,
                                            sizeof(*c->cells));
    if (!c->cells) { free(c); return NULL; }
    c->gy = grid_y;
    c->gx = grid_x;
    /* identity quaternion everywhere (no rotation yet) */
    for (int i = 0; i < grid_y * grid_x; i++) c->cells[i].q[0] = 1.0f;
    return c;
}

int wubu_canvas_set(wubu_canvas_t *c, int y, int x,
                    const wubu_canvas_cell_t *cell) {
    if (!c || !cell || y < 0 || y >= c->gy || x < 0 || x >= c->gx) return -1;
    c->cells[(size_t)y * c->gx + x] = *cell;
    return 0;
}

int wubu_canvas_get(const wubu_canvas_t *c, int y, int x,
                    wubu_canvas_cell_t *out) {
    if (!c || !out || y < 0 || y >= c->gy || x < 0 || x >= c->gx) return -1;
    *out = c->cells[(size_t)y * c->gx + x];
    return 0;
}

int wubu_canvas_grid_y(const wubu_canvas_t *c) { return c ? c->gy : 0; }
int wubu_canvas_grid_x(const wubu_canvas_t *c) { return c ? c->gx : 0; }

/* ---- the decode function (the VHFDecoder, ours) ----
 * Positional encoding: sin/cos at 2^k·pi for k in [0, num_freqs).
 * Sample the quaternion+amplitude field bilinearly at the coordinate,
 * concatenate, and project through a small learned-ish MLP (fixed
 * weights here; the amoeba trains the projection later). The output is
 * the shared embedding space — any coordinate is queryable. */

#define PE_FREQS 6
#define PE_DIM   (2 + PE_FREQS * 2)      /* [cy, cx] + sin/cos pairs */

static void pe_encode(float cy, float cx, float *out) {
    out[0] = cy;
    out[1] = cx;
    for (int k = 0; k < PE_FREQS; k++) {
        float f = (float)(1 << k) * 3.14159265358979f;
        out[2 + k * 2] = sinf(cy * f);
        out[3 + k * 2] = cosf(cx * f);
    }
}

int wubu_canvas_decode(const wubu_canvas_t *c, float cy, float cx,
                       float *out, int out_dim) {
    if (!c || !out || out_dim < 8) return -1;
    if (cy < -1.0f) cy = -1.0f; if (cy > 1.0f) cy = 1.0f;
    if (cx < -1.0f) cx = -1.0f; if (cx > 1.0f) cx = 1.0f;

    /* map [-1,1] -> grid coords */
    float fy = (cy + 1.0f) * 0.5f * (float)(c->gy - 1);
    float fx = (cx + 1.0f) * 0.5f * (float)(c->gx - 1);
    int y0 = (int)fy, x0 = (int)fx;
    if (y0 < 0) y0 = 0; if (y0 > c->gy - 1) y0 = c->gy - 1;
    if (x0 < 0) x0 = 0; if (x0 > c->gx - 1) x0 = c->gx - 1;
    int y1 = y0 + 1 < c->gy ? y0 + 1 : y0;
    int x1 = x0 + 1 < c->gx ? x0 + 1 : x0;
    float ty = fy - (float)y0, tx = fx - (float)x0;

    /* bilinear sample of the 5-field (quaternion + amplitude) */
    wubu_canvas_cell_t c00, c01, c10, c11;
    memset(&c00, 0, sizeof(c00)); memset(&c01, 0, sizeof(c01));
    memset(&c10, 0, sizeof(c10)); memset(&c11, 0, sizeof(c11));
    wubu_canvas_get(c, y0, x0, &c00);
    wubu_canvas_get(c, y0, x1, &c01);
    wubu_canvas_get(c, y1, x0, &c10);
    wubu_canvas_get(c, y1, x1, &c11);

    float field[5] = {0};
    for (int d = 0; d < 5; d++) {
        float v00 = d < 4 ? c00.q[d] : c00.amp;
        float v01 = d < 4 ? c01.q[d] : c01.amp;
        float v10 = d < 4 ? c10.q[d] : c10.amp;
        float v11 = d < 4 ? c11.q[d] : c11.amp;
        float top = v00 * (1 - tx) + v01 * tx;
        float bot = v10 * (1 - tx) + v11 * tx;
        field[d] = top * (1 - ty) + bot * ty;
    }
    /* normalize the interpolated quaternion (orientation must stay a
     * rotation; the amplitude is separate) */
    float qn = sqrtf(field[0]*field[0] + field[1]*field[1] +
                     field[2]*field[2] + field[3]*field[3]) + 1e-6f;
    for (int d = 0; d < 4; d++) field[d] /= qn;

    /* the decode head: PE + field -> out_dim (fixed projection —
     * the amoeba trains this later). */
    float pe[PE_DIM];
    pe_encode(cy, cx, pe);
    for (int d = 0; d < out_dim; d++) {
        float acc = 0.0f;
        /* PE projection (deterministic, seeded per output dim) */
        unsigned s = (unsigned)(d * 2654435761u + 1u);
        for (int k = 0; k < PE_DIM; k++) {
            s = s * 1103515245u + 12345u;
            float w = ((float)((s >> 16) & 0x7FFF) / 16384.0f) - 1.0f;
            acc += pe[k] * w * 0.2f;
        }
        /* field projection */
        s = (unsigned)(d * 40503u + 7u);
        for (int k = 0; k < 5; k++) {
            s = s * 1103515245u + 12345u;
            float w = ((float)((s >> 16) & 0x7FFF) / 16384.0f) - 1.0f;
            acc += field[k] * w;
        }
        out[d] = tanhf(acc);   /* bounded, in [-1, 1] */
    }
    return 0;
}

void wubu_canvas_free(wubu_canvas_t *c) {
    if (!c) return;
    free(c->cells);
    free(c);
}
