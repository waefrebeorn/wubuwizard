/* wubu_canvas.h -- THE VHF CANVAS (research/066): the latent space as a
 * coordinate-addressable field.
 *
 * The user's prior art (AUDIO/wubusynth/vhf_tool.py): the latent space
 * is shaped like the VGA signal — a canvas where every modality has
 * its own coordinates, encoded as per-cell quaternion + amplitude, and
 * decoded by sampling the field at (y,x) with positional encoding.
 *
 * This module makes that real in C11, connected to the stack we built:
 *   - the encoder slot (wubu_encoder) WRITES field cells
 *   - the canvas (this) IS the field — addressable by coordinate
 *   - the router (wubu_router) READS the field (top-K sampling)
 *   - the KV namespace (wubu_kvfs) mounts the canvas: /kv/canvas/(y,x)
 *
 * Cell = unit quaternion (orientation, 4) + amplitude (presence, 1).
 * Decode = positional encoding (sin/cos at 2^k·pi freqs) + bilinear
 * field sample -> the shared embedding space.
 *
 * C11, opaque, self-contained (sin/cos from <math.h>).
 */
#ifndef WUBU_CANVAS_H
#define WUBU_CANVAS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_canvas wubu_canvas_t;

/* A field cell: orientation (unit quaternion) + amplitude. */
typedef struct {
    float q[4];       /* unit quaternion (the orientation field) */
    float amp;        /* amplitude (presence/energy at the cell) */
} wubu_canvas_cell_t;

/* Create a canvas: grid_y × grid_x cells. Returns NULL on failure. */
wubu_canvas_t *wubu_canvas_create(int grid_y, int grid_x);

/* WRITE a cell (the encoder slot's output lands here). */
int wubu_canvas_set(wubu_canvas_t *c, int y, int x,
                    const wubu_canvas_cell_t *cell);

/* READ a cell. */
int wubu_canvas_get(const wubu_canvas_t *c, int y, int x,
                    wubu_canvas_cell_t *out);

/* DECODE (the VHFDecoder, ours): sample the field at a normalized
 * coordinate (cy, cx in [-1, 1]) with positional encoding -> out
 * [out_dim floats]. The decode function is a first-class module —
 * any coordinate can be queried (implicit neural field). */
int wubu_canvas_decode(const wubu_canvas_t *c, float cy, float cx,
                       float *out, int out_dim);

/* The canvas geometry. */
int wubu_canvas_grid_y(const wubu_canvas_t *c);
int wubu_canvas_grid_x(const wubu_canvas_t *c);

/* Free the canvas. */
void wubu_canvas_free(wubu_canvas_t *c);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_CANVAS_H */
