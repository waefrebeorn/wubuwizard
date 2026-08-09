/*
 * wubu_model_format.c — Format-agnostic model loading (ADR-002).
 *
 * The engine must not be coupled to a single weight format. This module
 * implements the adapter registry: each format (GGUF, safetensors, ONNX)
 * provides a wubu_model_format_t vtable; wubu_model_open() dispatches
 * to the right one.
 *
 * Research 066-B1/B3 convergence: "Abstract formats behind a vtable so
 * the engine never knows or cares which serializer produced a checkpoint."
 */
#include "wubu_model_format.h"
#include <string.h>
#include <stdlib.h>

/* ---- Registry (fixed-capacity, no malloc in registry path) ---- */
#define MAX_FORMATS 8
static wubu_model_format_t *g_formats[MAX_FORMATS];
static int g_nformats = 0;

int wubu_model_format_register(const wubu_model_format_t *fmt) {
    if (!fmt || !fmt->name || !fmt->open) return -1;
    for (int i = 0; i < g_nformats; i++) {
        if (g_formats[i] == fmt) return -1;  /* already registered */
        if (strcmp(g_formats[i]->name, fmt->name) == 0) return -1;
    }
    if (g_nformats >= MAX_FORMATS) return -1;
    g_formats[g_nformats++] = (wubu_model_format_t *)fmt;
    return 0;
}

const wubu_model_format_t *wubu_model_format_for(const char *path) {
    if (!path) return NULL;
    /* Try each registered adapter's probe(). */
    for (int i = 0; i < g_nformats; i++) {
        if (g_formats[i]->probe && g_formats[i]->probe(path))
            return g_formats[i];
    }
    /* Fallback: match by extension. */
    const char *dot = strrchr(path, '.');
    if (!dot) return NULL;
    for (int i = 0; i < g_nformats; i++) {
        if (g_formats[i]->extension &&
            strcasecmp(dot, g_formats[i]->extension) == 0)
            return g_formats[i];
    }
    return NULL;
}

wubu_format_ctx_t *wubu_model_open(const char *path) {
    const wubu_model_format_t *fmt = wubu_model_format_for(path);
    if (!fmt) {
        fprintf(stderr, "wubu_model_open: no adapter for %s\n", path);
        return NULL;
    }
    return fmt->open(path);
}

void wubu_model_close(wubu_format_ctx_t *ctx) {
    if (!ctx) return;
    /* Each adapter stores its vtable pointer at ctx[0] (convention). */
    wubu_model_format_t *fmt = *(wubu_model_format_t **)ctx;
    if (fmt && fmt->close)
        fmt->close(ctx);
}

/* ---- Built-in adapters ---- */
/* Forward declarations — each adapter lives in its own .c file. */
extern wubu_model_format_t wubu_format_gguf;
extern wubu_model_format_t wubu_format_safetensors;
/* ONNX adapter: real protobuf initializer parser (wubu_model_format_onnx.c) */
extern wubu_model_format_t wubu_format_onnx;

void wubu_model_format_register_gguf(void)    { wubu_model_format_register(&wubu_format_gguf); }
void wubu_model_format_register_safetensors(void) { wubu_model_format_register(&wubu_format_safetensors); }
void wubu_model_format_register_onnx(void)      { wubu_model_format_register(&wubu_format_onnx); }

void wubu_model_format_register_all(void) {
    wubu_model_format_register_gguf();
    wubu_model_format_register_safetensors();
    wubu_model_format_register_onnx();
}
