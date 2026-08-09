/*
 * wubu_model_format_onnx.c — ONNX format adapter (real, 2026-08-09).
 *
 * Parses the ONNX protobuf (ModelProto -> GraphProto -> initializers) so
 * every weight tensor in an .onnx model file LOADS inside the wizard. The
 * initializers ARE the weights; the graph nodes (the compute ops) are the
 * inference port's concern (Moonshine ASR, MobileSAM, piper — queued).
 *
 * Wire format notes (protobuf):
 *   tag = (field << 3) | wire_type;  wire types: 0=varint, 1=64-bit,
 *   2=len-delimited, 5=32-bit.
 *   ModelProto.graph = field 7 (len-delimited).
 *   GraphProto.initializer = field 5 (len-delimited, repeated TensorProto).
 *   TensorProto: dims=1 (varint, repeated), data_type=2 (varint), name=8
 *   (string), float_data=4 (packed f32), int64_data=7 (packed i64),
 *   raw_data=9 (bytes).
 *
 * License: WaefreBeorn-UMV3
 */
#include "wubu_model_format.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct {
    char *name;
    int64_t *dims;
    int n_dims;
    int data_type;      /* ONNX TensorProto.DataType code */
    const uint8_t *data;/* points into the file buffer (raw_data or typed) */
    size_t data_len;    /* bytes */
} onnx_tensor_t;

typedef struct {
    /* wubu_model_close reads the vtable at ctx[0] (registry convention) —
     * this MUST stay the first member. */
    wubu_model_format_t *fmt;
    uint8_t *buf;
    size_t size;
    onnx_tensor_t *tensors;
    int n_tensors;
} onnx_ctx_t;

/* defined at the bottom of this file (forward decl for onnx_open) */
wubu_model_format_t wubu_format_onnx;

/* ---- protobuf primitives ---- */
static int64_t pb_varint(const uint8_t **pp, const uint8_t *end) {
    const uint8_t *p = *pp;
    uint64_t v = 0;
    int shift = 0;
    while (p < end && shift < 64) {
        uint8_t b = *p++;
        v |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) { *pp = p; return (int64_t)v; }
        shift += 7;
    }
    *pp = end;
    return -1;
}

static const uint8_t *pb_skip(const uint8_t *p, const uint8_t *end, int wt) {
    if (wt == 0) { int64_t v; const uint8_t *q = p; v = pb_varint(&q, end); (void)v; return q; }
    if (wt == 1) return p + 8 <= end ? p + 8 : end;
    if (wt == 2) {
        int64_t len = pb_varint(&p, end);
        if (len < 0 || len > end - p) return end;
        return p + len;
    }
    if (wt == 5) return p + 4 <= end ? p + 4 : end;
    return end;  /* unknown wire type: stop this message */
}

/* ---- TensorProto parse ---- */
static int parse_tensor(const uint8_t *p, const uint8_t *end, onnx_tensor_t *t) {
    memset(t, 0, sizeof(*t));
    int64_t dims_buf[16];
    int n_dims = 0;
    const uint8_t *raw = NULL; size_t raw_len = 0;
    const uint8_t *fdata = NULL; size_t fdata_len = 0;
    const uint8_t *i64data = NULL; size_t i64data_len = 0;
    const uint8_t *i32data = NULL; size_t i32data_len = 0;

    while (p < end) {
        int64_t tag = pb_varint(&p, end);
        if (tag < 0) break;
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        if (field == 1 && wt == 0 && n_dims < 16) {
            dims_buf[n_dims++] = pb_varint(&p, end);
        } else if (field == 2 && wt == 0) {
            t->data_type = (int)pb_varint(&p, end);
        } else if (field == 4 && wt == 2) {   /* float_data */
            int64_t len = pb_varint(&p, end);
            if (len < 0 || len > end - p) break;
            fdata = p; fdata_len = (size_t)len;   /* p now AT the data */
            p += fdata_len;
        } else if (field == 5 && wt == 2) {   /* int32_data */
            int64_t len = pb_varint(&p, end);
            if (len < 0 || len > end - p) break;
            i32data = p; i32data_len = (size_t)len;
            p += i32data_len;
        } else if (field == 7 && wt == 2) {   /* int64_data */
            int64_t len = pb_varint(&p, end);
            if (len < 0 || len > end - p) break;
            i64data = p; i64data_len = (size_t)len;
            p += i64data_len;
        } else if (field == 8 && wt == 2) {   /* name */
            int64_t len = pb_varint(&p, end);
            if (len < 0 || len > end - p) break;
            t->name = (char *)malloc((size_t)len + 1);
            if (!t->name) break;
            memcpy(t->name, p, (size_t)len);
            t->name[len] = 0;
            p += len;
        } else if (field == 9 && wt == 2) {   /* raw_data */
            int64_t len = pb_varint(&p, end);
            if (len < 0 || len > end - p) break;
            raw = p; raw_len = (size_t)len;   /* p now AT the data */
            p += raw_len;
        } else {
            p = pb_skip(p, end, wt);
            if (p >= end) break;
        }
    }

    if (n_dims > 0) {
        t->dims = (int64_t *)malloc((size_t)n_dims * sizeof(int64_t));
        if (!t->dims) return -1;
        memcpy(t->dims, dims_buf, (size_t)n_dims * sizeof(int64_t));
        t->n_dims = n_dims;
    }
    if (raw) { t->data = raw; t->data_len = raw_len; }
    else if (fdata) { t->data = fdata; t->data_len = fdata_len; }
    else if (i32data) { t->data = i32data; t->data_len = i32data_len; }
    else if (i64data) { t->data = i64data; t->data_len = i64data_len; }
    return 0;
}

/* ---- probe: .onnx + plausible protobuf header (field 1 varint) ---- */
static int onnx_probe(const char *path) {
    const char *ext = strrchr(path, '.');
    if (!ext || strcasecmp(ext, ".onnx") != 0) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint8_t hdr[4];
    size_t n = fread(hdr, 1, 4, f);
    fclose(f);
    /* field 1, wire type 0 (varint), value 1..30 = plausible ir_version */
    return (n >= 2 && hdr[0] == 0x08 && hdr[1] >= 1 && hdr[1] <= 30) ? 1 : 0;
}

static wubu_format_ctx_t *onnx_open(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    onnx_ctx_t *c = (onnx_ctx_t *)calloc(1, sizeof(onnx_ctx_t));
    if (!c) { fclose(f); return NULL; }
    c->fmt = &wubu_format_onnx;   /* registry convention: vtable at ctx[0] */
    c->buf = (uint8_t *)malloc((size_t)sz);
    c->size = (size_t)sz;
    if (!c->buf || fread(c->buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(c->buf); free(c); fclose(f); return NULL;
    }
    fclose(f);

    /* ModelProto: find field 7 (graph) */
    const uint8_t *p = c->buf, *end = c->buf + c->size;
    const uint8_t *graph = NULL; size_t graph_len = 0;
    while (p < end) {
        int64_t tag = pb_varint(&p, end);
        if (tag < 0) break;
        int field = (int)(tag >> 3), wt = (int)(tag & 7);
        if (field == 7 && wt == 2) {
            int64_t len = pb_varint(&p, end);
            if (len < 0 || len > end - p) break;
            graph = p; graph_len = (size_t)len;
            p += len;
        } else {
            p = pb_skip(p, end, wt);
        }
    }
    if (!graph) { free(c->buf); free(c); return NULL; }

    /* GraphProto: field 5 (initializer) — count first, then parse */
    {
        const uint8_t *g = graph, *gend = graph + graph_len;
        int n_init = 0;
        while (g < gend) {
            int64_t tag = pb_varint(&g, gend);
            if (tag < 0) break;
            int field = (int)(tag >> 3), wt = (int)(tag & 7);
            if (field == 5 && wt == 2) {
                int64_t len = pb_varint(&g, gend);
                if (len < 0 || len > gend - g) break;
                n_init++;
                g += len;
            } else {
                g = pb_skip(g, gend, wt);
            }
        }
        c->tensors = (onnx_tensor_t *)calloc(n_init > 0 ? n_init : 1, sizeof(onnx_tensor_t));
        if (!c->tensors) { free(c->buf); free(c); return NULL; }
        g = graph;
        int idx = 0;
        while (g < gend && idx < n_init) {
            int64_t tag = pb_varint(&g, gend);
            if (tag < 0) break;
            int field = (int)(tag >> 3), wt = (int)(tag & 7);
            if (field == 5 && wt == 2) {
                int64_t len = pb_varint(&g, gend);
                if (len < 0 || len > gend - g) break;
                if (parse_tensor(g, g + len, &c->tensors[idx]) == 0)
                    idx++;
                g += len;
            } else {
                g = pb_skip(g, gend, wt);
            }
        }
        c->n_tensors = idx;
    }
    if (c->n_tensors == 0) { free(c->buf); free(c->tensors); free(c); return NULL; }
    return (wubu_format_ctx_t *)c;
}

static void onnx_close(wubu_format_ctx_t *ctx) {
    onnx_ctx_t *c = (onnx_ctx_t *)ctx;
    if (!c) return;
    for (int i = 0; i < c->n_tensors; i++) {
        free(c->tensors[i].name);
        free(c->tensors[i].dims);
    }
    free(c->tensors);
    free(c->buf);
    free(c);
}

static int onnx_get_tensor(wubu_format_ctx_t *ctx, const char *name,
                           const void **data, int *n_dims, const int64_t **shape) {
    onnx_ctx_t *c = (onnx_ctx_t *)ctx;
    if (!name) return 0;
    for (int i = 0; i < c->n_tensors; i++) {
        if (c->tensors[i].name && strcmp(c->tensors[i].name, name) == 0) {
            *data = c->tensors[i].data;
            *n_dims = c->tensors[i].n_dims;
            *shape = c->tensors[i].dims;
            return 1;
        }
    }
    return 0;
}

static const char *onnx_tensor_name(wubu_format_ctx_t *ctx, int idx) {
    onnx_ctx_t *c = (onnx_ctx_t *)ctx;
    return (idx >= 0 && idx < c->n_tensors) ? c->tensors[idx].name : NULL;
}

static int onnx_tensor_count(wubu_format_ctx_t *ctx) {
    onnx_ctx_t *c = (onnx_ctx_t *)ctx;
    return c->n_tensors;
}

static int onnx_get_int(wubu_format_ctx_t *ctx, const char *key, int64_t *val) {
    (void)ctx; (void)key; (void)val;
    return 0;  /* scalar metadata not extracted (not needed for loading) */
}
static int onnx_get_str(wubu_format_ctx_t *ctx, const char *key, const char **val) {
    (void)ctx; (void)key; (void)val;
    return 0;
}
static const char *onnx_meta_key(wubu_format_ctx_t *ctx, int idx) {
    (void)ctx; (void)idx;
    return NULL;
}
static int onnx_meta_count(wubu_format_ctx_t *ctx) {
    (void)ctx;
    return 0;
}

wubu_model_format_t wubu_format_onnx = {
    .name      = "onnx",
    .extension = ".onnx",
    .probe     = onnx_probe,
    .open      = onnx_open,
    .close     = onnx_close,
    .get_tensor = onnx_get_tensor,
    .tensor_name = onnx_tensor_name,
    .tensor_count = onnx_tensor_count,
    .get_int   = onnx_get_int,
    .get_str   = onnx_get_str,
    .meta_key  = onnx_meta_key,
    .meta_count = onnx_meta_count,
};

/* registration lives in wubu_model_format.c (wubu_format_onnx extern) */
