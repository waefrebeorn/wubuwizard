/*
 * wubu_hostkv.c -- EG-47 INT8 host-tier KV record. Pure C11.
 */
#include "wubu_hostkv.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static uint16_t to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    return (uint16_t)(u >> 16);
}

static float from_bf16(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

static uint32_t get_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_u16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
}

static uint16_t get_u16(const unsigned char *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

int wubu_hostkv_pack(const float *row, int heads, int head_dim,
                     unsigned char *record, size_t record_cap) {
    size_t elems, payload_off, used;
    if (!row || !record || heads <= 0 || head_dim <= 0 ||
        record_cap < WUBU_HOSTKV_RECORD_BYTES) return -1;
    elems = (size_t)heads * (size_t)head_dim;
    payload_off = WUBU_HOSTKV_HEADER_BYTES + (size_t)heads * 2u;
    used = payload_off + elems;
    if (used > WUBU_HOSTKV_RECORD_BYTES ||
        payload_off < WUBU_HOSTKV_HEADER_BYTES) return -1;

    memset(record, 0, WUBU_HOSTKV_RECORD_BYTES);
    put_u32(record, WUBU_HOSTKV_MAGIC);
    put_u32(record + 4, (uint32_t)heads);
    put_u32(record + 8, (uint32_t)head_dim);
    put_u32(record + 12, (uint32_t)elems);
    put_u32(record + 16, (uint32_t)payload_off);

    for (int h = 0; h < heads; h++) {
        const float *src = row + (size_t)h * head_dim;
        float absmax = 0.0f;
        float scale;
        for (int d = 0; d < head_dim; d++) {
            float a = fabsf(src[d]);
            if (a > absmax) absmax = a;
        }
        scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;
        put_u16(record + WUBU_HOSTKV_HEADER_BYTES + (size_t)h * 2u,
                to_bf16(scale));
        for (int d = 0; d < head_dim; d++) {
            float q = src[d] / scale;
            int v = (int)(q >= 0.0f ? floorf(q + 0.5f) : ceilf(q - 0.5f));
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            record[payload_off + (size_t)h * head_dim + d] =
                (unsigned char)(signed char)v;
        }
    }
    return (int)used;
}

int wubu_hostkv_unpack(const unsigned char *record, size_t used,
                       float *row, int heads, int head_dim) {
    unsigned elems, payload_off;
    if (!record || !row || heads <= 0 || head_dim <= 0 ||
        used < WUBU_HOSTKV_HEADER_BYTES ||
        used > WUBU_HOSTKV_RECORD_BYTES) return -1;
    if (get_u32(record) != WUBU_HOSTKV_MAGIC ||
        (int)get_u32(record + 4) != heads ||
        (int)get_u32(record + 8) != head_dim) return -1;
    elems = get_u32(record + 12);
    payload_off = get_u32(record + 16);
    if (elems != (unsigned)heads * (unsigned)head_dim ||
        payload_off + elems != used ||
        used > WUBU_HOSTKV_RECORD_BYTES) return -1;

    for (int h = 0; h < heads; h++) {
        float scale = from_bf16(
            get_u16(record + WUBU_HOSTKV_HEADER_BYTES + (size_t)h * 2u));
        for (int d = 0; d < head_dim; d++) {
            signed char q = (signed char)record[payload_off +
                (size_t)h * head_dim + d];
            row[(size_t)h * head_dim + d] = (float)q * scale;
        }
    }
    return 0;
}

int wubu_hostkv_mover_copy(const unsigned char *src, unsigned char *dst,
                           size_t nbytes) {
    if (!src || !dst || nbytes == 0) return -1;
    memcpy(dst, src, nbytes);
    return 0;
}
