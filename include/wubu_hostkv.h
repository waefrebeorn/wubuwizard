#ifndef WUBU_HOSTKV_H
#define WUBU_HOSTKV_H

/*
 * wubu_hostkv.h -- EG-47: INT8 host-tier KV row-local record
 * (HiQCache lineage), reimplemented in C11.
 *
 * Each KV row is packed into a fixed 1152-byte aligned record: a 64-byte
 * header, one BF16 absmax scale per head, then INT8 payload. The CUDA mover
 * treats the record as opaque bytes, so no device-side format change is
 * needed when a row moves between host tiers.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_HOSTKV_RECORD_BYTES 1152u
#define WUBU_HOSTKV_HEADER_BYTES 64u
#define WUBU_HOSTKV_MAGIC 0x4b564849u /* "HIKV" */

/* Pack heads*head_dim floats into an aligned record (record_cap >= 1152).
 * Returns bytes used or -1 on bad arguments. */
int wubu_hostkv_pack(const float *row, int heads, int head_dim,
                     unsigned char *record, size_t record_cap);

/* Unpack a record previously produced by wubu_hostkv_pack.
 * Returns 0 or -1. */
int wubu_hostkv_unpack(const unsigned char *record, size_t used,
                       float *row, int heads, int head_dim);

/* Opaque mover: copies whole records without interpreting them. */
int wubu_hostkv_mover_copy(const unsigned char *src, unsigned char *dst,
                           size_t nbytes);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_HOSTKV_H */
