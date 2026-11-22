#ifndef WUBU_SSD_ROW_REUSE_H
#define WUBU_SSD_ROW_REUSE_H

/*
 * wubu_ssd_row_reuse.h -- EG-07: sliding-window weight-row reuse with
 * coalesced reads (LLM in a Flash §2-3 lineage), reimplemented in C11.
 *
 * A window of k consecutive rows is kept resident. Advancing by one row reads
 * only the entering row instead of re-reading the whole window; a full refill
 * is issued in coalesced chunks of at least coalesce_bytes. Read syscalls and
 * bytes are counted so a gate can prove the reuse actually happened.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_ssd_row_reader wubu_ssd_row_reader_t;

wubu_ssd_row_reader_t *wubu_ssd_row_reader_create(const uint8_t *data,
                                                   size_t size,
                                                   size_t row_bytes,
                                                   int window_rows,
                                                   size_t coalesce_bytes);
void wubu_ssd_row_reader_free(wubu_ssd_row_reader_t *r);

/* Make [start, start+window_rows) resident and return a pointer to the
 * contiguous window. The pointer is valid until the next window call. */
const uint8_t *wubu_ssd_row_window(wubu_ssd_row_reader_t *r, int start);

size_t wubu_ssd_row_reader_reads(const wubu_ssd_row_reader_t *r);
uint64_t wubu_ssd_row_reader_bytes(const wubu_ssd_row_reader_t *r);
int wubu_ssd_row_reader_delta_updates(const wubu_ssd_row_reader_t *r);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SSD_ROW_REUSE_H */
