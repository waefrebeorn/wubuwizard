#ifndef WUBU_SLEEP_ALLOC_H
#define WUBU_SLEEP_ALLOC_H

/*
 * wubu_sleep_alloc.h -- EG-16: tagged sleep/wake allocator (vLLM
 * CuMemAllocator.offload_tags lineage), reimplemented in portable C11.
 *
 * Models the device-resident byte budget with a real VRAM-sized host arena:
 * each allocation owns a fixed slot. sleep(tag_mask) copies tagged slots to
 * host staging and releases their device bytes; wake(tag_mask) restores only
 * the requested tags back into the same slots. Data is byte-exact through the
 * round trip. This is the policy/accounting model; actual CUDA VRAM release
 * is a backend concern and is not claimed by this gate.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_sleep_alloc wubu_sleep_alloc_t;

wubu_sleep_alloc_t *wubu_sleep_alloc_create(size_t device_capacity);
void wubu_sleep_alloc_free(wubu_sleep_alloc_t *a);

/* Allocate `bytes` in a fixed slot owned by `tag` (bitmask). NULL on OOM. */
void *wubu_sleep_alloc_push(wubu_sleep_alloc_t *a, const char *name,
                            unsigned tag, size_t bytes);

/* Copy tagged resident slots to host, release their device bytes.
 * Returns the number of blocks offloaded, or -1 on bad argument. */
int wubu_sleep_alloc_sleep(wubu_sleep_alloc_t *a, unsigned tag_mask);

/* Restore tagged host-staged slots into their original device slots.
 * Returns the number of blocks restored, or -1 on bad argument. */
int wubu_sleep_alloc_wake(wubu_sleep_alloc_t *a, unsigned tag_mask);

size_t wubu_sleep_alloc_device_used(const wubu_sleep_alloc_t *a);
size_t wubu_sleep_alloc_host_bytes(const wubu_sleep_alloc_t *a);
int    wubu_sleep_alloc_resident(const wubu_sleep_alloc_t *a, const char *name);
int    wubu_sleep_alloc_nblocks(const wubu_sleep_alloc_t *a);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SLEEP_ALLOC_H */
