#ifndef WUBU_VIRT_KV_H
#define WUBU_VIRT_KV_H

/*
 * wubu_virt_kv.h -- EG-107: vAttention-style KV memory contract.
 * A large contiguous virtual KV range is reserved up front; physical pages
 * are mapped on demand as sequence lengths grow, so existing attention kernels
 * still see one contiguous pointer per request. This C11 gate models the
 * reservation/mapping contract, not CUDA VMM calls or measured throughput.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_VIRT_MAX_BATCH 32

typedef struct wubu_virt_kv wubu_virt_kv_t;

wubu_virt_kv_t *wubu_virt_kv_create(int batch, int max_len,
                                    int page_size, int physical_pages);
void wubu_virt_kv_free(wubu_virt_kv_t *p);

size_t wubu_virt_kv_virtual_tokens(const wubu_virt_kv_t *p);

/* Allocate a batch index for a request id. Returns 0 on success, -1 if full. */
int wubu_virt_kv_alloc_req(wubu_virt_kv_t *p, int request_id);

/* Map pages needed by each active sequence length. 0/-1 on page exhaustion. */
int wubu_virt_kv_step(wubu_virt_kv_t *p, const int *seq_len, int n);

/* Contiguous virtual token offset for (batch index, token). */
long wubu_virt_kv_offset(const wubu_virt_kv_t *p, int batch_idx, int token);

size_t wubu_virt_kv_mapped_pages(const wubu_virt_kv_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_VIRT_KV_H */
