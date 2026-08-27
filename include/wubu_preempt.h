#ifndef WUBU_PREEMPT_H
#define WUBU_PREEMPT_H

#include <stdint.h>
#include <stddef.h>

/* Callback to read one KV page into buf; returns bytes written, or 0 on error */
typedef size_t (*wubu_preempt_kv_reader_t)(void *userdata, int page_idx,
                                           void *buf, size_t cap);

/* Callback to write one KV page from buf; returns bytes consumed, or 0 on error */
typedef size_t (*wubu_preempt_kv_writer_t)(void *userdata, int page_idx,
                                           const void *buf, size_t len);

typedef struct wubu_preempt wubu_preempt_t;

/* Create preemption context for checkpoint_dir */
wubu_preempt_t *wubu_preempt_create(const char *checkpoint_dir);

/* Destroy context */
void wubu_preempt_destroy(wubu_preempt_t *p);

/* Save sequence state to checkpoint file */
int wubu_preempt_save(wubu_preempt_t *p, uint32_t seq_id,
                      int n_tokens, int n_kv_pages, int pos,
                      float temperature, int top_k, float top_p,
                      wubu_preempt_kv_reader_t reader, void *userdata);

/* Restore sequence state from checkpoint file */
int wubu_preempt_restore(wubu_preempt_t *p, uint32_t seq_id,
                         wubu_preempt_kv_writer_t writer, void *userdata);

/* Check if checkpoint exists for seq_id */
int wubu_preempt_exists(const wubu_preempt_t *p, uint32_t seq_id);

/* Delete checkpoint for seq_id */
int wubu_preempt_delete(wubu_preempt_t *p, uint32_t seq_id);

#endif /* WUBU_PREEMPT_H */
