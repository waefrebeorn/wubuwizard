#ifndef WUBU_CONV_STORE_H
#define WUBU_CONV_STORE_H

/*
 * wubu_conv_store.h -- EG-99: content-addressed conversation/KV blocks.
 * basalt H3: blocks are keyed by the token prefix they belong to AND by their
 * bytes, so shared prefixes deduplicate across conversations. Checkpoints are
 * bounded to the root and the last N turn ends.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_CONV_MAX_FILES 4096
#define WUBU_CONV_MAX_CP    64

typedef struct wubu_conv_store wubu_conv_store_t;

/* block_bytes >= 1, turn_states >= 1. NULL on bad input. */
wubu_conv_store_t *wubu_conv_store_create(size_t block_bytes, int turn_states);
void wubu_conv_store_free(wubu_conv_store_t *p);

/* Store data in block-sized pieces keyed by (prefix_id, bytes).
 * Returns the number of new physical files written, -1 on bad input. */
int wubu_conv_store_write(wubu_conv_store_t *p, uint64_t prefix_id,
                          const unsigned char *data, size_t bytes);

int wubu_conv_store_file_count(const wubu_conv_store_t *p);
int wubu_conv_store_has_block(const wubu_conv_store_t *p, uint64_t prefix_id,
                              const unsigned char *data, size_t bytes);

/* Keep the root checkpoint plus the most recent turn_states turns. */
int wubu_conv_store_checkpoint(wubu_conv_store_t *p, int is_root, int token_pos);
int wubu_conv_store_checkpoint_count(const wubu_conv_store_t *p);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_CONV_STORE_H */
