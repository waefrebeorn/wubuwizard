#ifndef WUBU_AOT_PACK_H
#define WUBU_AOT_PACK_H

/*
 * wubu_aot_pack.h -- EG-66c: AOT kernel pack selection by shape.
 * A build-time table maps (dtype, token count) ranges to compiled kernel
 * variants. Ranges for a dtype must not overlap, so selection is unambiguous
 * and needs no runtime cost model. This gates the dispatch table, not the
 * compiled kernels themselves.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_AOT_MAX_PACKS 16

typedef struct {
    int dtype;
    int min_n, max_n;
    int variant;
} wubu_aot_pack_entry_t;

typedef struct wubu_aot_pack wubu_aot_pack_t;

wubu_aot_pack_t *wubu_aot_pack_create(const wubu_aot_pack_entry_t *entries, int n);
void wubu_aot_pack_free(wubu_aot_pack_t *p);

int wubu_aot_pack_count(const wubu_aot_pack_t *p);

/* 0 and writes the variant; -1 when the shape is unsupported. */
int wubu_aot_pack_select(const wubu_aot_pack_t *p, int dtype, int n,
                         int *variant_out);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_AOT_PACK_H */
