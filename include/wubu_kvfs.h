/* wubu_kvfs.h — KV namespace layer (G1: path-addressable KV cache)
 *
 * The KV cache is a file system. This module provides the address
 * layer: paths resolve to (block, offset) pairs that route
 * directly into the KV cache tensors. Read and write go through
 * the namespace — every datum is a file with a path.
 *
 * Speed-kernel contract:
 *   - Resolution is O(1): the mount table is an FNV-1a hash table
 *     (open addressing), not a linear scan. Longest-prefix lookup
 *     walks parent segments, each a hash probe.
 *   - Resolve once, use many: wubu_kvfs_open() returns an opaque
 *     handle carrying the precomputed absolute float offset and
 *     limit. Handle reads/writes are a bounds check + memcpy —
 *     zero string ops on the hot path.
 *   - The mount struct is cold: paths are only touched at mount/
 *     snapshot time, never on the I/O path.
 *
 * C11, opaque struct, minimal includes. No third-party deps.
 *
 * API:
 *   wubu_kvfs_create(block_size, n_blocks) — create the namespace
 *   wubu_kvfs_mount(fs, path, start_block, n_blocks) — mount a region
 *   wubu_kvfs_unmount(fs, path)            — unmount a subtree
 *   wubu_kvfs_lookup(fs, path, out_block, out_offset) — resolve path
 *   wubu_kvfs_open(fs, path)               — resolve to a hot handle
 *   wubu_kvfs_handle_read(h, kv_base, dst, n)  — hot read (memcpy)
 *   wubu_kvfs_handle_write(h, kv_base, src, n) — hot write (memcpy)
 *   wubu_kvfs_handle_close(h)              — release the handle
 *   wubu_kvfs_block_size(fs)               — the namespace block size
 *   wubu_kvfs_read(fs, path, kv_base, dst, n)  — path read (convenience)
 *   wubu_kvfs_write(fs, path, kv_base, src, n) — path write (convenience)
 *   wubu_kvfs_snapshot_json(fs, out_len)   — JSON view of the namespace
 *   wubu_kvfs_mount_count(fs)              — registered mount count
 *   wubu_kvfs_free(fs)                     — destroy
 *
 * The namespace IS the KV cache. Every datum is a file with a path.
 * The mount table is the address translation layer (virtual→physical,
 * same idea as OS page tables / PagedAttention). The tensor layer
 * (paged/ring KV, MLA latent space) stays intact; only the addressing
 * layer is new.
 *
 * Design lineage (7-hop):
 *   PagedAttention (SOSP'23) → block tables = virtual memory
 *   RadixAttention (2312.07104) → radix tree = directory structure
 *   MemGPT (2310.08560) → the model pages its own memory
 *   Mooncake (FAST'25) → disaggregated KV pool over CPU/DRAM/SSD
 *   LMCache (2510.09665) → tiered persistent KV chunks as files
 *   Infini-attention (2404.07143) → compressive synthesized writes
 *   Gemma 3 12B (2503.19786) → single encoder, all inputs are data
 */

#ifndef WUBU_KVFS_H
#define WUBU_KVFS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque namespace handle */
typedef struct wubu_kvfs wubu_kvfs_t;

/* Opaque resolved-path handle: precomputed (offset, limit).
 * Created once by wubu_kvfs_open(), then used for hot I/O. */
typedef struct wubu_kvfs_handle wubu_kvfs_handle_t;

/* A mounted KV region: a path prefix that maps to a contiguous
 * block range in the flat KV cache tensor. */
typedef struct {
    char path[256];          /* mount path, e.g. "/kv/in" (cold) */
    uint32_t start_block;    /* first block in the flat KV range */
    uint32_t n_blocks;       /* how many blocks this mount covers */
    uint32_t block_size;     /* floats per block */
    size_t   abs_offset;     /* precomputed start_block * block_size */
    size_t   abs_limit;      /* precomputed (start_block + n_blocks) * block_size */
} wubu_kvfs_mount_t;

/* Create a KV namespace with a given block size and total block
 * count. Returns NULL on allocation failure. */
wubu_kvfs_t *wubu_kvfs_create(uint32_t block_size, uint32_t total_blocks);

/* Mount a KV region at a path. The path becomes a directory entry.
 * Returns 0 on success, -1 if the path is already mounted or out
 * of blocks. */
int wubu_kvfs_mount(wubu_kvfs_t *fs, const char *path,
                    uint32_t start_block, uint32_t n_blocks);

/* Unmount a subtree (removes the mount and all children).
 * Returns 0 on success, -1 if path not found. */
int wubu_kvfs_unmount(wubu_kvfs_t *fs, const char *path);

/* Resolve a path to a (block, offset) pair. The block is the
 * index into the flat KV tensor; offset is the byte offset
 * within that block. Returns 0 on success, -1 if the path
 * is not mounted or does not exist. */
int wubu_kvfs_lookup(const wubu_kvfs_t *fs, const char *path,
                     uint32_t *out_block, size_t *out_offset);

/* ---- speed-kernel hot path: resolve once, use many ---- */

/* Resolve a path to an opaque handle. The handle precomputes the
 * absolute float offset and the byte limit of the mount, so
 * wubu_kvfs_handle_read/write are a bounds check + memcpy with
 * zero string operations. Returns NULL if the path is unmounted.
 * The handle references fs; call wubu_kvfs_handle_close() when
 * done, and do not use handles after wubu_kvfs_free(fs). */
wubu_kvfs_handle_t *wubu_kvfs_open(const wubu_kvfs_t *fs, const char *path);

/* Hot read: copy n_floats floats from the KV tensor at the
 * handle's precomputed offset into dst. Returns 0 on success,
 * -1 if the read would exceed the mount limit or args invalid. */
int wubu_kvfs_handle_read(const wubu_kvfs_handle_t *h,
                          const float *kv_base, float *dst, size_t n_floats);

/* Hot write: copy n_floats floats from src into the KV tensor at
 * the handle's precomputed offset. Returns 0 on success, -1 if
 * the write would exceed the mount limit or args invalid. */
int wubu_kvfs_handle_write(const wubu_kvfs_handle_t *h,
                           float *kv_base, const float *src, size_t n_floats);

/* Precomputed absolute float offset of this handle. */
size_t wubu_kvfs_handle_offset(const wubu_kvfs_handle_t *h);

/* Maximum floats this handle can address (mount limit - offset). */
size_t wubu_kvfs_handle_capacity(const wubu_kvfs_handle_t *h);

/* Release a resolved handle. */
void wubu_kvfs_handle_close(wubu_kvfs_handle_t *h);

/* The namespace's block size (floats per block). */
uint32_t wubu_kvfs_block_size(const wubu_kvfs_t *fs);

/* ---- path-based convenience (cold-ish: resolves then I/Os) ---- */

/* Read KV data from a path into dst. Reads n_floats floats
 * starting at the resolved (block, offset). kv_base is the
 * base pointer of the flat KV tensor. Returns 0 on success,
 * -1 if the path is not mounted or the read would exceed
 * the mount's block range. */
int wubu_kvfs_read(const wubu_kvfs_t *fs, const char *path,
                       const float *kv_base, float *dst, size_t n_floats);

/* Write KV data from src into a path. Writes n_floats floats
 * starting at the resolved (block, offset). kv_base is the
 * base pointer of the flat KV tensor. Returns 0 on success,
 * -1 if the path is not mounted or the write would exceed
 * the mount's block range. */
int wubu_kvfs_write(wubu_kvfs_t *fs, const char *path,
                        float *kv_base, const float *src, size_t n_floats);

/* JSON snapshot of the namespace: every mount point with its
 * block range. Caller frees with free(). */
char *wubu_kvfs_snapshot_json(const wubu_kvfs_t *fs, size_t *out_len);

/* Registered mount count. */
int wubu_kvfs_mount_count(const wubu_kvfs_t *fs);

/* Destroy the namespace. Does NOT free the backing KV tensor.
 * Any open handles become invalid. */
void wubu_kvfs_free(wubu_kvfs_t *fs);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_KVFS_H */
