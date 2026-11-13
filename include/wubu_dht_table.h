/*
 * wubu_dht_table.h -- the DHT routing table (BEP 5, the state half).
 *
 * The addressing core (wubu_dht.c) knows who a node IS; this module remembers
 * WHO ANSWERED.  A DHT without state is just a protocol definition: it can
 * encode a question and parse an answer, but it cannot choose whom to ask or
 * notice that someone stopped replying.
 *
 * The rules that matter:
 *
 *   * Buckets are keyed by SHARED PREFIX LENGTH AGAINST OUR OWN ID, so a
 *     bucket holds nodes at roughly the same distance from us.  Bucket 159 is
 *     the exact-match neighbourhood (one bit differs), bucket 0 is everything
 *     that shares nothing with us.
 *   * A bucket holds at most 8.  When it is full the STALEST entry is evicted
 *     -- a live peer is worth more than a remembered one, and a dead slot
 *     otherwise stays occupied forever.
 *   * Our own id is never stored: asking ourselves is a loop, and reporting
 *     ourselves as a candidate is a lie.
 *   * A zero id is refused for the same reason: it is what an uninitialised
 *     peer looks like, and treating it as real poisons every distance
 *     calculation downstream.
 *
 * C11, libc only.
 */
#ifndef WUBU_DHT_TABLE_H
#define WUBU_DHT_TABLE_H

#include <stdbool.h>
#include <stdint.h>

#include "wubu_dht.h"

#define WUBU_DHT_TABLE_BUCKETS 160       /* one per shared-prefix length */
#define WUBU_DHT_BUCKET_MAX     8
#define WUBU_DHT_K              8        /* the "k" of kademlia */

typedef struct {
    uint8_t  id[WUBU_DHT_ID_LEN];
    uint32_t vip;                        /* the SDR virtual address */
    uint16_t vport;
    uint64_t last_seen_ms;               /* liveness; eviction picks the least */
} wubu_dht_peer_t;

typedef struct {
    uint8_t        mine[WUBU_DHT_ID_LEN];          /* our own id */
    int            have_mine;
    wubu_dht_peer_t bucket[WUBU_DHT_TABLE_BUCKETS][WUBU_DHT_BUCKET_MAX];
    int            count[WUBU_DHT_TABLE_BUCKETS];
    int            total;
} wubu_dht_table_t;

/* Bind the table to OUR id.  Every bucket index is measured from it, so a
 * table without one cannot place anything. */
void wubu_dht_table_init(wubu_dht_table_t *t, const uint8_t mine[WUBU_DHT_ID_LEN]);

/* Record that `id` answered from `vip:vport` at `now_ms`.
 *  - an existing entry for that id is REFRESHED, never duplicated;
 *  - a full bucket evicts its stalest member first;
 *  - our own id and the zero id are refused.
 * Returns 0 on success, -1 when refused (or the table has no local id). */
int wubu_dht_table_add(wubu_dht_table_t *t, const uint8_t id[WUBU_DHT_ID_LEN],
                       uint32_t vip, uint16_t vport, uint64_t now_ms);

/* Copy the up-to-`max` ids closest to `target`, nearest first, into `out`
 * (each row WUBU_DHT_ID_LEN bytes).  Returns how many were written. */
int wubu_dht_table_closest(const wubu_dht_table_t *t,
                           const uint8_t target[WUBU_DHT_ID_LEN],
                           uint8_t *out, int max);

/* The answer to a find_node query for `target`: the k closest ids WE know.
 * Same ordering as wubu_dht_table_closest. */
int wubu_dht_table_find_node(const wubu_dht_table_t *t,
                             const uint8_t target[WUBU_DHT_ID_LEN],
                             uint8_t *out, int max);

/* Drop entries not seen within `window_ms` of `now_ms`.  Returns how many
 * were reaped -- the same discipline as the SDR session layer, where a port is
 * reusable only after its peer is reaped, not merely silent. */
int wubu_dht_table_reap(wubu_dht_table_t *t, uint64_t now_ms, uint64_t window_ms);

int wubu_dht_table_count(const wubu_dht_table_t *t);

#endif /* WUBU_DHT_TABLE_H */
