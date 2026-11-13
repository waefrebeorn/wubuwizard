/*
 * wubu_dht_table.c -- the DHT routing table (BEP 5, the state half).
 *
 * C11, libc only.
 */
#include "wubu_dht_table.h"

#include <string.h>

static const uint8_t g_zero_id[WUBU_DHT_ID_LEN];

void wubu_dht_table_init(wubu_dht_table_t *t, const uint8_t mine[WUBU_DHT_ID_LEN])
{
    memset(t, 0, sizeof(*t));
    if (mine) {
        memcpy(t->mine, mine, WUBU_DHT_ID_LEN);
        t->have_mine = 1;
    }
}

static int is_zero(const uint8_t id[WUBU_DHT_ID_LEN])
{
    return memcmp(id, g_zero_id, WUBU_DHT_ID_LEN) == 0;
}

int wubu_dht_table_add(wubu_dht_table_t *t, const uint8_t id[WUBU_DHT_ID_LEN],
                       uint32_t vip, uint16_t vport, uint64_t now_ms)
{
    if (!t || !id || !t->have_mine) return -1;
    if (is_zero(id)) return -1;
    if (memcmp(id, t->mine, WUBU_DHT_ID_LEN) == 0) return -1;

    int bi = wubu_dht_shared_prefix_bits(t->mine, id);

    /* Already known: refresh it in place, never duplicate. */
    for (int i = 0; i < t->count[bi]; i++) {
        if (memcmp(t->bucket[bi][i].id, id, WUBU_DHT_ID_LEN) == 0) {
            t->bucket[bi][i].vip        = vip;
            t->bucket[bi][i].vport      = vport;
            t->bucket[bi][i].last_seen_ms = now_ms;
            return 0;
        }
    }

    /* Full bucket: evict the STALEST so a live peer can take the slot. */
    if (t->count[bi] >= WUBU_DHT_BUCKET_MAX) {
        int stale = 0;
        for (int i = 1; i < t->count[bi]; i++)
            if (t->bucket[bi][i].last_seen_ms <
                t->bucket[bi][stale].last_seen_ms)
                stale = i;
        t->bucket[bi][stale] = (wubu_dht_peer_t){
            .vip = vip, .vport = vport, .last_seen_ms = now_ms
        };
        memcpy(t->bucket[bi][stale].id, id, WUBU_DHT_ID_LEN);
        return 0;
    }

    wubu_dht_peer_t *p = &t->bucket[bi][t->count[bi]];
    memcpy(p->id, id, WUBU_DHT_ID_LEN);
    p->vip = vip;
    p->vport = vport;
    p->last_seen_ms = now_ms;
    t->count[bi]++;
    t->total++;
    return 0;
}

/* Selection of up to `max` nearest peers from the whole table.  A full sort is
 * unnecessary and would allocate; an insertion into a fixed `out` keeps the
 * cost bounded by max*total and needs no heap. */
static int collect_nearest(const wubu_dht_table_t *t,
                           const uint8_t target[WUBU_DHT_ID_LEN],
                           uint8_t *out, int max)
{
    if (!t || !target || !out || max <= 0) return 0;

    int n = 0;
    for (int b = 0; b < WUBU_DHT_TABLE_BUCKETS; b++) {
        for (int i = 0; i < t->count[b]; i++) {
            const wubu_dht_peer_t *p = &t->bucket[b][i];
            /* Walk back while the previous entry is FARTHER than this one;
             * stop as soon as it is nearer-or-equal, so the list stays
             * non-decreasing by distance. */
            int pos = n;
            while (pos > 0) {
                const uint8_t *prev = out + (size_t)(pos - 1) * WUBU_DHT_ID_LEN;
                if (wubu_dht_distance_cmp(target, prev, p->id) <= 0) break;
                pos--;
            }
            if (pos >= max) continue;        /* not close enough to keep */

            /* Shift down to make room, dropping the farthest when full. */
            int tail = (n < max) ? n : max - 1;
            for (int k = tail; k > pos; k--)
                memcpy(out + (size_t)k * WUBU_DHT_ID_LEN,
                       out + (size_t)(k - 1) * WUBU_DHT_ID_LEN,
                       WUBU_DHT_ID_LEN);
            memcpy(out + (size_t)pos * WUBU_DHT_ID_LEN, p->id, WUBU_DHT_ID_LEN);
            if (n < max) n++;
        }
    }
    return n;
}

int wubu_dht_table_closest(const wubu_dht_table_t *t,
                           const uint8_t target[WUBU_DHT_ID_LEN],
                           uint8_t *out, int max)
{
    if (!t || !target || !out || max <= 0) return 0;
    int n = collect_nearest(t, target, out, max);
    return n;
}

int wubu_dht_table_find_node(const wubu_dht_table_t *t,
                             const uint8_t target[WUBU_DHT_ID_LEN],
                             uint8_t *out, int max)
{
    /* find_node answers with the k closest WE know; that is exactly
     * closest(), capped at k. */
    if (max > WUBU_DHT_K) max = WUBU_DHT_K;
    if (max <= 0) max = WUBU_DHT_K;
    return wubu_dht_table_closest(t, target, out, max);
}

int wubu_dht_table_reap(wubu_dht_table_t *t, uint64_t now_ms, uint64_t window_ms)
{
    if (!t) return -1;
    int reaped = 0;
    for (int b = 0; b < WUBU_DHT_TABLE_BUCKETS; b++) {
        int w = 0;
        for (int i = 0; i < t->count[b]; i++) {
            wubu_dht_peer_t *p = &t->bucket[b][i];
            if (now_ms - p->last_seen_ms > window_ms) { reaped++; continue; }
            if (w != i) t->bucket[b][w] = *p;
            w++;
        }
        t->count[b] = w;
    }
    t->total -= reaped;
    return reaped;
}

int wubu_dht_table_count(const wubu_dht_table_t *t)
{
    return t ? t->total : 0;
}
