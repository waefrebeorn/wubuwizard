/*
 * wubu_dht.h -- BitTorrent DHT (BEP 5): node identity, distance, KRPC.
 *
 * Why: a magnet names a model by info-hash, but something has to say WHERE.
 * Trackers are one answer and they are a host that can disappear; the DHT is
 * the answer where no host is needed -- peers ask each other, and the answer
 * is found by walking a keyspace rather than looking up a record.
 *
 * The keyspace rules that matter:
 *
 *   * A NODE ID is DERIVED, never typed in.  It is SHA-1 over whatever the
 *     node commits to (its seed), so two nodes cannot claim the same id by
 *     both typing it, and an id cannot be chosen to sit somewhere convenient.
 *   * Distance is XOR over the 160-bit id, compared big-endian.  It is a
 *     METRIC (a^b == 0 iff a==b, symmetric, triangle inequality), which is
 *     what lets the walk converge.
 *   * Bucketing is by SHARED PREFIX LENGTH, so a bucket holds ids at roughly
 *     the same distance and the closest buckets are known precisely.
 *
 * This module covers identity, distance and the four KRPC messages the swarm
 * needs (ping, find_node, get_peers, announce_peer).  It does not carry
 * traffic: encoding produces bytes, decoding validates them, and the socket
 * belongs to the caller -- the same split every other leg uses.
 *
 * C11, libc only.
 */
#ifndef WUBU_DHT_H
#define WUBU_DHT_H

#include <stddef.h>
#include <stdint.h>

#define WUBU_DHT_ID_LEN      20
#define WUBU_DHT_TOKEN_LEN   8    /* opaque, per-request, returned verbatim */
#define WUBU_DHT_MAX_MSG     4096

typedef enum {
    WUBU_DHT_Q_PING = 1,
    WUBU_DHT_Q_FIND_NODE = 2,
    WUBU_DHT_Q_GET_PEERS = 3,
    WUBU_DHT_Q_ANNOUNCE_PEER = 4
} wubu_dht_query_t;

/* Derive a node id from `seed`.  SHA-1(seed) -- deterministic, so the same
 * seed yields the same id, and an id cannot be chosen.  Returns 0, or -1 on
 * NULL output. */
int wubu_dht_node_id(const uint8_t *seed, size_t len,
                     uint8_t id[WUBU_DHT_ID_LEN]);

/* d = a XOR b, byte for byte. */
void wubu_dht_distance(const uint8_t a[WUBU_DHT_ID_LEN],
                       const uint8_t b[WUBU_DHT_ID_LEN],
                       uint8_t out[WUBU_DHT_ID_LEN]);

/* Is |a,x| <, == or > |b,x|?  Returns -1/0/+1.  Used to order candidates. */
int wubu_dht_distance_cmp(const uint8_t x[WUBU_DHT_ID_LEN],
                          const uint8_t a[WUBU_DHT_ID_LEN],
                          const uint8_t b[WUBU_DHT_ID_LEN]);

/* How many leading BITS a and b share (0..160).  This is the bucket index. */
int wubu_dht_shared_prefix_bits(const uint8_t a[WUBU_DHT_ID_LEN],
                                const uint8_t b[WUBU_DHT_ID_LEN]);

/* -- KRPC ------------------------------------------------------- */

typedef struct {
    uint8_t id[WUBU_DHT_ID_LEN];         /* sender id (request) / responder */
    uint8_t target[WUBU_DHT_ID_LEN];     /* find_node target / info-hash */
    uint8_t token[WUBU_DHT_TOKEN_LEN];   /* announce/get_peers credential */
    uint8_t txn[4];                      /* transaction id; echoed verbatim */
    uint16_t port;                       /* announce: the peer's port */
    int      have_target;
    int      have_token;
    int      have_port;
    int      have_txn;
} wubu_dht_args_t;

typedef struct {
    wubu_dht_query_t query;
    char             name[16];           /* "ping", "find_node", ... */
} wubu_dht_query_info_t;

const wubu_dht_query_info_t *wubu_dht_query_info(wubu_dht_query_t q);
const wubu_dht_query_info_t *wubu_dht_query_by_name(const char *name);

/* Encode a request.  Returns bytes written, or -1 on a bad argument or a
 * buffer too small. */
long wubu_dht_encode_request(wubu_dht_query_t q, const wubu_dht_args_t *args,
                             uint8_t *out, size_t n);

/* Encode a response carrying only the responder id, or one carrying a
 * 20-byte value (a node id, or the close-node list is left to the caller).
 * Returns bytes written, or -1. */
long wubu_dht_encode_response(const uint8_t id[WUBU_DHT_ID_LEN],
                              const uint8_t token[WUBU_DHT_TOKEN_LEN],
                              int have_token, const uint8_t *value,
                              size_t value_len, uint8_t *out, size_t n);

typedef struct {
    int      is_response;                /* 1 = r, 0 = e */
    int      is_error;
    int      error_code;
    uint8_t  id[WUBU_DHT_ID_LEN];
    uint8_t  token[WUBU_DHT_TOKEN_LEN];
    uint8_t  value[WUBU_DHT_ID_LEN];     /* a node id, when present */
    int      have_id;
    int      have_token;
    int      have_value;
} wubu_dht_decoded_t;

/* Decode a KRPC datagram.  On success returns 0 and fills `out`; EVERY
 * failure -- a non-bencode body, a missing id, a wrong-shaped value, trailing
 * bytes -- returns -1 and leaves `out` untouched, so nothing leaks why. */
int wubu_dht_decode(const uint8_t *buf, size_t n, wubu_dht_decoded_t *out);

#endif /* WUBU_DHT_H */
