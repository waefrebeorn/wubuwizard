/*
 * wubu_bitswap.h -- bitswap message shapes (BIP? / go-bitswap schema).
 *
 * bitswap is how a libp2p peer gets BLOCKS once it knows WHAT it wants: a
 * WantList says "I want these CIDs", a Block carries the bytes, a
 * BlockPresence answers "I have it" / "I don't".  The messages are protobuf.
 *
 * This module encodes and decodes them.  It does not carry them: producing
 * bytes and validating structure is its whole job, and the socket belongs to
 * the caller -- the same split every other leg uses.
 *
 * Fail-closed: a truncated message, an unknown wire type, a length that runs
 * past the end, a field repeated where it may not be, or a trailing byte after
 * a complete message is refused with -1 and leaves the caller's output
 * untouched.  Nothing guesses at a meaning.
 *
 * C11, libc only.
 */
#ifndef WUBU_BITSWAP_H
#define WUBU_BITSWAP_H

#include <stddef.h>
#include <stdint.h>

#include "wubu_cid.h"          /* a CID marshals to raw bytes for the wire */

/* go-bitswap enums */
typedef enum { WUBU_BITSWAP_HAVE = 0, WUBU_BITSWAP_DONT_HAVE = 1 }
    wubu_bitswap_presence_t;

typedef enum { WUBU_BITSWAP_WANT_BLOCK = 0, WUBU_BITSWAP_WANT_HAVE = 1 }
    wubu_bitswap_want_t;

#define WUBU_BITSWAP_MAX_ENTRIES   64
#define WUBU_BITSWAP_MAX_CID       128
#define WUBU_BITSWAP_MAX_BLOCK     (1 << 20)   /* 1 MiB: a bitswap block */
#define WUBU_BITSWAP_MAX_PREFIX    64
#define WUBU_BITSWAP_MAX_MESSAGE   (1 << 21)

typedef struct {
    uint8_t cid[WUBU_BITSWAP_MAX_CID];    /* the marshalled CID */
    size_t  cid_len;
    int32_t priority;                     /* negative means "don't send" */
    int     cancel;
    int     want_type;                    /* wubu_bitswap_want_t */
    int     send_dont_have;
} wubu_bitswap_entry_t;

typedef struct {
    wubu_bitswap_entry_t entries[WUBU_BITSWAP_MAX_ENTRIES];
    int count;
    int full;                             /* whether this is the whole list */
} wubu_bitswap_wantlist_t;

typedef struct {
    uint8_t prefix[WUBU_BITSWAP_MAX_PREFIX];
    size_t  prefix_len;
    uint8_t *data;                        /* owned by the caller */
    size_t   data_len;
} wubu_bitswap_block_t;

typedef struct {
    uint8_t cid[WUBU_BITSWAP_MAX_CID];
    size_t  cid_len;
    int     presence;                     /* wubu_bitswap_presence_t */
} wubu_bitswap_block_presence_t;

/* A decoded message: whatever the wire carried. */
typedef struct {
    wubu_bitswap_wantlist_t wantlist;
    int have_wantlist;

    wubu_bitswap_block_t blocks[WUBU_BITSWAP_MAX_ENTRIES];
    int block_count;

    wubu_bitswap_block_presence_t presences[WUBU_BITSWAP_MAX_ENTRIES];
    int presence_count;
} wubu_bitswap_message_t;

/* ---- protobuf primitives, public because other legs reuse them ------ */

/* Field wire types from the protobuf spec. */
enum { WUBU_PB_VARINT = 0, WUBU_PB_64BIT = 1, WUBU_PB_BYTES = 2,
       WUBU_PB_32BIT = 5 };

long wubu_pb_tag(uint8_t *out, size_t n, unsigned field, int wire);
long wubu_pb_varint(uint8_t *out, size_t n, uint64_t v);

/* Write a complete field: tag + value.  Returns bytes written or -1. */
long wubu_pb_varint_field(uint8_t *out, size_t n, unsigned field, uint64_t v);
long wubu_pb_bytes_field(uint8_t *out, size_t n, unsigned field,
                         const uint8_t *data, size_t len);

/* Read one field header from `in` at `*pos`.
 *  - for a VARINT, *value receives it;
 *  - for BYTES, the span pointers point into `in`;
 *  - returns 0, or -1 on a truncated or malformed header.
 * Other wire types are refused: bitswap never uses them. */
int wubu_pb_read_header(const uint8_t *in, size_t n, size_t *pos,
                        unsigned *field, int *wire, uint64_t *value,
                        const uint8_t **bytes, size_t *len);

/* ---- bitswap messages --------------------------------------------- */

long wubu_bitswap_encode_entry(const wubu_bitswap_entry_t *e,
                               uint8_t *out, size_t n);
long wubu_bitswap_encode_wantlist(const wubu_bitswap_wantlist_t *wl,
                                  uint8_t *out, size_t n);
long wubu_bitswap_encode_block(const wubu_bitswap_block_t *b,
                               uint8_t *out, size_t n);
long wubu_bitswap_encode_presence(const wubu_bitswap_block_presence_t *p,
                                  uint8_t *out, size_t n);

/* Decode any bitswap message.  Returns 0 and fills `out`, or -1. */
int wubu_bitswap_decode(const uint8_t *buf, size_t n,
                        wubu_bitswap_message_t *out);

/* Encode a complete Message: wantlist (may be NULL), blocks and presences.
 * This is the shape that goes on the wire -- the parts above are its pieces.
 * Returns bytes written, or -1. */
long wubu_bitswap_encode_message(const wubu_bitswap_wantlist_t *wl,
                                 const wubu_bitswap_block_t *blocks,
                                 int block_count,
                                 const wubu_bitswap_block_presence_t *presences,
                                 int presence_count,
                                 uint8_t *out, size_t n);

#endif /* WUBU_BITSWAP_H */
