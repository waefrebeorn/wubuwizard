/*
 * wubu_multistream.h -- multistream-select protocol negotiation.
 *
 * Before any bitswap message travels, both peers must agree on WHICH protocol
 * the connection speaks.  multistream-select does that by exchanging
 * length-prefixed strings:
 *
 *     <varint length><bytes>      where length counts the bytes INCLUDING \n
 *     /multistream/1.0.0\n        the header every side sends first
 *     /ipfs/id/1.0.0\n            a protocol proposal
 *     na\n                        the answer when a proposal is unknown
 *
 * This module frames and reads those messages and picks a protocol.  It does
 * not open connections: framing and choice are its whole job, and the socket
 * belongs to the caller.
 *
 * Fail-closed: a frame whose length runs past the end, a header that is not
 * byte-identical, a proposal that does not start with '/', a string containing
 * a NUL, or a 'na' that is not exactly three bytes is refused with -1 and
 * leaves the caller's output untouched.
 *
 * C11, libc only.
 */
#ifndef WUBU_MULTISTREAM_H
#define WUBU_MULTISTREAM_H

#include <stddef.h>
#include <stdint.h>

#define WUBU_MS_HEADER          "/multistream/1.0.0\n"
#define WUBU_MS_HEADER_LEN      19
#define WUBU_MS_NA              "na\n"
#define WUBU_MS_NA_LEN          3
#define WUBU_MS_MAX_PROTO       256
#define WUBU_MS_MAX_FRAME       (1 << 20)

/* Encode a protocol proposal such as "/ipfs/bitswap/1.2.0\n".  The newline is
 * appended for you if missing.  Returns bytes written or -1. */
long wubu_ms_encode(const char *proto, uint8_t *out, size_t n);

/* Encode the 'na' refusal. */
long wubu_ms_encode_na(uint8_t *out, size_t n);

/* Encode an opaque data frame. */
long wubu_ms_encode_data(const uint8_t *data, size_t len, uint8_t *out,
                         size_t n);

/* Encode the multistream header. */
long wubu_ms_encode_header(uint8_t *out, size_t n);

enum {
    WUBU_MS_STRING = 1,   /* *str holds the message, NUL-terminated */
    WUBU_MS_DATA   = 2    /* data/len hold the payload */
};

/* Read ONE framed message from buf[*pos..n).
 *  - returns WUBU_MS_STRING and sets *str (into a static NUL-terminated copy
 *    held in *scratch when scratch is non-NULL), or
  *  - returns WUBU_MS_DATA with the span pointers into buf, or
 *  - returns -1 on a malformed frame.
 * Advances *pos past the frame. */
int wubu_ms_read(const uint8_t *buf, size_t n, size_t *pos,
                 const char **str, char *scratch, size_t scratch_len,
                 const uint8_t **data, size_t *len);

/* Convenience: read a single string message, refusing data frames. */
int wubu_ms_read_string(const uint8_t *buf, size_t n, size_t *pos,
                        const char **str, char *scratch, size_t scratch_len);

/* True when s is a well-formed multistream header. */
int wubu_ms_is_header(const char *s, size_t len);

/* Choose among wanted protocols.  incoming is the peer's proposal.  Returns
 * the index of the first wanted match, or -1 when none matches (the caller
 * then answers 'na'). */
int wubu_ms_select(const char *const *wanted, int nwanted,
                   const char *incoming, size_t ilen);

#endif /* WUBU_MULTISTREAM_H */
