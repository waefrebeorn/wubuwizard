/*
 * wubu_torrent.h -- read a .torrent well enough to name it.
 *
 * A BitTorrent v1 info-hash is the SHA-1 of the BENCODED INFO DICT of the
 * .torrent -- the exact byte span of the `info` value, not a digest of the
 * files and not a digest of some re-serialisation.  Get that wrong and the
 * magnet addresses nothing while looking like it works.
 *
 * So this module does exactly two things:
 *   - validate the .torrent's outer dictionary and locate the `info` value's
 *     byte span (wubu_torrent_info_span);
 *   - SHA-1 that span into a btih (wubu_torrent_info_hash).
 *
 * It deliberately does NOT re-encode the dictionary.  Hashing a re-encoded
 * copy produces a different hash unless the original was already canonical,
 * and a hash of something that isn't the file's own bytes is a lie.
 *
 * Fail-closed: a truncated file, a non-dictionary top level, a bad string
 * length, an unsorted/malformed dictionary or a missing `info` all return -1
 * and leave the caller's output untouched.  Every failure is the same -1, so
 * nothing leaks WHY.
 *
 * C11, libc only.
 */
#ifndef WUBU_TORRENT_H
#define WUBU_TORRENT_H

#include <stddef.h>
#include <stdint.h>

/* Locate the byte span [start,end) of the outer dictionary's `info` value.
 * Returns 0 on success, -1 on any malformed input. */
int wubu_torrent_info_span(const uint8_t *buf, size_t len,
                           size_t *start, size_t *end);

/* The v1 info-hash of the .torrent: SHA-1 over the `info` span, as 40
 * lowercase hex chars plus nul.  Returns 0, or -1 on a malformed file, a
 * NULL/short output buffer, or an internal error. */
int wubu_torrent_info_hash(const uint8_t *buf, size_t len, char out[41]);

/* Is the span a structurally valid bencode value?  Used by the test to prove
 * the span we hashed really is the info dict and not a mis-offset slice. */
int wubu_bencode_validate(const uint8_t *buf, size_t len);

#endif /* WUBU_TORRENT_H */
