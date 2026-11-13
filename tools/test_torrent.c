/*
 * test_torrent.c -- the info-hash must be over the .torrent's OWN `info` bytes.
 *
 * The vectors below were built and hashed with Python's hashlib, independently
 * of this implementation, so a wrong span cannot pass by agreeing with a bug:
 * each case asserts the exact byte range AND the digest of that range.
 */
#include "wubu_torrent.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

static const uint8_t kTorrentA[] = {
    100, 56, 58, 97, 110, 110, 111, 117, 110, 99, 101, 51, 58, 85, 82, 76,
    52, 58, 105, 110, 102, 111, 100, 52, 58, 110, 97, 109, 101, 52, 58, 116,
    101, 115, 116, 49, 50, 58, 112, 105, 101, 99, 101, 32, 108, 101, 110, 103,
    116, 104, 105, 49, 54, 51, 56, 52, 101, 54, 58, 112, 105, 101, 99, 101,
    115, 50, 48, 58, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97,
    97, 97, 97, 97, 97, 97, 97, 97, 101, 101,
};  /* 90 bytes */

static const uint8_t kTorrentB[] = {
    100, 56, 58, 97, 110, 110, 111, 117, 110, 99, 101, 51, 58, 85, 82, 76,
    55, 58, 99, 111, 109, 109, 101, 110, 116, 54, 58, 104, 101, 108, 108, 111,
    32, 52, 58, 105, 110, 102, 111, 100, 53, 58, 102, 105, 108, 101, 115, 108,
    100, 54, 58, 108, 101, 110, 103, 116, 104, 105, 49, 48, 50, 52, 101, 52,
    58, 112, 97, 116, 104, 108, 56, 58, 100, 105, 114, 47, 102, 105, 108, 101,
    101, 101, 101, 52, 58, 110, 97, 109, 101, 57, 58, 109, 117, 108, 116, 105,
    46, 98, 105, 110, 49, 50, 58, 112, 105, 101, 99, 101, 32, 108, 101, 110,
    103, 116, 104, 105, 51, 50, 55, 54, 56, 101, 54, 58, 112, 105, 101, 99,
    101, 115, 52, 48, 58, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98,
    98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 101, 101,
};  /* 175 bytes */

int main(void)
{
    printf("=== test_torrent ===\n");

    /* -- A: minimal single file -- */
    {
        size_t start = 0, end = 0;
        CHECK(wubu_torrent_info_span(kTorrentA, sizeof(kTorrentA),
                                     &start, &end) == 0,
              "the minimal torrent yields an info span");
        CHECK(start == 22 && end == 89,
              "the span is exactly the info dict and nothing else");
        CHECK(end - start == 67, "the span is the whole dict, not part of it");
        CHECK(wubu_bencode_validate(kTorrentA + start, end - start) == 0,
              "the span is itself one well-formed bencode value");

        char hex[41];
        CHECK(wubu_torrent_info_hash(kTorrentA, sizeof(kTorrentA), hex) == 0,
              "the info hash computes");
        CHECK(strcmp(hex, "b0d0f9cbeb6b6b687f6dbd04c6e88818054b6738") == 0,
              "the info hash matches the independent SHA-1 of that span");
    }

    /* -- B: multi-file, keys before info, nested list -- */
    {
        size_t start = 0, end = 0;
        CHECK(wubu_torrent_info_span(kTorrentB, sizeof(kTorrentB),
                                     &start, &end) == 0,
              "the multi-file torrent yields an info span");
        CHECK(start == 39 && end == 174,
              "info is found after other keys, not assumed at a fixed offset");
        CHECK(wubu_bencode_validate(kTorrentB + start, end - start) == 0,
              "the nested span is well formed");

        char hex[41];
        CHECK(wubu_torrent_info_hash(kTorrentB, sizeof(kTorrentB), hex) == 0,
              "the multi-file info hash computes");
        CHECK(strcmp(hex, "6b24ef57c284c9a92953e875de4a6ca93c06e6c1") == 0,
              "the multi-file info hash matches the independent SHA-1");
    }

    /* -- the same bytes must NOT hash the same as their container -- */
    {
        char whole[41], info[41];
        CHECK(wubu_torrent_info_hash(kTorrentA, sizeof(kTorrentA), info) == 0,
              "the info hash is available for comparison");
        /* Hashing the whole file must differ from hashing the info dict: if
           they ever agree, the span logic is broken. */
        extern int wubu_sha1(const void *, size_t, uint8_t[20]);
        extern int wubu_sha1_hex(const uint8_t[20], char[41]);
        uint8_t d[20];
        wubu_sha1(kTorrentA, sizeof(kTorrentA), d);
        wubu_sha1_hex(d, whole);
        CHECK(strcmp(whole, info) != 0,
              "the whole file hashes differently from its info dict");
    }

    /* -- fail closed -- */
    {
        char hex[41];
        size_t s = 0, e = 0;

        CHECK(wubu_torrent_info_hash(NULL, 0, hex) == -1, "NULL is refused");
        CHECK(wubu_torrent_info_hash(kTorrentA, sizeof(kTorrentA), NULL) == -1,
              "a NULL output buffer is refused");

        CHECK(wubu_torrent_info_span(kTorrentA, 22, &s, &e) == -1,
              "a truncated file is refused");
        CHECK(wubu_torrent_info_span(kTorrentA, sizeof(kTorrentA), NULL, &e) == -1,
              "a NULL start out-param is refused");

        /* Not a dictionary at the top level. */
        CHECK(wubu_torrent_info_span((const uint8_t *)"4:spam", 6, &s, &e) == -1,
              "a non-dictionary top level is refused");

        /* A dictionary with no info key. */
        CHECK(wubu_torrent_info_span((const uint8_t *)"d8:announce3:URLe", 18,
                                     &s, &e) == -1,
              "a dictionary without an info key is refused");

        /* Leading zero in a string length is non-canonical. */
        CHECK(wubu_torrent_info_span((const uint8_t *)"d04:infod4:namee", 17,
                                     &s, &e) == -1,
              "a leading-zero length prefix is refused");

        /* Trailing garbage after the outer dictionary. */
        {
            uint8_t junk[96];
            memcpy(junk, kTorrentA, sizeof(kTorrentA));
            junk[sizeof(kTorrentA)] = 'X';
            CHECK(wubu_bencode_validate(junk, sizeof(kTorrentA) + 1) == -1,
                  "trailing bytes after one value are refused");
        }

        /* An empty file and an empty info dict are both refused, not hashed:
           the first has nothing to find, the second names nothing. */
        CHECK(wubu_torrent_info_hash((const uint8_t *)"", 0, hex) == -1,
              "an empty file is refused");
        CHECK(wubu_torrent_info_hash((const uint8_t *)"d4:infode", 9, hex) == -1,
              "an empty info dict is refused");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
