/*
 * test_dht.c -- identity must be derived, distance must be a metric, and the
 * KRPC messages must round-trip.  The negative cases matter more: a DHT that
 * accepts a malformed datagram will route on lies.
 */
#include "wubu_dht.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

static void seed_n(uint8_t *s, size_t n, uint8_t base)
{
    for (size_t i = 0; i < n; i++) s[i] = (uint8_t)(base + i);
}

int main(void)
{
    printf("=== test_dht ===\n");

    /* -- identity is DERIVED, deterministic, not typed in -- */
    {
        uint8_t id1[WUBU_DHT_ID_LEN], id2[WUBU_DHT_ID_LEN], id3[WUBU_DHT_ID_LEN];
        uint8_t sa[8], sb[8];
        seed_n(sa, sizeof(sa), 1);
        seed_n(sb, sizeof(sb), 2);

        CHECK(wubu_dht_node_id(sa, sizeof(sa), id1) == 0,
              "a node id derives from its seed");
        wubu_dht_node_id(sa, sizeof(sa), id2);
        CHECK(memcmp(id1, id2, WUBU_DHT_ID_LEN) == 0,
              "the same seed yields the same id (deterministic)");
        wubu_dht_node_id(sb, sizeof(sb), id3);
        CHECK(memcmp(id1, id3, WUBU_DHT_ID_LEN) != 0,
              "a different seed yields a different id");
        CHECK(wubu_dht_node_id(NULL, 0, id1) == 0,
              "an empty seed still derives an id");
        CHECK(wubu_dht_node_id(sa, sizeof(sa), NULL) == -1,
              "a NULL output is refused");
        CHECK(wubu_dht_node_id(NULL, 4, id1) == -1,
              "a non-zero length with NULL data is refused");
    }

    /* -- distance is XOR, so it is symmetric and zero only when equal -- */
    {
        uint8_t a[WUBU_DHT_ID_LEN], b[WUBU_DHT_ID_LEN];
        uint8_t dab[WUBU_DHT_ID_LEN], dba[WUBU_DHT_ID_LEN];
        memset(a, 0xA5, sizeof(a));
        memset(b, 0x0F, sizeof(b));

        wubu_dht_distance(a, b, dab);
        wubu_dht_distance(b, a, dba);
        CHECK(memcmp(dab, dba, WUBU_DHT_ID_LEN) == 0,
              "distance is symmetric");
        CHECK(dab[0] == (uint8_t)(0xA5 ^ 0x0F),
              "distance is byte-wise XOR");

        wubu_dht_distance(a, a, dab);
        uint8_t zeros[WUBU_DHT_ID_LEN] = {0};
        CHECK(memcmp(dab, zeros, WUBU_DHT_ID_LEN) == 0,
              "distance to itself is zero");
    }

    /* -- ordering by distance works against a known anchor -- */
    {
        uint8_t x[WUBU_DHT_ID_LEN] = {0};
        uint8_t near[WUBU_DHT_ID_LEN]  = {0};
        uint8_t far[WUBU_DHT_ID_LEN]   = {0};
        near[19] = 1;                       /* distance 1  */
        far[0]   = 0x80;                    /* distance 2^159 */

        CHECK(wubu_dht_distance_cmp(x, near, far) < 0,
              "the nearer id sorts first");
        CHECK(wubu_dht_distance_cmp(x, far, near) > 0,
              "swapping the candidates swaps the verdict");
        CHECK(wubu_dht_distance_cmp(x, near, near) == 0,
              "equal distances compare equal");
    }

    /* -- bucketing by shared prefix -- */
    {
        uint8_t a[WUBU_DHT_ID_LEN] = {0};
        uint8_t b[WUBU_DHT_ID_LEN] = {0};

        CHECK(wubu_dht_shared_prefix_bits(a, b) == 160,
              "identical ids share all 160 bits");

        b[0] = 0x80;
        CHECK(wubu_dht_shared_prefix_bits(a, b) == 0,
              "differing in the first bit shares 0 bits");

        b[0] = 0x01;
        CHECK(wubu_dht_shared_prefix_bits(a, b) == 7,
              "differing in the last bit of the first byte shares 7 bits");

        b[0] = 0x00;
        b[19] = 0x01;
        CHECK(wubu_dht_shared_prefix_bits(a, b) == 159,
              "differing in the final bit shares 159 bits");

        /* Monotonicity: moving the difference earlier can only reduce it. */

        int ok = 1;
        for (int bit = 0; bit < 160; bit++) {
            memset(b, 0, sizeof(b));
            b[bit / 8] = (uint8_t)(0x80 >> (bit % 8));
            int v = wubu_dht_shared_prefix_bits(a, b);
            if (v != bit) ok = 0;           /* differs exactly at `bit` */
        }
        CHECK(ok, "a single differing bit reports its exact position");
    }

    /* -- query names -- */
    {
        CHECK(strcmp(wubu_dht_query_info(WUBU_DHT_Q_PING)->name, "ping") == 0,
              "ping has a name");
        CHECK(strcmp(wubu_dht_query_by_name("announce_peer")->name,
                     "announce_peer") == 0,
              "a query can be found by name");
        CHECK(wubu_dht_query_by_name("no_such_query") == NULL,
              "an unknown query name is refused");
        CHECK(wubu_dht_query_info((wubu_dht_query_t)99) == NULL,
              "an unknown query number is refused");
    }

    /* -- a request round-trips -- */
    {
        wubu_dht_args_t args;
        memset(&args, 0, sizeof(args));
        memcpy(args.id, "\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a"
                        "\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14",
                WUBU_DHT_ID_LEN);
        args.have_txn = 1;
        memcpy(args.txn, "\xde\xad\xbe\xef", 4);
        memcpy(args.target, "\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a"
                            "\x1b\x1c\x1d\x1e\x1f\x20\x21\x22\x23\x24",
                WUBU_DHT_ID_LEN);
        args.have_target = 1;

        uint8_t buf[WUBU_DHT_MAX_MSG];
        long n = wubu_dht_encode_request(WUBU_DHT_Q_FIND_NODE, &args,
                                         buf, sizeof(buf));
        CHECK(n > 0, "a find_node request encodes");
        /* The BEP 5 shape: keys a, q, t, y in order, with y last. */
        CHECK(strstr((char *)buf, "1:y1:qe") != NULL,
              "the message declares itself a query");
        CHECK(strstr((char *)buf, "1:q9:find_node") != NULL,
              "the query name rides along");

        wubu_dht_decoded_t d;
        CHECK(wubu_dht_decode(buf, (size_t)n, &d) == 0,
              "the request decodes");
        CHECK(!d.is_response && !d.is_error, "it reads as a query");
        CHECK(d.have_id && memcmp(d.id, args.id, WUBU_DHT_ID_LEN) == 0,
              "the sender id survives");
        CHECK(d.have_token && memcmp(d.token, args.txn, 4) == 0,
              "the transaction id survives");
    }

    /* -- a response with a token round-trips -- */
    {
        uint8_t id[WUBU_DHT_ID_LEN], tok[WUBU_DHT_TOKEN_LEN];
        memset(id, 0x42, sizeof(id));
        memcpy(tok, "tok12345", WUBU_DHT_TOKEN_LEN);

        uint8_t buf[WUBU_DHT_MAX_MSG];
        long n = wubu_dht_encode_response(id, tok, 1, NULL, 0,
                                          buf, sizeof(buf));
        CHECK(n > 0, "a response encodes");
        CHECK(strstr((char *)buf, "1:y1:re") != NULL,
              "the message declares itself a response");

        wubu_dht_decoded_t d;
        CHECK(wubu_dht_decode(buf, (size_t)n, &d) == 0, "the response decodes");
        CHECK(d.is_response && !d.is_error, "it reads as a response");
        CHECK(d.have_id && memcmp(d.id, id, WUBU_DHT_ID_LEN) == 0,
              "the responder id survives");
        CHECK(d.have_token && memcmp(d.token, tok, WUBU_DHT_TOKEN_LEN) == 0,
              "the token comes back verbatim, which is what an announce needs");
    }

    /* -- an error response -- */
    {
        /* Hand-build a BEP 5 error: d1:eli201e4:msg!e1:t4:txid1:y1:ee */
        static const char *err =
            "d1:eli201e6:nope!!e1:t4:txid1:y1:ee";
        wubu_dht_decoded_t d;
        CHECK(wubu_dht_decode((const uint8_t *)err, strlen(err), &d) == 0,
              "an error datagram decodes");
        CHECK(d.is_response && d.is_error, "it reads as an error");
        CHECK(d.error_code == 201, "the error code is recovered");
    }

    /* -- fail closed -- */
    {
        uint8_t id[WUBU_DHT_ID_LEN] = {0};
        uint8_t buf[WUBU_DHT_MAX_MSG];
        wubu_dht_decoded_t untouched;
        memset(&untouched, 0xA5, sizeof(untouched));

        long n = wubu_dht_encode_response(id, NULL, 0, NULL, 0, buf, sizeof(buf));
        CHECK(n > 0, "a bare response is available for tampering");

        /* Trailing garbage. */
        uint8_t bad[WUBU_DHT_MAX_MSG];
        memcpy(bad, buf, (size_t)n);
        bad[n] = 'X';
        CHECK(wubu_dht_decode(bad, (size_t)n + 1, &untouched) == -1,
              "trailing bytes are refused");

        /* A non-bencode body. */
        CHECK(wubu_dht_decode((const uint8_t *)"not bencode", 11,
                              &untouched) == -1,
              "a non-bencode datagram is refused");

        /* Truncated. */
        CHECK(wubu_dht_decode(buf, (size_t)n - 1, &untouched) == -1,
              "a truncated datagram is refused");

        CHECK(wubu_dht_decode(NULL, 0, &untouched) == -1, "NULL is refused");

        /* An id that is not 20 bytes. */
        static const char *short_id =
            "d1:rd2:id3:abc e1:y1:re";      /* note: malformed on purpose */
        CHECK(wubu_dht_decode((const uint8_t *)short_id, strlen(short_id),
                              &untouched) == -1,
              "a short id is refused");

        {
            int same = 1;
            for (size_t i = 0; i < sizeof(untouched); i++)
                if (((uint8_t *)&untouched)[i] != 0xA5) same = 0;
            CHECK(same, "a refused decode leaks nothing");
        }

        /* A request without a transaction id cannot be matched later. */
        wubu_dht_args_t a2;
        memset(&a2, 0, sizeof(a2));
        memcpy(a2.id, id, WUBU_DHT_ID_LEN);
        CHECK(wubu_dht_encode_request(WUBU_DHT_Q_PING, &a2, buf,
                                      sizeof(buf)) == -1,
              "a request without a transaction id is refused");
        CHECK(wubu_dht_encode_request((wubu_dht_query_t)99, &a2, buf,
                                      sizeof(buf)) == -1,
              "an unknown query is refused");
        CHECK(wubu_dht_encode_request(WUBU_DHT_Q_PING, NULL, buf,
                                      sizeof(buf)) == -1,
              "NULL arguments are refused");
        CHECK(wubu_dht_encode_request(WUBU_DHT_Q_PING, &a2, buf, 4) == -1,
              "a buffer too small is refused");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
