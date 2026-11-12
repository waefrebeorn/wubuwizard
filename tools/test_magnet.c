/*
 * test_magnet.c -- a magnet must address a real model and refuse everything
 * else.  Every assertion is about the shape a swarm peer relies on: the URI
 * round-trips, the info-hash is well formed, and a malformed URI leaks
 * nothing about why it was refused.
 */
#include "wubu_magnet.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

static const char *kBtih =
    "0123456789abcdef0123456789abcdef01234567";

int main(void)
{
    printf("=== test_magnet ===\n");

    /* -- the info-hash shape -- */
    CHECK(wubu_magnet_valid_btih(kBtih), "40 hex chars is a v1 info-hash");
    CHECK(!wubu_magnet_valid_btih("0123456789abcdef0123456789abcdef0123456"),
          "39 chars is not an info-hash");
    CHECK(!wubu_magnet_valid_btih("0123456789abcdef0123456789abcdef012345678"),
          "41 chars is not an info-hash");
    CHECK(!wubu_magnet_valid_btih("0123456789abcdef0123456789abcdef0123456g"),
          "a non-hex char is not an info-hash");
    CHECK(!wubu_magnet_valid_btih(NULL), "NULL is not an info-hash");
    CHECK(!wubu_magnet_valid_btih(""), "empty is not an info-hash");

    /* -- build -- */
    wubu_magnet_t m;
    memset(&m, 0, sizeof(m));
    memcpy(m.btih, kBtih, WUBU_MAGNET_BTIH_LEN + 1);

    char uri[1024];
    long n = wubu_magnet_build(&m, uri, sizeof(uri));
    CHECK(n > 0, "a bare magnet builds");
    CHECK(strcmp(uri, "magnet:?xt=urn:btih:" "0123456789abcdef0123456789abcdef01234567") == 0,
          "the bare magnet is exactly the BEP 9 form");

    snprintf(m.name, sizeof(m.name), "Qwen3.6-35B-A3B");
    n = wubu_magnet_build(&m, uri, sizeof(uri));
    CHECK(n > 0 && strstr(uri, "&dn=Qwen3.6-35B-A3B") != NULL,
          "the display name rides along");

    /* A name with a space and a slash must be percent-encoded, or the URI
       would parse as several parameters. */
    snprintf(m.name, sizeof(m.name), "My Model/1");
    n = wubu_magnet_build(&m, uri, sizeof(uri));
    CHECK(n > 0 && strstr(uri, "&dn=My%20Model%2F1") != NULL,
          "the display name is percent-encoded");
    CHECK(strchr(strstr(uri, "&dn=") + 4, ' ') == NULL,
          "no raw space survives in the URI");

    snprintf(m.trackers[0], sizeof(m.trackers[0]),
             "udp://tracker.example:1337/announce");
    m.tracker_count = 1;
    m.length = 69321221376ull;
    n = wubu_magnet_build(&m, uri, sizeof(uri));
    CHECK(n > 0 && strstr(uri, "&tr=udp://tracker.example:1337/announce") != NULL,
          "the tracker rides along");
    CHECK(strstr(uri, "&xl=69321221376") != NULL, "the exact length rides along");

    /* -- round trip -- */
    wubu_magnet_t p;
    memset(&p, 0, sizeof(p));
    CHECK(wubu_magnet_parse(uri, &p) == 0, "the URI parses");
    CHECK(strcmp(p.btih, kBtih) == 0, "the info-hash survives");
    CHECK(strcmp(p.name, "My Model/1") == 0,
          "the percent-encoded name decodes back to the original");
    CHECK(p.tracker_count == 1 &&
          strcmp(p.trackers[0], "udp://tracker.example:1337/announce") == 0,
          "the tracker survives");
    CHECK(p.length == 69321221376ull, "the exact length survives");

    /* -- fail closed -- */
    wubu_magnet_t untouched;
    memset(&untouched, 0xA5, sizeof(untouched));

    CHECK(wubu_magnet_parse("http://example/x", &untouched) == -1,
          "a non-magnet URI is refused");
    CHECK(wubu_magnet_parse("magnet:?dn=no-xt", &untouched) == -1,
          "a magnet with no xt is refused (it names nothing)");
    CHECK(wubu_magnet_parse("magnet:?xt=urn:sha1:abc", &untouched) == -1,
          "an unknown exact topic is refused");
    CHECK(wubu_magnet_parse("magnet:?xt=urn:btih:tooshort", &untouched) == -1,
          "a short info-hash is refused");
    CHECK(wubu_magnet_parse("magnet:?xt=urn:btih:0123456789abcdef0123456789abcdef01234567&dn=%ZZ",
                            &untouched) == -1,
          "a bad percent escape is refused");
    CHECK(wubu_magnet_parse("magnet:?xt=urn:btih:0123456789abcdef0123456789abcdef01234567&dn",
                            &untouched) == -1,
          "a parameter with no value is refused");
    CHECK(wubu_magnet_parse(NULL, &untouched) == -1, "NULL is refused");
    {
        int same = 1;
        for (size_t i = 0; i < sizeof(untouched); i++)
            if (((uint8_t *)&untouched)[i] != 0xA5) same = 0;
        CHECK(same, "a refused parse leaves the caller's struct untouched");
    }

    /* An over-long display name must be refused, not truncated: a truncated
       name is a different model's address. */
    {
        char big[600];
        memset(big, 'a', sizeof(big));
        big[sizeof(big) - 1] = '\0';
        char long_uri[700];
        snprintf(long_uri, sizeof(long_uri),
                 "magnet:?xt=urn:btih:%s&dn=%s", kBtih, big);
        CHECK(wubu_magnet_parse(long_uri, &untouched) == -1,
              "an over-long display name is refused");
    }

    /* -- unknown parameters are ignored (BEP 9 lets senders add them) -- */
    {
        char extra[512];
        snprintf(extra, sizeof(extra),
                 "magnet:?xt=urn:btih:%s&ws=http://example/&x=1", kBtih);
        wubu_magnet_t q;
        CHECK(wubu_magnet_parse(extra, &q) == 0,
              "unknown parameters do not break the parse");
        CHECK(strcmp(q.btih, kBtih) == 0, "the info-hash still parses");
    }

    /* -- the build side -- */
    {
        wubu_magnet_t bad = m;
        memcpy(bad.btih, "short", 6);
        CHECK(wubu_magnet_build(&bad, uri, sizeof(uri)) == -1,
              "a malformed info-hash does not build a magnet");
        CHECK(wubu_magnet_build(&m, uri, 8) == -1,
              "a buffer too small is refused");
        CHECK(wubu_magnet_build(NULL, uri, sizeof(uri)) == -1,
              "a NULL magnet is refused");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
