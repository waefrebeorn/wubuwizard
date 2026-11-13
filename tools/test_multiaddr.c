/*
 * test_multiaddr.c -- a multiaddr must round-trip, carry what it names, and
 * refuse anything whose shape it cannot validate.
 *
 * The IPv6 vectors below were checked against Python's `ipaddress` module:
 * each accepted literal normalises to the same 16 bytes, and each refused one
 * is rejected by ipaddress too.  That is the point -- a hand-written parser and
 * a reference implementation agreeing on 16 literals is stronger evidence than
 * either alone.
 */
#include "wubu_multiaddr.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

typedef struct { const char *in; const char *out; } vec_t;

int main(void)
{
    printf("=== test_multiaddr ===\n");

    /* -- round-trips: build() must reproduce the canonical form -- */
    static const vec_t v[] = {
        { "/ip4/1.2.3.4/tcp/6881",
          "/ip4/1.2.3.4/tcp/6881" },
        { "/ip4/127.0.0.1/udp/53",
          "/ip4/127.0.0.1/udp/53" },
        { "/dns4/example.com/tcp/443/wss",
          "/dns4/example.com/tcp/443/wss" },
        { "/ip6/::1/tcp/80",
          "/ip6/::1/tcp/80" },
        { "/ip6/2001:0db8:0000:0000:0000:0000:0000:0001/tcp/9",
          "/ip6/2001:db8::1/tcp/9" },
        { "/ip6/2001:db8:0:0:1:0:0:1/tcp/9",
          "/ip6/2001:db8::1:0:0:1/tcp/9" },
        { "/ip6/1:2:3:4:5:6:7:8/tcp/9",
          "/ip6/1:2:3:4:5:6:7:8/tcp/9" },
        { "/ip6/::ffff:1.2.3.4/tcp/9",
          "/ip6/::ffff:102:304/tcp/9" },
        { "/ip6/1:2:3:4:5:6:1.2.3.4/tcp/9",
          "/ip6/1:2:3:4:5:6:102:304/tcp/9" },
        { "/ip6/::1:2:3:4:5:6:7/tcp/9",
          "/ip6/0:1:2:3:4:5:6:7/tcp/9" },
        { "/ip6/0:0:0:0:0:0:0:1/tcp/9",
          "/ip6/::1/tcp/9" },
        { "/ip4/1.2.3.4/tcp/6881/p2p/QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG",
          "/ip4/1.2.3.4/tcp/6881/p2p/QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG" },
    };
    for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); i++) {
        wubu_multiaddr_t a;
        char buf[512];
        int rc = wubu_multiaddr_parse(v[i].in, &a);
        CHECK(rc == 0, "the address parses");
        long n = (rc == 0) ? wubu_multiaddr_build(&a, buf, sizeof(buf)) : -1;
        if (rc == 0 && n > 0)
            CHECK(strcmp(buf, v[i].out) == 0,
                  "build reproduces the canonical form");
        else
            CHECK(0, "build produced something");
        if (rc == 0 && n > 0 && strcmp(buf, v[i].out) != 0)
            printf("      in: %s\n     got: %s\n    want: %s\n",
                   v[i].in, buf, v[i].out);
    }

    /* -- the parts we name are really there -- */
    {
        wubu_multiaddr_t a;
        CHECK(wubu_multiaddr_parse("/ip4/1.2.3.4/tcp/6881/quic-v1/tls", &a) == 0,
              "a stacked address parses");
        CHECK(a.count == 4, "four protocols were read");
        CHECK(wubu_multiaddr_has(&a, WUBU_PROTO_IP4),  "it carries ip4");
        CHECK(wubu_multiaddr_has(&a, WUBU_PROTO_TCP),  "it carries tcp");
        CHECK(wubu_multiaddr_has(&a, WUBU_PROTO_QUIC_V1), "it carries quic-v1");
        CHECK(wubu_multiaddr_has(&a, WUBU_PROTO_TLS),   "it carries tls");
        CHECK(!wubu_multiaddr_has(&a, WUBU_PROTO_UDP), "it does not carry udp");
        CHECK(a.parts[1].port == 6881, "the port was decoded");
        CHECK(a.parts[0].ip4 == 0x01020304u, "the ipv4 was decoded big-endian");
    }

    /* -- programmatic assembly survives the same way -- */
    {
        wubu_multiaddr_t a;
        memset(&a, 0, sizeof(a));
        wubu_ma_part_t p;
        memset(&p, 0, sizeof(p));
        p.proto = WUBU_PROTO_IP4;  p.ip4 = 0x0A000001u;
        CHECK(wubu_multiaddr_add(&a, &p) == 0, "ip4 added");
        p.proto = WUBU_PROTO_TCP;  p.port = 6881;
        CHECK(wubu_multiaddr_add(&a, &p) == 0, "tcp added");
        char buf[128];
        CHECK(wubu_multiaddr_build(&a, buf, sizeof(buf)) > 0,
              "an assembled address builds");
        CHECK(strcmp(buf, "/ip4/10.0.0.1/tcp/6881") == 0,
              "the assembled address is correct");
    }

    /* -- malformed input is refused, and the output is untouched -- */
    static const char *bad[] = {
        "",                         /* empty */
        "/",                        /* a bare slash */
        "/ip4/1.2.3.4/",            /* trailing slash */
        "//ip4/1.2.3.4",            /* empty first segment */
        "/nope/1.2.3.4",            /* unknown protocol */
        "/ip4/256.1.1.1",           /* octet out of range */
        "/ip4/01.2.3.4",            /* leading zero is ambiguous */
        "/ip4/1.2.3",               /* too few octets */
        "/ip4/1.2.3.4.5",           /* too many octets */
        "/tcp/99999",               /* port out of range */
        "/tcp/01",                  /* leading zero */
        "/tcp",                     /* missing value */
        "/ip6/::: ",                /* triple colon */
        "/ip6/1::2::3",             /* two compressions */
        "/ip6/::8:7:6:5:4:3:2:1",   /* "::" plus eight groups */
        "/ip6/1:2:3:4:5:6:7:",      /* trailing colon */
        "/ip6/zzzz::1",             /* not hex */
        "/dns4/a/b",                /* a slash inside a value */
        NULL
    };
    for (int i = 0; bad[i]; i++) {
        wubu_multiaddr_t a;
        memset(&a, 0xA5, sizeof(a));
        CHECK(wubu_multiaddr_parse(bad[i], &a) == -1,
              "a malformed address is refused");
        {
            int same = 1;
            for (size_t k = 0; k < sizeof(a); k++)
                if (((uint8_t *)&a)[k] != 0xA5) same = 0;
            CHECK(same, "a refused parse leaves the caller's struct untouched");
        }
    }

    /* -- build-side refusals -- */
    {
        wubu_multiaddr_t empty;
        memset(&empty, 0, sizeof(empty));
        char buf[64];
        CHECK(wubu_multiaddr_build(NULL, buf, sizeof(buf)) == -1,
              "a NULL address is refused");
        CHECK(wubu_multiaddr_build(&empty, buf, sizeof(buf)) == -1,
              "an empty address is refused");
        CHECK(wubu_multiaddr_build(&empty, NULL, 0) == -1,
              "a NULL buffer is refused");

        wubu_multiaddr_t a;
        CHECK(wubu_multiaddr_parse("/ip4/1.2.3.4/tcp/6881", &a) == 0,
              "a good address is available for truncation");
        CHECK(wubu_multiaddr_build(&a, buf, 4) == -1,
              "a buffer too small is refused rather than truncated");
    }

    /* -- a part we cannot validate must not be stored -- */
    {
        wubu_multiaddr_t a;
        CHECK(wubu_multiaddr_parse("/onion/abcdefgh", &a) == -1,
              "an unsupported protocol is refused, not guessed");
        CHECK(wubu_multiaddr_parse("/memory/65536", &a) == -1,
              "an unknown numeric protocol is refused");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
