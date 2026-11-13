/*
 * test_dht_table.c -- a routing table must remember who answered, keep the
 * closest ones in the right order, and forget the dead rather than the live.
 */
#include "wubu_dht_table.h"

#include <stdio.h>
#include <string.h>

static int failures, checks;
#define CHECK(c, m) do { checks++; if (!(c)) { printf("  FAIL: %s\n", m); failures++; } } while (0)

/* Build an id that shares exactly `prefix` leading bits with an all-zero id:
 * the first differing bit is global bit `prefix`. */
static void id_with_prefix(uint8_t id[WUBU_DHT_ID_LEN], int prefix, uint8_t tail)
{
    memset(id, 0, WUBU_DHT_ID_LEN);
    int byte = prefix / 8;
    int bit  = prefix % 8;               /* 0 = MSB of that byte */
    if (byte < WUBU_DHT_ID_LEN)
        id[byte] = (uint8_t)(0x80 >> bit);
    id[WUBU_DHT_ID_LEN - 1] |= tail;     /* distinguish siblings in the bucket */
}

int main(void)
{
    printf("=== test_dht_table ===\n");

    uint8_t mine[WUBU_DHT_ID_LEN] = {0};
    wubu_dht_table_t t;

    /* -- a table with no local id cannot place anything -- */
    wubu_dht_table_init(&t, NULL);
    {
        uint8_t id[WUBU_DHT_ID_LEN];
        id_with_prefix(id, 0, 0x10);
        CHECK(wubu_dht_table_add(&t, id, 1, 1, 100) == -1,
              "a table without our own id refuses to place anyone");
        CHECK(wubu_dht_table_count(&t) == 0, "nothing was stored");
    }

    wubu_dht_table_init(&t, mine);

    /* -- our own id and the zero id are never stored -- */
    {
        uint8_t id[WUBU_DHT_ID_LEN] = {0};
        id[19] = 0x80;
        CHECK(wubu_dht_table_add(&t, id, 1, 1, 100) == 0,
              "a normal peer is accepted");
        CHECK(wubu_dht_table_add(&t, mine, 2, 2, 100) == -1,
              "our own id is refused (asking ourselves is a loop)");
        { uint8_t zero[WUBU_DHT_ID_LEN] = {0}; CHECK(wubu_dht_table_add(&t, zero, 3, 3, 100) == -1,
              "the zero id is refused"); }
        CHECK(wubu_dht_table_count(&t) == 1, "only the real peer was stored");
    }

    /* -- adding the same id twice REFRESHES it rather than duplicating -- */
    {
        wubu_dht_table_t t2;
        wubu_dht_table_init(&t2, mine);
        uint8_t id[WUBU_DHT_ID_LEN];
        id_with_prefix(id, 0, 0x10);
        wubu_dht_table_add(&t2, id, 10, 1000, 500);
        wubu_dht_table_add(&t2, id, 20, 2000, 900);
        CHECK(wubu_dht_table_count(&t2) == 1,
              "re-adding an id does not create a second entry");
    }

    /* -- a full bucket evicts its STALEST member -- */
    {
        wubu_dht_table_t t3;
        wubu_dht_table_init(&t3, mine);

        /* 16 ids sharing exactly 152 leading bits with the zero id, so they
           all land in ONE bucket (which holds 8). */
        for (int v = 0; v < 16; v++) {
            uint8_t id[WUBU_DHT_ID_LEN];
            memset(id, 0, sizeof(id));
            id[19] = (uint8_t)(0x10 | v);      /* first diff at bit 152 */
            CHECK(wubu_dht_table_add(&t3, id, 1, (uint16_t)v, 1000 + v) == 0,
                  "a sibling id is accepted");
        }
        CHECK(wubu_dht_table_count(&t3) == WUBU_DHT_BUCKET_MAX,
              "a bucket holds at most 8");
        /* The first 8 (oldest timestamps) were pushed out by the newest 8. */
        uint8_t key[WUBU_DHT_ID_LEN];
        memset(key, 0, sizeof(key));
        key[19] = 0x10;                        /* v == 0: the stalest */
        uint8_t ids[16][WUBU_DHT_ID_LEN];
        int n = wubu_dht_table_closest(&t3, key, (uint8_t *)ids, 16);
        CHECK(n == WUBU_DHT_BUCKET_MAX, "eight peers are reachable");
        int found_oldest = 0;
        for (int i = 0; i < n; i++)
            if (memcmp(ids[i], key, WUBU_DHT_ID_LEN) == 0) found_oldest = 1;
        CHECK(!found_oldest,
              "the stalest entry was evicted, not the freshest");
    }

    /* -- closest() orders by distance to the TARGET -- */
    {
        wubu_dht_table_t t4;
        wubu_dht_table_init(&t4, mine);

        /* Distances from the target: 0x00..01 (nearest), 0x40..00,
           0xC0..00 (farthest).  Chosen so the ORDER is not just a matter of
           which high nibble differs -- it tests the whole comparison. */
        uint8_t target[WUBU_DHT_ID_LEN] = {0};
        target[0] = 0x80;
        uint8_t near[WUBU_DHT_ID_LEN] = {0};   /* matches the top bit    */
        near[0] = 0x80;
        near[19] = 0x01;                       /* distance 0x00..01       */
        uint8_t mid[WUBU_DHT_ID_LEN] = {0};
        mid[0] = 0xC0;                         /* distance 0x40..00       */
        uint8_t far[WUBU_DHT_ID_LEN] = {0};
        far[0] = 0x40;                         /* distance 0xC0..00       */

        wubu_dht_table_add(&t4, mid,  1, 1, 100);
        wubu_dht_table_add(&t4, far,  2, 2, 100);
        wubu_dht_table_add(&t4, near, 3, 3, 100);

        uint8_t ids[8][WUBU_DHT_ID_LEN];
        int n = wubu_dht_table_closest(&t4, target, (uint8_t *)ids, 8);
        CHECK(n == 3, "all three peers are returned");
        CHECK(memcmp(ids[0], near, WUBU_DHT_ID_LEN) == 0,
              "the nearest id to the target comes first");
        CHECK(memcmp(ids[1], mid, WUBU_DHT_ID_LEN) == 0,
              "the next-nearest follows");
        CHECK(memcmp(ids[2], far, WUBU_DHT_ID_LEN) == 0,
              "the farthest is last");

        /* Monotonic distances across the whole list. */
        int mono = 1;
        for (int i = 1; i < n; i++) {
            uint8_t d_prev[WUBU_DHT_ID_LEN], d_cur[WUBU_DHT_ID_LEN];
            wubu_dht_distance(target, ids[i - 1], d_prev);
            wubu_dht_distance(target, ids[i], d_cur);
            if (memcmp(d_prev, d_cur, WUBU_DHT_ID_LEN) > 0) mono = 0;
        }
        CHECK(mono, "distances are non-decreasing down the list");

        /* `max` is respected. */
        uint8_t few[2][WUBU_DHT_ID_LEN];
        int nf = wubu_dht_table_closest(&t4, target, (uint8_t *)few, 2);
        CHECK(nf == 2, "asking for two returns two");
        CHECK(memcmp(few[0], near, WUBU_DHT_ID_LEN) == 0,
              "the two closest are the nearest ones");
    }

    /* -- find_node answers with at most k -- */
    {
        wubu_dht_table_t t5;
        wubu_dht_table_init(&t5, mine);
        for (int v = 0; v < 20; v++) {
            uint8_t id[WUBU_DHT_ID_LEN];
            id_with_prefix(id, 4, (uint8_t)v);
            wubu_dht_table_add(&t5, id, 1, (uint16_t)v, 100);
        }
        uint8_t target[WUBU_DHT_ID_LEN] = {0};
        uint8_t ids[24][WUBU_DHT_ID_LEN];
        int n = wubu_dht_table_find_node(&t5, target, (uint8_t *)ids, 24);
        CHECK(n == WUBU_DHT_K, "find_node answers with exactly k candidates");
        (void)ids;
    }

    /* -- reaping drops the silent, not merely the old -- */
    {
        wubu_dht_table_t t6;
        wubu_dht_table_init(&t6, mine);
        for (int i = 0; i < 4; i++) {
            uint8_t id[WUBU_DHT_ID_LEN];
            id_with_prefix(id, 2, (uint8_t)i);
            wubu_dht_table_add(&t6, id, 1, (uint16_t)i, 1000);   /* seen early */
        }
        for (int i = 0; i < 4; i++) {
            uint8_t id[WUBU_DHT_ID_LEN];
            id_with_prefix(id, 6, (uint8_t)i);
            wubu_dht_table_add(&t6, id, 1, (uint16_t)(i + 4), 9000); /* recent */
        }
        CHECK(wubu_dht_table_count(&t6) == 8, "all eight are stored");

        int r = wubu_dht_table_reap(&t6, 10000, 2000);   /* window: 2000ms */
        CHECK(r == 4, "the four silent peers are reaped");
        CHECK(wubu_dht_table_count(&t6) == 4, "the live ones remain");
        CHECK(wubu_dht_table_reap(&t6, 10000, 2000) == 0,
              "a second reap finds nothing more");
    }

    /* -- bad arguments -- */
    {
        wubu_dht_table_t t7;
        wubu_dht_table_init(&t7, mine);
        uint8_t target[WUBU_DHT_ID_LEN] = {0};
        uint8_t ids[4][WUBU_DHT_ID_LEN];
        CHECK(wubu_dht_table_closest(&t7, target, (uint8_t *)ids, 0) == 0,
              "a zero max returns nothing");
        CHECK(wubu_dht_table_closest(&t7, NULL, (uint8_t *)ids, 4) == 0,
              "a NULL target returns nothing");
        CHECK(wubu_dht_table_add(&t7, NULL, 1, 1, 1) == -1,
              "a NULL id is refused");
    }

    printf("=== %d checks, %d failed ===\n", checks, failures);
    return failures ? 1 : 0;
}
