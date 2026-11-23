/* test_eg27_isq.c -- EG-27 gate: ISQ before device-load planner. */
#include <stdio.h>

#include "wubu_isq_plan.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

#define RAW  (100u << 20)
#define QUANT (25u << 20)

int main(void) {
    wubu_isq_decision_t a, b, c, bad;

    a = wubu_isq_decide(RAW, QUANT, 50.0, 10.0, 2e9, 200u << 20);
    printf("EG-27 A isq=%d host=%.1fms dev=%.1fms bytes=%u\n",
           a.isq, a.host_total_ms, a.device_total_ms, (unsigned)a.bytes_to_device);
    CHECK(a.isq == 0 && a.bytes_to_device == RAW);
    CHECK(a.host_total_ms > a.device_total_ms);

    b = wubu_isq_decide(RAW, QUANT, 5.0, 10.0, 2e9, 200u << 20);
    printf("EG-27 B isq=%d host=%.1fms dev=%.1fms bytes=%u\n",
           b.isq, b.host_total_ms, b.device_total_ms, (unsigned)b.bytes_to_device);
    CHECK(b.isq == 1 && b.bytes_to_device == QUANT);
    CHECK(b.host_total_ms < b.device_total_ms);

    c = wubu_isq_decide(RAW, QUANT, 500.0, 10.0, 2e9, 50u << 20);
    printf("EG-27 C forced=%d isq=%d bytes=%u\n",
           c.forced_by_budget, c.isq, (unsigned)c.bytes_to_device);
    CHECK(c.forced_by_budget == 1 && c.isq == 1);
    CHECK(c.bytes_to_device == QUANT);

    bad = wubu_isq_decide(RAW, QUANT, 1.0, 1.0, 0.0, 200u << 20);
    CHECK(bad.isq == 0 && bad.bytes_to_device == RAW);

    if (failures) {
        fprintf(stderr, "test_eg27_isq: FAIL (%d)\n", failures);
        return 1;
    }
    printf("test_eg27_isq: PASS\n");
    return 0;
}
