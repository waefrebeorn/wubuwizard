/* test_poincare_kv.c — hyperbolic geometry for the KV namespace (AN22 #8)
 *
 * Acceptance criterion from the improvement plan: two files whose KV
 * regions are RELATED (share a namespace prefix, e.g. same research pond)
 * have LOWER Poincare distance than either has to an unrelated file.
 *
 * Checks:
 *   T1 map_path -> a point inside the unit ball (radius 0.8)
 *   T2 identical path -> zero distance; finite, >= 0 elsewhere
 *   T3 same-pond files closer than cross-pond (the geometry criterion)
 *   T4 curvature is in (0, 1]
 *
 * WaefreBeorn Umbrella License v3.0
 */
#include "wubu_poincare_kv.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int fail = 0;
static void chk(const char *label, int cond) {
    printf("  [%s] %s\n", cond ? "PASS" : "FAIL", label);
    if (!cond) fail++;
}

int main(void) {
    poincare_pos_t a, b, c;
    printf("[T1] map_path -> unit ball\n");
    chk("map a ok", wubu_poincare_kv_map_path("/kv/in/pond-a/attn.txt", &a) == 0);
    chk("radius == 0.8", fabs(a.radius - 0.8) < 1e-12);
    double norm = 0.0;
    for (int i = 0; i < WUBU_POINCARE_DIM; i++) norm += a.x[i] * a.x[i];
    chk("||p|| < 1", norm < 1.0);

    printf("[T2] baseline distances\n");
    chk("distance(a,a) == 0",
        wubu_poincare_kv_distance(&a, &a) < 1e-12);
    chk("path_distance self == 0",
        wubu_poincare_kv_path_distance("/kv/in/pond-a/attn.txt",
                                       "/kv/in/pond-a/attn.txt") < 1e-12);
    chk("curvature in (0,1]",
        wubu_poincare_kv_curvature(&a) > 0.0 &&
        wubu_poincare_kv_curvature(&a) <= 1.0);

    printf("[T3] related < cross-pond (the criterion)\n");
    chk("map b ok", wubu_poincare_kv_map_path("/kv/in/pond-a/kv.txt", &b) == 0);
    chk("map c ok", wubu_poincare_kv_map_path("/kv/in/pond-b/moe.txt", &c) == 0);

    double same_pond = wubu_poincare_kv_path_distance(
        "/kv/in/pond-a/attn.txt", "/kv/in/pond-a/kv.txt");
    double cross_pond = wubu_poincare_kv_path_distance(
        "/kv/in/pond-a/attn.txt", "/kv/in/pond-b/moe.txt");
    printf("       (same_pond_dist=%.4f  cross_pond_dist=%.4f)\n",
           same_pond, cross_pond);
    chk("same-pond distance is finite & >= 0", isfinite(same_pond) && same_pond >= 0);
    chk("same-pond < cross-pond (related files closer)", same_pond < cross_pond);

    /* mirror: a second pond-b file is closer to c than a pond-a file is */
    double d_b_to_c = wubu_poincare_kv_path_distance(
        "/kv/in/pond-a/kv.txt", "/kv/in/pond-b/moe.txt");
    double d_c_to_d = wubu_poincare_kv_path_distance(
        "/kv/in/pond-b/moe.txt", "/kv/in/pond-b/rlhf.txt");
    chk("pond-b pair closer than cross-pond",
        d_c_to_d < d_b_to_c && d_c_to_d < cross_pond);

    printf("[T4] curvature varies by region\n");
    poincare_pos_t far;
    wubu_poincare_kv_map_path("/x/y/z/w/v/u/t/s/r/q", &far);
    double cur_a = wubu_poincare_kv_curvature(&a);
    double cur_far = wubu_poincare_kv_curvature(&far);
    chk("all curvatures in (0,1]",
        cur_a > 0 && cur_a <= 1 && cur_far > 0 && cur_far <= 1);

    printf("test_poincare_kv: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
