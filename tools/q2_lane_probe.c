/* Probe the lane order of the Q2_0 kernel's 2-bit unpacking.
 *
 * WHY: the reference test for ty=42 must pair each weight with the SAME activation index the kernel pairs it with. The dot
 * is commutative, so a whole-row sum cannot reveal the pairing; a reference written with the wrong order would report a
 * mismatch that is not a kernel bug. This probe determines the mapping empirically instead of assuming it.
 *
 * METHOD: one block, d = 1.0. Exactly one 2-bit field is set to q=3 (weight +2); every other field is q=1 (weight 0).
 * The activation is a one-hot at index k (a one-hot survives Q8_0 quantisation exactly). The kernel output is therefore
 * 2.0 when the probed field pairs with k, and 0.0 otherwise. Scanning k for each (byte, shift) yields the permutation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "wubu_iq2q8.h"
#include "wubu_iq_c11.h"

void wubu_iq_rows(int ggml_type, const void *w, size_t row_bytes, int n,
                  const void *const *act, int nt, float *const *out, int r0, int r1);
double in2_hires_now(void) { return 0.0; }
double in2_ty_sec[256]; long in2_ty_calls[256]; int in2_ty_par = 0; int in2_ty_depth = 0;

#define QK2 64
typedef struct { uint16_t d; uint8_t qs[16]; } blk42;

int main(void)
{
blk42 w; float x[QK2]; in2_q8_0_blk aq[2]; float out = 0.0f;
    const size_t rb = sizeof(blk42);

    printf("Q2_0 lane probe: sizeof(blk42)=%zu  (expect 18 = 2 + 16)\n", rb);
    printf("  field(byte,shift) -> activation index\n");
    int perm[16][4];
    for (int b = 0; b < 16; ++b)
        for (int s = 0; s < 4; ++s) {
            /* weights: all q=1 (weight 0) except this one field, q=3 (weight +2) */
            memset(&w, 0, sizeof w);
            w.d = in2_f32_to_f16(1.0f);
            for (int by = 0; by < 16; ++by) w.qs[by] = (uint8_t) (0x55u << 0); /* q=1 in all four fields: 0b01010101 */
            w.qs[b] = (uint8_t) ((w.qs[b] & ~(3u << (2 * s))) | (3u << (2 * s)));

            int found = -1;
            for (int k = 0; k < QK2 && found < 0; ++k) {
                for (int i = 0; i < QK2; ++i) x[i] = 0.0f;
                x[k] = 1.0f;
                in2_q8_0_quantise(x, 2, aq);
                const void *actp[1] = { aq }; float *op[1] = { &out };
                out = 0.0f;
                wubu_iq_rows(42, &w, rb, QK2, actp, 1, op, 0, 1);
                if (fabsf(out - 2.0f) < 1e-3f) found = k;
            }
            perm[b][s] = found;
        }

    for (int b = 0; b < 16; ++b) {
        printf("  byte %2d: ", b);
        for (int s = 0; s < 4; ++s) printf("[%2d] ", perm[b][s]);
        printf("\n");
    }
    /* summary: is it plain sequential (byte j, shift s -> 4j+s)? */
    int seq = 1, iso = 1; int seen[QK2]; memset(seen, 0, sizeof seen);
    for (int b = 0; b < 16; ++b) for (int s = 0; s < 4; ++s) {
        int v = perm[b][s];
        if (v < 0) { iso = 0; continue; }
        if (seen[v]++) iso = 0;              /* not injective => something is off */
        if (v != 4 * b + s) seq = 0;
    }
    printf("  sequential(4*byte+shift)=%s   injective=%s\n", seq ? "YES" : "NO", iso ? "YES" : "NO");
    return 0;
}
