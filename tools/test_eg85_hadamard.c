/* test_eg85_hadamard.c -- EG-85 gate: stateless Sylvester-Hadamard rotation.
 * Gates the existing src/wubu_hadamard.c module (no transform weight stored). */
#include <stdio.h>
#include <math.h>

#include "wubu_hadamard.h"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); failures++; \
} } while (0)

int main(void) {
    float in[8], once[8], twice[8], bad[8];
    double ein = 0.0, eout = 0.0, maxerr = 0.0;
    for (int i = 0; i < 8; ++i) in[i] = 0.25f * (float)((i * 3 + 1) % 7);

    /* Normalized Hadamard is its own inverse with no stored transform weight. */
    for (int i = 0; i < 8; ++i) once[i] = in[i];
    wubu_hadamard_fwd(once, 8);
    for (int i = 0; i < 8; ++i) twice[i] = once[i];
    wubu_hadamard_fwd(twice, 8);
    for (int i = 0; i < 8; ++i) {
        double e = fabs(twice[i] - in[i]);
        if (e > maxerr) maxerr = e;
        ein += (double)in[i] * in[i];
        eout += (double)once[i] * once[i];
    }
    CHECK(maxerr < 1e-5);
    CHECK(fabs(ein - eout) < 1e-5);       /* Parseval / orthogonality */
    printf("EG-85 n=8 involution maxerr=%.8f energy %.6f -> %.6f\n", maxerr, ein, eout);
    /* Non-power-of-two widths are refused by the power-of-two guard. */
    CHECK(wubu_is_pow2(8) == 1);
    CHECK(wubu_is_pow2(7) == 0);
    for (int i = 0; i < 8; ++i) bad[i] = in[i];
    wubu_hadamard_fwd(bad, 7);
    for (int i = 0; i < 8; ++i) CHECK(bad[i] == in[i]);
    wubu_hadamard_fwd(NULL, 8);           /* no-op on bad pointer */

    if (failures) { fprintf(stderr, "test_eg85_hadamard: FAIL (%d)\n", failures); return 1; }
    printf("test_eg85_hadamard: PASS\n");
    return 0;
}
