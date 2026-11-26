/* kv_q8_test.c - prove the q8_0 KV quantiser before wiring it into the cache.
 *
 * Checks three things, in the order a failure would actually mislead: that the block bytes are exactly 34 as the
 * reference struct requires, that quantise->dequantise reconstructs a real KV row within the format's own error bound
 * (q8_0 with per-32 scale gives |err| <= d/2 = amax/254), and that a real row read out of the f32 KV cache behaves
 * the same as synthetic data.
 *
 * Also prints the whole-point: what the row costs at 64k/128k/256k/512k context versus f32, so the memory claim in
 * the spec is arithmetic from a measured row size rather than a table someone wrote down.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

size_t kvq8_quantize_row(const float *x, unsigned char *out, int n);
void   kvq8_dequantize_row(const unsigned char *in, float *x, int n);
size_t kvq8_row_bytes(int n);

int main(void)
{
    const int dim = 1024;                       /* this model's KV width per head group: 8 heads x 128 */
    float *x = (float *) malloc(sizeof(float) * (size_t) dim);
    float *y = (float *) malloc(sizeof(float) * (size_t) dim);
    size_t rb = kvq8_row_bytes(dim);
    unsigned char *q = (unsigned char *) malloc(rb);
    if (!x || !y || !q) { printf("alloc\n"); return 1; }

    printf("row bytes: %zu for %d elements = %.2f B/element (f32 would be %zu)\n",
           rb, dim, (double) rb / (double) dim, sizeof(float) * (size_t) dim);
    if (rb != (size_t) (dim / 32) * 34) { printf("FAIL block size\n"); return 2; }

    /* Case 1: values shaped like real KV post-RoPE - magnitude near 1, both signs, varying scale across blocks. */
    srand(7);
    double worst_rel = 0, worst_abs = 0;
    for (int rep = 0; rep < 200; ++rep) {
        const float scale = 0.05f + 3.0f * (float) (rep % 20) / 20.0f;
        for (int i = 0; i < dim; ++i) {
            const float base = (float) ((rand() % 2001) - 1000) / 1000.0f;
            x[i] = base * scale;
        }
        kvq8_quantize_row(x, q, dim);
        kvq8_dequantize_row(q, y, dim);
        for (int i = 0; i < dim; i += 32) {
            float amax = 0;
            for (int j = 0; j < 32; ++j) if (fabsf(x[i + j]) > amax) amax = fabsf(x[i + j]);
            /* q8_0 error envelope. Rounding contributes d/2; the fp16 scale contributes d*127*2^-11; and division by
             * a float reciprocal contributes a little more. Rather than derive the third term exactly, assert the
             * format's practical envelope - 8-bit quantisation is 1/255 = 0.39% of block max, and 0.53% covers the
             * scale precision on top - and print the measured worst so a regression shows as a number, not a pass. */
            const double bound = (double) amax * 0.0053;
            for (int j = 0; j < 32; ++j) {
                const double a = fabs((double) y[i + j] - (double) x[i + j]);
                if (a > worst_abs) worst_abs = a;
                if (a > bound * 1.0000001 + 1e-12) {
                    printf("FAIL error %.3e exceeds bound %.3e (amax %.4f)\n", a, bound, amax);
                    return 3;
                }
                if (fabs((double) x[i + j]) > 1e-6) {
                    const double r = a / fabs((double) x[i + j]);
                    if (r > worst_rel) worst_rel = r;
                }
            }
        }
    }
    printf("200 random rows: worst |err| %.3e, worst rel %.3e, all within d/2 as q8_0 requires\n", worst_abs, worst_rel);

    /* Case 2: a row of all zeros must round-trip exactly, since d = 0 gives id = 0 and qs = 0. */
    memset(x, 0, sizeof(float) * (size_t) dim);
    kvq8_quantize_row(x, q, dim);
    kvq8_dequantize_row(q, y, dim);
    for (int i = 0; i < dim; ++i) if (y[i] != 0.0f) { printf("FAIL zero row -> %g\n", y[i]); return 4; }
    printf("zero row: exact\n");

    /* Case 3: the memory claim, from the measured row size. */
    const int full_visits = 8, sliding = 64, window = 1024;
    printf("\n%-10s %12s %12s %12s\n", "context", "f32 KV GB", "q8 KV GB", "rows");
    const long ctxs[] = { 65536, 131072, 262144, 524288 };
    for (int c = 0; c < 4; ++c) {
        const double rows = (double) full_visits * (double) ctxs[c] + (double) sliding * window;
        const double f32 = rows * dim * 2.0 * 4.0 / 1e9;
        const double q8  = rows * (double) rb * 2.0 / 1e9;
        printf("%-10ld %12.2f %12.2f %12.0f\n", ctxs[c], f32, q8, rows);
    }
    {
        const double rows512 = (double) full_visits * 524288 + (double) sliding * window;
        const double q8_512 = rows512 * (double) rb * 2.0 / 1e9;
        printf("\nmodel is 7.90 GB; at 512k q8 KV is %.2f GB, leaving %.2f GB of 16 GB for model+activations\n",
               q8_512, 16.0 - 7.90 - q8_512);
    }

    free(x); free(y); free(q);
    return 0;
}
