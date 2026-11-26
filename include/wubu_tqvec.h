/* wubu_tqvec.h - TurboQuant vector quantiser (data-oblivious, near-optimal distortion).
 *
 * Distinct from wubu_turboquant.h, which is a KV-cache tile manager despite the name.
 *
 * Typical use:
 *     wubu_tqvec *tq = wubu_tqvec_init(32, 3, 0xC0FFEEull);
 *     size_t nb = wubu_tqvec_code_bytes(tq);          // 12 bytes at d=32,b=3 ... plus nothing else
 *     unsigned char *codes = malloc(nb * n_items);
 *     for (i) wubu_tqvec_encode(tq, unit_vec[i], codes + i*nb);
 *     for (i) s[i] = wubu_tqvec_score(tq, query, codes + i*nb);   // scan
 *     ... take top-100, re-score those exactly ...
 */
#ifndef WUBU_TQVEC_H
#define WUBU_TQVEC_H

#include <stddef.h>
#include <stdint.h>

typedef struct wubu_tqvec wubu_tqvec;

/* d must be a positive multiple of 8; bits in [1,8]. The seed reproduces the whole transform, so the index needs no
 * stored matrix. Returns NULL on bad arguments or allocation failure. */
wubu_tqvec *wubu_tqvec_init(int d, int bits, uint64_t seed);
void        wubu_tqvec_free(wubu_tqvec *c);

int         wubu_tqvec_dim(const wubu_tqvec *c);
int         wubu_tqvec_bits(const wubu_tqvec *c);
size_t      wubu_tqvec_code_bytes(const wubu_tqvec *c);

/* x is expected to be (approximately) unit-norm; the construction assumes the unit sphere. */
void        wubu_tqvec_encode(const wubu_tqvec *c, const float *x, void *code);
float       wubu_tqvec_score(const wubu_tqvec *c, const float *q, const void *code);
void        wubu_tqvec_decode(const wubu_tqvec *c, const void *code, float *out);

#endif
