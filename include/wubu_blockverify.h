#ifndef WUBU_BLOCKVERIFY_H
#define WUBU_BLOCKVERIFY_H

/*
 * wubu_blockverify.h -- EG-61: block verification that keeps the longest
 * accepted prefix and emits one distribution-exact correction on rejection
 * (SGLang speculative-decoding lineage), reimplemented in C11.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Verify draft[0..n-1] against target p / draft q. uniforms holds two values
 * per position: acceptance draw and correction draw. Returns emitted token
 * count (accepted prefix plus optional correction); sets *accepted to the
 * accepted prefix length. Distribution of the first emitted token is p. */
int wubu_block_verify(const float *p, const float *q, int vocab,
                      const int *draft, int n, const float *uniforms,
                      int *out, int *accepted);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_BLOCKVERIFY_H */
