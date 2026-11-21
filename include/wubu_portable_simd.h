/*
 * wubu_portable_simd.h — x86/ARM portable intrinsic gate.
 *
 * Rule: intrinsics stay in x86 translation units. Every other backend gets
 * the scalar path that already exists under #else in the owning module.
 * This header only supplies the include and a no-op prefetch on non-x86.
 */
#ifndef WUBU_PORTABLE_SIMD_H
#define WUBU_PORTABLE_SIMD_H

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define WUBU_X86_SIMD 1
#  include <immintrin.h>
#  define WUBU_PREFETCH(p, hint) _mm_prefetch((const char *)(p), (hint))
#else
#  define WUBU_X86_SIMD 0
#  define WUBU_PREFETCH(p, hint) ((void)0)
#endif

#endif /* WUBU_PORTABLE_SIMD_H */
