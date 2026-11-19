/* wubu_alloc.h -- paired release for aligned allocations, cross-platform.
 * On Windows the wubu_win.h shim maps posix_memalign to _aligned_malloc,
 * whose blocks MUST be released with _aligned_free -- plain free()
 * corrupts the UCRT heap (0xC0000374). On POSIX, free() is correct.
 * Include this in any file that frees posix_memalign/aligned_alloc memory. */
#ifndef WUBU_ALLOC_H
#define WUBU_ALLOC_H
#if defined(_WIN32)
#include <malloc.h>
#define wubu_aligned_free(p) _aligned_free(p)
#else
#include <stdlib.h>
#define wubu_aligned_free(p) free(p)
#endif
#endif /* WUBU_ALLOC_H */
