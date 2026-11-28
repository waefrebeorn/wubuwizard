/* Stream-count probe: does achieved bandwidth depend on HOW MANY streams a core walks, at equal total bytes?
 *
 * The engine sustains 0.81 GB/s over ~192 short expert slabs per token while a plain sequential read of the same bytes sustains
 * 2.2-8.0 GB/s. If that is prefetcher capacity, then reading the SAME total bytes as 1 contiguous stream must be much faster
 * than as 8 or 192 interleaved streams. This probe answers that with no kernel, no file, no mmap: pure access pattern.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>

static double now(void){ static LARGE_INTEGER f; LARGE_INTEGER c;
    if(!f.QuadPart) QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart/(double)f.QuadPart; }

#define TOTAL   (256u * 1024 * 1024)     /* bytes read per measurement */
#define SLAB    (1536u * 1024)          /* 1.5 MB - close to the engine's ~1.37 MB expert slab */
#define STRIDE  (16u * 1024 * 1024)     /* streams start 16 MB apart so they cannot share prefetch windows */

static uint64_t sink;
static inline uint64_t walk(const unsigned char *p, size_t n)
{
    uint64_t a = 0;
    for (size_t i = 0; i < n; i += 64) a += p[i];      /* one load per cache line: pure memory, no compute */
    return a;
}

static void measure(const char *label, unsigned char *buf, int nstreams, int nslabs_per_stream)
{
    uint64_t acc = 0;
    double t0 = now();
    for (int s = 0; s < nslabs_per_stream; ++s)
        for (int k = 0; k < nstreams; ++k) {
            const size_t off = (size_t)k * STRIDE + (size_t)(s % 8) * SLAB;
            acc += walk(buf + off, SLAB);
        }
    double dt = now() - t0;
    const double bytes = (double)nstreams * nslabs_per_stream * SLAB;
    sink = acc;
    printf("  %-28s streams=%3d  %8.2f MB  %6.2f GB/s\n", label, nstreams, bytes / 1048576.0, bytes / dt / 1e9);
}

int main(void)
{
    unsigned char *buf = (unsigned char *) VirtualAlloc(NULL, 512u * 1024 * 1024,
                                                        MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!buf) { printf("alloc failed\n"); return 2; }
    for (size_t i = 0; i < 512u * 1024 * 1024; i += 4096) buf[i] = (unsigned char)(i >> 12);  /* fault in once */

    printf("stream-count probe, %.0f MB per measurement, slab=%.1f MB, stride=%.0f MB\n",
           TOTAL / 1048576.0, SLAB / 1048576.0, STRIDE / 1048576.0);

    /* 1 stream: walk one slab repeatedly -> the TRULY contiguous case is measured by using nstreams=1, nslabs=171 */
    measure("1 stream (1 slab, reused)", buf, 1, (int)(TOTAL / SLAB));
    measure("8 stream (8 slabs round-robin)", buf, 8, (int)(TOTAL / (8 * SLAB)));
    measure("24 stream (24 slabs round-robin)", buf, 24, (int)(TOTAL / (24 * SLAB)));

    VirtualFree(buf, 0, MEM_RELEASE);
    printf("  (sink %llu)\n", (unsigned long long)sink);
    return 0;
}
