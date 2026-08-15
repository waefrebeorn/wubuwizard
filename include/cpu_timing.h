#ifndef WUBU_CPU_TIMING_H
#define WUBU_CPU_TIMING_H

/**
 * cpu_timing.h — Cycle-accurate CPU timing utilities
 *
 * Adapted from tailslayer (LaurieWired/tailslayer):
 *   rdtsc_lfence, rdtscp_lfence, clflush, mfence, pin_to_core,
 *   virt_to_phys, compute_channel, tsc_calibrate
 *
 * For use in profiling CUDA kernels, benchmarking CPU ops, and
 * detecting memory system behavior (cache misses, DRAM refresh).
 *
 * All functions are static inline — zero call overhead.
 */

#define WUBU_HOSTED
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <fcntl.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================
// Cycle-accurate timing (x86_64)
// ============================================================

/** Read timestamp counter with lfence serialization. Returns CPU cycles. */
static inline uint64_t wubu_rdtsc(void) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    uint64_t lo, hi;
    __asm__ __volatile__ ("lfence\nrdtsc" : "=a"(lo), "=d"(hi));
    return (hi << 32) | lo;
#else
    return 0;
#endif
}

/** Read timestamp counter and processor ID with lfence. Returns CPU cycles + AUX (core ID). */
static inline uint64_t wubu_rdtscp(uint32_t *aux) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    uint64_t lo, hi;
    __asm__ __volatile__ ("lfence\nrdtscp" : "=a"(lo), "=d"(hi), "=c"(*aux));
    return (hi << 32) | lo;
#else
    if (aux) *aux = 0;
    return 0;
#endif
}

/** Pin current thread to a specific CPU core. */
static inline int wubu_pin_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

// ============================================================
// Cache management
// ============================================================

/** Flush a single cache line containing address `addr`. */
static inline void wubu_clflush(const volatile void *addr) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    __asm__ __volatile__ ("clflush %0" :: "m"(*(const volatile char *)addr));
#endif
}

/** Flush range of memory. */
static inline void wubu_clflush_range(const volatile void *addr, size_t size) {
    const char *ptr = (const char *)addr;
    for (size_t i = 0; i < size; i += 64) {
        wubu_clflush(ptr + i);
    }
}

/** Memory barriers */
static inline void wubu_mfence(void) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    __asm__ __volatile__ ("mfence" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/** Read TSC with serialization and return frequency estimate (cycles/sec). */
static inline double wubu_tsc_calibrate(int seconds) {
    uint64_t start = wubu_rdtsc();
    sleep(seconds);
    uint64_t end = wubu_rdtsc();
    return (double)(end - start) / seconds;
}

/** Virtual to physical address translation (Linux only, requires root). */
static inline uint64_t wubu_virt_to_phys(void *vaddr) {
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    
    uint64_t v = (uint64_t)vaddr;
    uint64_t page = v / 4096;
    uint64_t entry;
    if (lseek(fd, page * 8, SEEK_SET) == (off_t)-1) { close(fd); return 0; }
    if (read(fd, &entry, 8) != 8) { close(fd); return 0; }
    close(fd);
    
    if (!(entry & (1ULL << 63))) return 0; // Not present
    
    uint64_t pfn = entry & ((1ULL << 55) - 1);
    return (pfn * 4096) + (v % 4096);
}


static inline void wubu_lfence(void) {
#if (defined(__x86_64__) || defined(__i386__)) && !defined(__CUDACC__)
    __asm__ __volatile__ ("lfence" ::: "memory");
#else
    __sync_synchronize();
#endif
}

/** Calibrate TSC frequency in GHz using clock_gettime. */
static inline double wubu_tsc_calibrate_ghz(void) {
    struct timespec t0, t1;
    uint32_t aux;
    uint64_t tsc0 = wubu_rdtscp(&aux);
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Busy-wait ~100ms for calibration
    struct timespec deadline = t0;
    deadline.tv_nsec += 100000000;
    if (deadline.tv_nsec >= 1000000000) { deadline.tv_sec++; deadline.tv_nsec -= 1000000000; }
    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (t1.tv_sec > deadline.tv_sec || (t1.tv_sec == deadline.tv_sec && t1.tv_nsec >= deadline.tv_nsec))
            break;
    }
    uint64_t tsc1 = wubu_rdtscp(&aux);

    double elapsed_ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                        (double)(t1.tv_nsec - t0.tv_nsec);
    return (double)(tsc1 - tsc0) / elapsed_ns;
}

/**
 * Convert TSC cycles to nanoseconds.
 */
static inline double wubu_cycles_to_ns(uint64_t cycles, double tsc_ghz) {
    return (double)cycles / tsc_ghz;
}

/**
 * Convert TSC cycles to microseconds.
 */
static inline double wubu_cycles_to_us(uint64_t cycles, double tsc_ghz) {
    return (double)cycles / (tsc_ghz * 1000.0);
}

// ============================================================
// Benchmark helpers: timed section
// ============================================================

/**
 * Begin a timed section. Returns start cycle count.
 * Call wubu_timing_end() after the section.
 */
static inline uint64_t wubu_timing_start(void) {
    wubu_lfence();
    return wubu_rdtsc();
}

/**
 * End a timed section. Returns elapsed cycles.
 */
static inline uint64_t wubu_timing_end(uint64_t start) {
    uint32_t aux;
    uint64_t end = wubu_rdtscp(&aux);
    wubu_lfence();
    return end - start;
}

/**
 * Clflush+reload timing probe: measure read latency for a single address.
 * Returns cycles taken to reload after clflush.
 */
static inline uint64_t wubu_cache_probe(volatile void *addr) {
    wubu_clflush((void*)addr);
    wubu_mfence();
    wubu_lfence();
    uint64_t t0 = wubu_rdtsc();
    (void)*(volatile char*)addr;  // force read
    return wubu_timing_end(t0);
}

// ============================================================
// Memory channel info
// ============================================================

/**
 * Compute DRAM channel number from physical address.
 * Channel bit varies by platform: 6 for DDR4, 8 for DDR5 typical.
 * Returns 0 or 1 for dual-channel, or more for multi-channel.
 */
static inline int wubu_compute_channel(uint64_t phys_addr, int channel_bit) {
    return (int)((phys_addr >> channel_bit) & 1);
}

/** Print system timing info to stderr */
static inline void wubu_print_timing_info(double tsc_ghz) {
    fprintf(stderr, "  TSC: %.3f GHz (%.2f cycles/ns)\n",
            tsc_ghz, tsc_ghz);
    fprintf(stderr, "  1 cycle = %.2f ns\n", 1.0 / tsc_ghz);
    fprintf(stderr, "  1 us = %.0f cycles\n", tsc_ghz * 1000.0);
    fprintf(stderr, "  1 ms = %.0f cycles\n", tsc_ghz * 1000000.0);
}


#ifdef __cplusplus
}
#endif

#endif // WUBU_CPU_TIMING_H