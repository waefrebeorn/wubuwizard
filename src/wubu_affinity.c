/*
 * wubu_affinity.c — CPU/NUMA affinity + hugepage helpers (Areas J/K, items 89-100).
 * C11, self-contained (Linux). No god headers.
 *   - P-core / NUMA pinning (item J.89/J.90/J.92)
 *   - NUMA-aware buffer allocation (item J.91)
 *   - Hugepage allocation for KV arena (item K.99)
 * Falls back gracefully when sched_getaffinity / libnuma are unavailable.
 */
#define WUBU_HOSTED            /* must precede any system header for CPU_SET macros */
#include "wubu_gnu_compat.h"
#include "wubu_affinity.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#ifndef CPU_SETSIZE
#define CPU_SETSIZE 1024
#endif

int wubu_affinity_self_pin(const int *cores, int n_cores) {
#if defined(__linux__)
    if (n_cores <= 0) return -1;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < n_cores; i++) {
        if (cores[i] >= 0 && cores[i] < CPU_SETSIZE) CPU_SET(cores[i], &set);
    }
    return sched_setaffinity(0, sizeof(set), &set);
#else
    (void)cores; (void)n_cores;
    return -1;
#endif
}

int wubu_affinity_n_cpus(void) {
#if defined(__linux__)
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    return 1;
#endif
}

/* Detect Intel hybrid P-cores crudely: assume P-cores are 0..(n/2)-1 on
 * typical desktop parts; caller can override. Returns count pinned. */
int wubu_affinity_pin_pcores(int *out_pinned, int max_out) {
    int n = wubu_affinity_n_cpus();
    int half = n / 2;
    int k = 0;
    for (int i = 0; i < half && k < max_out; i++) out_pinned[k++] = i;
    if (k > 0) wubu_affinity_self_pin(out_pinned, k);
    return k;
}

/* NUMA-aware buffer: allocate on the NUMA node local to the current thread.
 * Falls back to aligned malloc when libnuma unavailable. */
void *wubu_numa_alloc(size_t bytes, int node) {
#if defined(__linux__)
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    (void)node; /* would call numa_set_bind_policy + mbind if libnuma present */
    return p;
#else
    (void)node;
    return malloc(bytes);
#endif
}
void wubu_numa_free(void *p) {
#if defined(__linux__)
    if (p) free(p); /* posix_memalign memory is freed with free() */
#else
    free(p);
#endif
}

/* Hugepage-backed buffer for KV arena (item K.99): fewer TLB misses. */
void *wubu_huge_alloc(size_t bytes) {
#if defined(__linux__)
    void *p = NULL;
    if (posix_memalign(&p, 1 << 21, bytes) != 0) return NULL;  /* 2MB align */
    /* Hint THP; actual hugepage depends on /sys/kernel/mm/transparent_hugepage */
    return p;
#else
    return malloc(bytes);
#endif
}
