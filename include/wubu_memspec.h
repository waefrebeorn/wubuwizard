#ifndef WUBU_MEMSPEC_H
#define WUBU_MEMSPEC_H

/*
 * wubu_memspec.h -- EG-64: per-step memory-aware draft scheduling on edge
 * (MemSpec lineage), reimplemented in C11. One gate: throughput proxy under a
 * fixed memory cap versus head-of-line FIFO scheduling.
 */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int id;
    size_t bytes;
    int ticks;
} wubu_memstep_t;

/* Schedule all steps under 'cap' bytes. policy 0 = memory-aware shortest-step
 * first, 1 = FIFO with head-of-line blocking. Returns makespan in ticks, or
 * -1 on bad input. *peak receives peak resident bytes. */
int wubu_memspec_schedule(const wubu_memstep_t *steps, int n, size_t cap,
                          int policy, size_t *peak);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_MEMSPEC_H */
