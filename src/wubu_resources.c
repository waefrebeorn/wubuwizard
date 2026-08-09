/*
 * wubu_resources.c -- the RESOURCE LEDGER (see the header).
 * getrusage + /proc/self/status (WSL/Linux).
 */
#define _POSIX_C_SOURCE 199309L   /* before ANY include: CLOCK_MONOTONIC */

#include "wubu_resources.h"

#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>

static uint64_t read_rss_kb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    uint64_t rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%llu", (unsigned long long *)&rss);
            break;
        }
    }
    fclose(f);
    return rss;
}

void wubu_res_snapshot(wubu_res_t *r)
{
    if (!r) return;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    memset(r, 0, sizeof(*r));
    r->rss_kb = read_rss_kb();
    r->cpu_sec = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1e6 +
                 (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1e6;
    r->soft_fitness = 1.0;   /* no budget yet = comfortable */
}

void wubu_res_update(wubu_res_t *r, uint64_t events_since_last)
{
    if (!r) return;
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double now_cpu = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1e6 +
                     (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1e6;
    static struct timespec last_wall = {0, 0};
    struct timespec now_wall;
    clock_gettime(CLOCK_MONOTONIC, &now_wall);
    double wall = 0;
    if (last_wall.tv_sec != 0) {
        wall = (double)(now_wall.tv_sec - last_wall.tv_sec) +
               (double)(now_wall.tv_nsec - last_wall.tv_nsec) / 1e9;
        if (wall < 0) wall = 0;
    }
    last_wall = now_wall;

    r->cpu_delta = now_cpu - r->cpu_sec;
    r->cpu_sec = now_cpu;
    r->wall_delta = wall;
    r->rss_kb = read_rss_kb();
    r->throughput = wall > 0 ? (double)events_since_last / wall : 0.0;

    /* the soft fitness: RSS within the budget + CPU not spiking (a
     * 1.0 delta over a short wall window means the run is CPU-bound) */
    double f = 1.0;
    if (r->budget_kb > 0) {
        double mem = (double)r->rss_kb / (double)r->budget_kb;
        if (mem > 1.0) f *= 0.2; else if (mem > 0.8) f *= 0.6;
    }
    if (r->wall_delta > 0.05 && r->cpu_delta / r->wall_delta > 1.5)
        f *= 0.7;   /* the CPU spiked past the wall (thrash) */
    r->soft_fitness = f;
}

void wubu_res_set_budget(wubu_res_t *r, uint64_t budget_kb)
{
    if (r) r->budget_kb = budget_kb;
}
