#ifndef WUBU_EXPERT_STAGE_H
#define WUBU_EXPERT_STAGE_H

/*
 * wubu_expert_stage.h -- EG-45c: coordinated expert layer staging.
 * While layer i computes, the next layer's expert weights are staged into a
 * second buffer. Any staging time beyond the compute window is a stall.
 * This gates the schedule arithmetic with synthetic bandwidth, not a measured
 * I/O trace or a real double-buffer implementation.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_EXPERT_STAGE_MAX_LAYERS 256

int wubu_expert_stage_plan(const int *bytes, const double *compute, int n,
                           double bandwidth_bytes_per_s, double *stall_out);

uint64_t wubu_expert_stage_overlapped_bytes(const int *bytes, int n);

#ifdef __cplusplus
}
#endif
#endif /* WUBU_EXPERT_STAGE_H */
