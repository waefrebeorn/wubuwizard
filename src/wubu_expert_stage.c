/* wubu_expert_stage.c -- EG-45c expert staging schedule gate. Strict C11. */
#include "wubu_expert_stage.h"

int wubu_expert_stage_plan(const int *bytes, const double *compute, int n,
                           double bandwidth_bytes_per_s, double *stall_out)
{
    double stall = 0.0;
    if (!bytes || !compute || !stall_out || n < 1 ||
        n > WUBU_EXPERT_STAGE_MAX_LAYERS ||
        !(bandwidth_bytes_per_s > 0.0)) return -1;
    for (int i = 0; i < n; ++i)
        if (bytes[i] <= 0 || !(compute[i] >= 0.0)) return -1;
    for (int i = 0; i + 1 < n; ++i) {
        double stage = (double)bytes[i + 1] / bandwidth_bytes_per_s;
        if (stage > compute[i]) stall += stage - compute[i];
    }
    *stall_out = stall;
    return 0;
}

uint64_t wubu_expert_stage_overlapped_bytes(const int *bytes, int n)
{
    uint64_t total = 0;
    if (!bytes || n < 0 || n > WUBU_EXPERT_STAGE_MAX_LAYERS) return 0;
    for (int i = 1; i < n; ++i) {
        if (bytes[i] <= 0) return 0;
        total += (uint64_t)bytes[i];
    }
    return total;
}
