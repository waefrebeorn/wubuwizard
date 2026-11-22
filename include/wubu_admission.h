#ifndef WUBU_ADMISSION_H
#define WUBU_ADMISSION_H

/*
 * wubu_admission.h -- EG-10: one shared admission validator with a token
 * budget, queue capacity and timeout (TGI shared validator / TensorRT
 * capacity admission lineage), reimplemented in C11.
 *
 * Every request passes the same validator before it can consume engine
 * capacity. Reservations are freed by release() or by timeout expiry.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    WUBU_ADMIT_OK = 0,
    WUBU_ADMIT_ERR_ARG = 1,
    WUBU_ADMIT_OVER_BUDGET = 2,
    WUBU_ADMIT_TIMEOUT = 3,
    WUBU_ADMIT_QUEUE_FULL = 4,
    WUBU_ADMIT_DUPLICATE = 5
} wubu_admit_status_t;

typedef struct wubu_admission wubu_admission_t;

wubu_admission_t *wubu_admission_create(uint64_t token_budget,
                                         int max_queue, int timeout_steps);
void wubu_admission_free(wubu_admission_t *a);

/* Validate and reserve. now_step is the caller's monotonic step counter. */
wubu_admit_status_t wubu_admission_validate(wubu_admission_t *a, int req_id,
                                              uint64_t tokens, int now_step);

/* Release a request's reservation. Returns 0 or -1 if unknown id. */
int wubu_admission_release(wubu_admission_t *a, int req_id);

/* Expire requests older than timeout_steps. Returns the number expired. */
int wubu_admission_step(wubu_admission_t *a, int now_step);

uint64_t wubu_admission_reserved(const wubu_admission_t *a);
int wubu_admission_queue_len(const wubu_admission_t *a);
uint64_t wubu_admission_accepted(const wubu_admission_t *a);
uint64_t wubu_admission_rejected(const wubu_admission_t *a);
uint64_t wubu_admission_expired(const wubu_admission_t *a);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_ADMISSION_H */
