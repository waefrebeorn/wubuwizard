/*
 * wubu_contracts.c -- the RUNTIME CONTRACTS (see the header).
 * Self-modification bounded by invariants, not just loss.
 */
#include "wubu_contracts.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_contracts_init(wubu_contracts_t *ct, wubu_hive_t *tissue)
{
    if (!ct || !tissue) return -1;
    memset(ct, 0, sizeof(*ct));
    ct->tissue = tissue;
    ct->next_version = 1;
    /* the default enforced set (the Lean-backed floor in runtime form):
     * ball closure (eps 1e-3), exp identity (1e-4), routing cap (1.0
     * = the expert load fraction), quant error (1e-2), finite guard */
    wubu_contract_kind_t kinds[5] = {
        WUBU_CT_BALL, WUBU_CT_EXP, WUBU_CT_ROUTE, WUBU_CT_QUANT, WUBU_CT_FINITE };
    float bounds[5] = { 1e-3f, 1e-4f, 1.0f, 1e-2f, 0.0f };
    for (int i = 0; i < 5; i++) {
        ct->list[ct->n].version = ct->next_version++;
        ct->list[ct->n].kind = kinds[i];
        ct->list[ct->n].bound = bounds[i];
        ct->list[ct->n].enabled = 1;
        ct->list[ct->n].batch = 0;
        ct->n++;
    }
    return 0;
}

int wubu_contracts_finite(const float *buf, size_t n)
{
    if (!buf) return 0;
    for (size_t i = 0; i < n; i++) {
        float v = buf[i];
        if (isnan(v) || isinf(v)) return 0;
    }
    return 1;
}

int wubu_contracts_check(wubu_contracts_t *ct, const float *probes)
{
    if (!ct || !probes) return -1;
    int violations = 0;
    for (int i = 0; i < ct->n; i++) {
        wubu_contract_t *c = &ct->list[i];
        if (!c->enabled) continue;
        ct->n_checks++;
        float v = probes[i];
        switch (c->kind) {
        case WUBU_CT_BALL:
            /* ball closure: the probe is ||x||/R - 1 (must be <= eps) */
            if (v > c->bound) violations++;
            break;
        case WUBU_CT_EXP:
            /* exp identity: |exp(a+b) - exp(a)exp(b)|/exp(a+b) (rel) */
            if (v > c->bound) violations++;
            break;
        case WUBU_CT_ROUTE:
            /* routing capacity: the probe is the load fraction */
            if (v > c->bound) violations++;
            break;
        case WUBU_CT_QUANT:
            /* quant error: the probe is the max abs relative error */
            if (v > c->bound) violations++;
            break;
        case WUBU_CT_FINITE:
            /* the probe is 1 = finite, 0 = NaN/Inf found */
            if (v < 0.5f) violations++;
            break;
        default:
            break;
        }
    }
    if (violations > 0) ct->n_violations += (uint64_t)violations;
    return violations;
}

uint32_t wubu_contracts_add(wubu_contracts_t *ct, wubu_contract_kind_t kind,
                            float bound, uint64_t batch)
{
    if (!ct || ct->n >= 8) return 0;
    wubu_contract_t *c = &ct->list[ct->n++];
    c->version = ct->next_version++;
    c->kind = kind;
    c->bound = bound;
    c->enabled = 1;
    c->batch = batch;
    /* the contract is a versioned hive meta-cell (the set evolves
     * under the same accept/rollback discipline) */
    if (ct->tissue) {
        wubu_contract_t *copy = (wubu_contract_t *)calloc(1, sizeof(wubu_contract_t));
        if (copy) { *copy = *c; wubu_hive_insert(ct->tissue, copy); }
    }
    return c->version;
}

void wubu_contracts_stats(const wubu_contracts_t *ct, char *buf, size_t cap)
{
    if (!ct || !buf || cap == 0) return;
    snprintf(buf, cap,
             "contracts=%d version=%u checks=%llu violations=%llu",
             ct->n, ct->next_version - 1,
             (unsigned long long)ct->n_checks,
             (unsigned long long)ct->n_violations);
}
