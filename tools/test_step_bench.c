#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define _POSIX_C_SOURCE 200809L
#include "wubu.h"
#include "wubu_train.h"
#include "wubu_runtime_dims.h"
int main(int argc, char **argv) {
    wubu_runtime_dims_t d; memset(&d,0,sizeof(d));
    d.vocab=16384; d.dim=512; d.layers=12; d.heads=8; d.kv_heads=1;
    d.head_dim=64; d.rope_dim=32; d.ffn_dim=2048; d.max_seq=2048;
    d.local_win=256; d.full_every=4; d.select_every=4; d.selectors=3;
    d.clip=10.0f; d.eps=1e-6f; d.rope_theta=10000.0f;
    wubu_runtime_dims_set(&d);
    wubu_model_t m;
    if (wubu_model_random_init(&m) != 0) { printf("init fail\n"); return 1; }
    wubu_train_t tr; memset(&tr,0,sizeof(tr));
    wubu_train_init(&tr, &m);
    wubu_buf_t b; memset(&b,0,sizeof(b));
    int seq = atoi(argv[1]);
    wubu_buf_alloc(&b, seq);
    uint16_t tok[4096];
    for (int i=0;i<seq;i++) tok[i]=(uint16_t)((i*7)%16384);
    wubu_train_zero_grad(&tr); wubu_train_microbatch(&m,&tr,&b,tok,seq);
    int nsteps = argc > 2 ? atoi(argv[2]) : 5;
    struct timespec t0,t1;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    float loss=0;
    for (int s=0;s<nsteps;s++){ wubu_train_zero_grad(&tr); loss=wubu_train_microbatch(&m,&tr,&b,tok,seq); }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    printf("%d steps %.2fs -> %.2fs/step | loss %.4f\n", nsteps, dt, dt/nsteps, loss);
    return 0;
}
