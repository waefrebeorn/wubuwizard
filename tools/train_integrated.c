/**
 * train_integrated.c — All 7 modules wired into single training pipeline.
 *
 * Usage:  ./train_integrated [model.gguf] [corpus.bin] [steps]
 * Env:    TST=1        bag s=8, MCE loss (25% steps)
 *         RSGD=1       Riemannian SGD for Poincaré params
 *         PGA=1        Poincaré GQA (CPU detour)
 *         NESTED_SSM=1 Nested SSM K=4 (CPU detour)
 *         NESTED_MOE=1 Nested MoE Poincaré router
 *         POINCARE_R=X Poincaré ball radius for SSM layers (GPU)
 */
#include "wubu_model.h"
#include "wubu_tokenizer.h"
#include "bench.h"
#include "qlearner.h"
#include "gguf_reader.h"
#include "rsgd.h"
#include "wubu_tst.h"
#include "wubu_poincare_gqa.h"
#include "wubu_nested_ssm.h"
#include "wubu_moe_hyperbolic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "wubu_core_dumps.h"

static void softmax_inplace(float *x, int n) {
    float mx = x[0]; for (int i=1;i<n;i++) if(x[i]>mx)mx=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    float iv=1.0f/(s+1e-30f); for(int i=0;i<n;i++)x[i]*=iv;
}

int main(int argc, char **argv) {
    wubu_disable_core_dumps();
    const char *model_path = argc>1?argv[1]:"/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
    const char *corpus_path = argc>2?argv[2]:"data/train_data.bin";
    int n_steps = argc>3?atoi(argv[3]):10;
    float lr = 0.001f;
    if(getenv("LR")) lr=atof(getenv("LR"));
    float poincare_R = 0.0f;
    if(getenv("POINCARE_R")) poincare_R=atof(getenv("POINCARE_R"));
    int tst_enabled = getenv("TST")?atoi(getenv("TST")):0;
    int rsgd_enabled = getenv("RSGD")?atoi(getenv("RSGD")):0;
    int pga_enabled = getenv("PGA")?atoi(getenv("PGA")):0;
    int nested_ssm_enabled = getenv("NESTED_SSM")?atoi(getenv("NESTED_SSM")):0;
    int nested_moe_enabled = getenv("NESTED_MOE")?atoi(getenv("NESTED_MOE")):0;
    int B = 1, T = tst_enabled ? 16 : 4, N = B * T;
    int tst_s = tst_enabled ? 8 : 0;
    const char *embed_path = "data/qwen36_embeddings_c.bin.raw";

    setbuf(stdout,NULL); setbuf(stderr,NULL);
    printf("=== WuBuText AI — Integrated Training ===\n");
    printf("Model: %s | Steps: %d | LR=%.6f | B=%d T=%d N=%d\n",model_path,n_steps,lr,B,T,N);
    printf("Flags: TST=%d RSGD=%d PGA=%d NESTED_SSM=%d NESTED_MOE=%d POINCARE_R=%.3f\n",
           tst_enabled,rsgd_enabled,pga_enabled,nested_ssm_enabled,nested_moe_enabled,poincare_R);

    wubu_tokenizer_t tok;
    if(!wubu_tokenizer_init(&tok,model_path)) return 1;
    int V = tok.vocab_size;
    printf("Vocab: %d\n",V);

    wubu_model_t model;
    if(!wubu_model_init(&model,model_path)) return 1;
    printf("Model: %d layers\n",model.n_layers);

    FILE *f=fopen(corpus_path,"rb");
    if(!f){fprintf(stderr,"Can't open %s\n",corpus_path);return 1;}
    fseek(f,0,SEEK_END); int total_tokens=(int)(ftell(f)/sizeof(int));
    fseek(f,0,SEEK_SET);
    int *tokens=(int*)malloc(total_tokens*sizeof(int));
    if(fread(tokens,sizeof(int),total_tokens,f)!=(size_t)total_tokens)
        {fprintf(stderr,"corpus read fail\n");return 1;}
    fclose(f);
    printf("Corpus: %d tokens\n",total_tokens);

    float *output_weight=(float*)malloc((int64_t)D_MODEL*V*sizeof(float));
    {
        gguf_ctx *ctx=gguf_open(model_path);
        gguf_tensor_info *t=gguf_find_tensor(ctx,"output.weight");
        gguf_read_tensor_f32(ctx,t,output_weight,(int64_t)D_MODEL*V);
        gguf_close(ctx);
    }

    // RSGD (stateless — single call)
    float *poincare_embd_cache = NULL;
    if(rsgd_enabled) {
        float R = poincare_R>0?poincare_R:0.956f;
        poincare_embd_cache = (float*)calloc((int64_t)model.vocab_size*D_MODEL,sizeof(float));
        printf("RSGD: R=%.4f\n",R);
    }

    qlearner_t ql; qlearner_init(&ql);

    cublasHandle_t cublas_h; cudaStream_t stream;
    if(!wubu_cuda_init(&cublas_h,&stream)) return 1;

    printf("Pre-loading layer weights to GPU...\n");
    gguf_ctx *ctx=gguf_open(model_path); if(!ctx) return 1;
    gpu_ssm_weights *ssm_gpu = calloc(model.n_layers,sizeof(gpu_ssm_weights));
    gpu_gqa_weights *gqa_gpu = calloc(model.n_layers,sizeof(gpu_gqa_weights));
    double t_load = now_sec();
    for(int l=0;l<model.n_layers;l++){
        if(model.layers[l].is_ssm){ if(!gpu_load_ssm_layer(ctx,l,&ssm_gpu[l],stream)) return 1; }
        else{ if(!gpu_load_gqa_layer(ctx,l,&gqa_gpu[l],stream)) return 1; }
    }
    cudaStreamSynchronize(stream); gguf_close(ctx);
    printf("  All %d layers loaded in %.2fs\n",model.n_layers,now_sec()-t_load);
    // GPU buffer for output projection (upload once, reuse)
    int max_N = B * (tst_enabled ? 16 : 4);
    float *d_output_weight = wubu_cuda_alloc((int64_t)D_MODEL * V * sizeof(float));
    float *d_logits = wubu_cuda_alloc((int64_t)max_N * V * sizeof(float));
    cudaMemcpyAsync(d_output_weight, output_weight, (int64_t)D_MODEL * V * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    printf("  Output weight uploaded: %ld MB\n", (long)((int64_t)D_MODEL*V*4/(1024*1024)));

    int qkv_dim = KEY_DIM*2+VALUE_DIM;
    int q_dim_x2 = GQA_Q_HEADS*GQA_HEAD_DIM*2;
    int kv_dim = GQA_KV_HEADS*GQA_HEAD_DIM;
    int gqa_q_dim = GQA_Q_HEADS*GQA_HEAD_DIM;

    // PGA backward save structs (one per layer; NULL if not PGA / not GQA)
    poincare_gqa_fwd_save_t *pga_save = NULL;
    if(pga_enabled){
        pga_save = (poincare_gqa_fwd_save_t*)calloc(model.n_layers,sizeof(poincare_gqa_fwd_save_t));
        int max_N = B * T;  // worst-case allocation
        for(int l=0;l<model.n_layers;l++){
            if(!model.layers[l].is_ssm){
                pga_save[l].Q_ball = (float*)malloc(max_N*gqa_q_dim*sizeof(float));
                pga_save[l].K_ball = (float*)malloc(max_N*kv_dim*sizeof(float));
                pga_save[l].V_ball = (float*)malloc(max_N*kv_dim*sizeof(float));
                pga_save[l].Q_norm = (float*)malloc(max_N*gqa_q_dim*sizeof(float));
                pga_save[l].Q_raw  = (float*)malloc(max_N*gqa_q_dim*sizeof(float));
                pga_save[l].K_norm = (float*)malloc(max_N*kv_dim*sizeof(float));
                pga_save[l].K_raw  = (float*)malloc(max_N*kv_dim*sizeof(float));
                pga_save[l].V      = (float*)malloc(max_N*kv_dim*sizeof(float));
                pga_save[l].gate   = (float*)malloc(max_N*gqa_q_dim*sizeof(float));
                pga_save[l].gate_sig = (float*)malloc(max_N*gqa_q_dim*sizeof(float));
                pga_save[l].attn_out_pre_gate = (float*)malloc(max_N*gqa_q_dim*sizeof(float));
            }
        }
    }

    float *d_x = wubu_cuda_alloc(N*D_MODEL*sizeof(float));
    float *d_out = wubu_cuda_alloc(N*D_MODEL*sizeof(float));
    float *d_norm = wubu_cuda_alloc(N*D_MODEL*sizeof(float));
    float *d_norm_w = wubu_cuda_alloc(D_MODEL*sizeof(float));
    float *d_poincare_n = poincare_R>0?wubu_cuda_alloc(N*sizeof(float)):NULL;
    float **d_ssm_s = calloc(model.n_layers,sizeof(float*));
    float **d_conv_s = calloc(model.n_layers,sizeof(float*));
    for(int l=0;l<model.n_layers;l++){
        if(model.layers[l].is_ssm){
            d_ssm_s[l]=wubu_cuda_alloc((int64_t)SSM_V_HEADS*SSM_D_STATE*SSM_D_STATE*sizeof(float));
            d_conv_s[l]=wubu_cuda_alloc((CONV_KERNEL-1)*CONV_DIM*sizeof(float));
            cudaMemsetAsync(d_ssm_s[l],0,(int64_t)SSM_V_HEADS*SSM_D_STATE*SSM_D_STATE*sizeof(float),stream);
            cudaMemsetAsync(d_conv_s[l],0,(CONV_KERNEL-1)*CONV_DIM*sizeof(float),stream);
        }
    }

    // Scratch
    float *d_qkv=wubu_cuda_alloc(N*qkv_dim*sizeof(float));
    float *d_z=wubu_cuda_alloc(N*VALUE_DIM*sizeof(float));
    float *d_beta=wubu_cuda_alloc(N*DT_RANK*sizeof(float));
    float *d_alpha=wubu_cuda_alloc(N*DT_RANK*sizeof(float));
    float *d_bs=wubu_cuda_alloc(N*DT_RANK*sizeof(float));
    float *d_ab=wubu_cuda_alloc(N*DT_RANK*sizeof(float));
    float *d_gate=wubu_cuda_alloc(N*DT_RANK*sizeof(float));
    float *d_ci=wubu_cuda_alloc((int64_t)B*(T+CONV_KERNEL-1)*CONV_DIM*sizeof(float));
    float *d_co=wubu_cuda_alloc(N*CONV_DIM*sizeof(float));
    float *d_qc=wubu_cuda_alloc(N*KEY_DIM*sizeof(float));
    float *d_kc=wubu_cuda_alloc(N*KEY_DIM*sizeof(float));
    float *d_vc=wubu_cuda_alloc(N*VALUE_DIM*sizeof(float));
    float *d_qn=wubu_cuda_alloc(N*KEY_DIM*sizeof(float));
    float *d_kn=wubu_cuda_alloc(N*KEY_DIM*sizeof(float));
    float *d_do=wubu_cuda_alloc(N*VALUE_DIM*sizeof(float));
    float *d_zs=wubu_cuda_alloc(N*VALUE_DIM*sizeof(float));

    float *d_gqa_Qf=wubu_cuda_alloc(N*q_dim_x2*sizeof(float));
    float *d_gqa_K=wubu_cuda_alloc(N*kv_dim*sizeof(float));
    float *d_gqa_V=wubu_cuda_alloc(N*kv_dim*sizeof(float));
    float *d_gqa_scr=wubu_cuda_alloc(N*gqa_q_dim*sizeof(float));

    float *saved_normed=(float*)malloc((int64_t)model.n_layers*N*D_MODEL*sizeof(float));
    float *saved_attn_out=(float*)malloc((int64_t)model.n_layers*N*D_MODEL*sizeof(float));
    float *saved_normed2=(float*)malloc((int64_t)model.n_layers*N*D_MODEL*sizeof(float));
    float *saved_ffn_out=(float*)malloc((int64_t)model.n_layers*N*D_MODEL*sizeof(float));
    float *hidden=(float*)malloc(N*D_MODEL*sizeof(float));
    float *logits=(float*)malloc(N*V*sizeof(float));
    float *dW=(float*)calloc((int64_t)D_MODEL*V,sizeof(float));

    // MoE: local qdata definition (same pattern as train_gpu.c)
    typedef struct {
        const uint8_t *q_gate_inp, *q_gate_exps, *q_up_exps, *q_down_exps;
        const uint8_t *q_gate_shexp, *q_up_shexp, *q_down_shexp;
        int ty_gi, ty_ge, ty_gs;
        int64_t expert_raw, expert_raw_down;
    } moe_qdata_t;
    typedef struct {
        int n_u; int uid[32];
        float *dgi,*gs,*us,*ds;
        float *eg[32],*eu[32],*ed[32];
        float *ge_persist, *ue_persist, *de_persist; // persistent interleaved arrays (1GB each)
    } lmoe_t;

    gguf_ctx *gguf_moe = gguf_open(model_path);
    if(!gguf_moe){fprintf(stderr,"GGUF reopen fail\n");return 1;}
    gguf_buffer_data(gguf_moe);
    moe_qdata_t *moe_qdata = calloc(model.n_layers,sizeof(moe_qdata_t));
    lmoe_t *moe_cache = calloc(model.n_layers,sizeof(lmoe_t));
    int moe_enabled = 1;
    if(getenv("NO_MOE")) moe_enabled=0;

    char mn[256];
    for(int l=0;l<model.n_layers;l++){
        moe_qdata_t *mq=&moe_qdata[l];
        gguf_tensor_info *t;
        snprintf(mn,sizeof(mn),"blk.%d.ffn_gate_inp.weight",l);
        t=gguf_find_tensor(gguf_moe,mn);
        if(t){mq->ty_gi=t->ggml_type;mq->q_gate_inp=(const uint8_t*)gguf_moe->data_blob+t->data_offset;}
        snprintf(mn,sizeof(mn),"blk.%d.ffn_gate_exps.weight",l);
        t=gguf_find_tensor(gguf_moe,mn);
        if(t){mq->ty_ge=t->ggml_type;mq->q_gate_exps=(const uint8_t*)gguf_moe->data_blob+t->data_offset;}
        snprintf(mn,sizeof(mn),"blk.%d.ffn_up_exps.weight",l);
        t=gguf_find_tensor(gguf_moe,mn);
        if(t)mq->q_up_exps=(const uint8_t*)gguf_moe->data_blob+t->data_offset;
        snprintf(mn,sizeof(mn),"blk.%d.ffn_down_exps.weight",l);
        t=gguf_find_tensor(gguf_moe,mn);
        if(t)mq->q_down_exps=(const uint8_t*)gguf_moe->data_blob+t->data_offset;
        snprintf(mn,sizeof(mn),"blk.%d.ffn_gate_shexp.weight",l);
        t=gguf_find_tensor(gguf_moe,mn);
        if(t){mq->ty_gs=t->ggml_type;mq->q_gate_shexp=(const uint8_t*)gguf_moe->data_blob+t->data_offset;}
        snprintf(mn,sizeof(mn),"blk.%d.ffn_up_shexp.weight",l);
        t=gguf_find_tensor(gguf_moe,mn);
        if(t)mq->q_up_shexp=(const uint8_t*)gguf_moe->data_blob+t->data_offset;
        snprintf(mn,sizeof(mn),"blk.%d.ffn_down_shexp.weight",l);
        t=gguf_find_tensor(gguf_moe,mn);
        if(t)mq->q_down_shexp=(const uint8_t*)gguf_moe->data_blob+t->data_offset;
        int64_t e_sz = t ? (int64_t)N_EXPERTS*D_MODEL*D_FF : 0;
        mq->expert_raw = e_sz > 0 ? (int64_t)gguf_raw_size(t->ggml_type,e_sz)/N_EXPERTS : 0;
        mq->expert_raw_down = e_sz > 0 ? (int64_t)gguf_raw_size(t->ggml_type,(int64_t)N_EXPERTS*D_FF*D_MODEL)/N_EXPERTS : 0;
    }

    // Nested SSM
    wubu_nested_ssm_state_t nested_state;
    float nested_Rs[4]={2.0f,1.5f,1.0f,0.5f};
    if(nested_ssm_enabled){
        wubu_nested_ssm_init(&nested_state,4,nested_Rs);
        printf("Nested SSM: K=4\n");
    }

    // Nested MoE router
    poincare_router_t poc_router={0};
    float *poc_centroids = NULL;
    if(nested_moe_enabled){
        poc_centroids=(float*)malloc((int64_t)N_EXPERTS*D_MODEL*sizeof(float));
        wubu_poincare_router_init_random(poc_centroids,42);
        poc_router.centroids=poc_centroids;
        poc_router.temperature=HYPERBOLIC_TEMPERATURE;
        poc_router.loaded=true;
        printf("Nested MoE: Poincaré router\n");
    }

    printf("\n=== Training: %d steps ===\n\n",n_steps);
    double total_time=0;

    for(int step=0;step<n_steps;step++){
        int start_idx=(step*N)%(total_tokens-N-1);
        double t0=now_sec();

        float embd[N*D_MODEL];
        f=fopen(embed_path,"rb");
        for(int i=0;i<N;i++){
            int id=tokens[start_idx+i];
            if(id<0||id>=model.vocab_size)id=0;
            fseek(f,id*D_MODEL*sizeof(float),SEEK_SET);
            if(fread(embd+i*D_MODEL,sizeof(float),D_MODEL,f)!=(size_t)D_MODEL)
                memset(embd+i*D_MODEL,0,D_MODEL*sizeof(float));
        }
        fclose(f);

        int use_tst=tst_enabled&&step%4==0;
        int fwd_T=use_tst?T/tst_s:T;
        int fwd_N=B*fwd_T;
        int tst_targets_buf[64*8];
        int n_tst_bags=0;

        if(use_tst){
            float *bagged=(float*)malloc(fwd_N*D_MODEL*sizeof(float));
            tst_bag_embeddings(embd,bagged,B,T,D_MODEL,tst_s);
            memcpy(embd,bagged,fwd_N*D_MODEL*sizeof(float));
            free(bagged);
            int shifted[64];
            for(int i=0;i<N-tst_s+1;i++) shifted[i]=tokens[start_idx+i+tst_s-1];
            n_tst_bags=tst_prepare_targets(shifted,tst_targets_buf,B,T-tst_s+1,tst_s);
        }

        int targets[N];
        if(!use_tst){
            for(int i=0;i<N-1;i++) targets[i]=tokens[start_idx+i+1];
            targets[N-1]=0;
        }

        // === GPU Forward ===
        cudaMemcpyAsync(d_x,embd,fwd_N*D_MODEL*sizeof(float),cudaMemcpyHostToDevice,stream);
        if(poincare_R>0.0f){
            wubu_cuda_norm(d_x,d_poincare_n,fwd_N,D_MODEL,stream);
            wubu_cuda_exp_map(d_x,d_poincare_n,poincare_R,d_x,fwd_N,D_MODEL,stream);
        }

        float *d_cur=d_x,*d_np=d_norm;
        for(int l=0;l<model.n_layers;l++){
            cudaMemcpyAsync(d_norm_w,model.layers[l].attn_norm_weight,D_MODEL*sizeof(float),cudaMemcpyHostToDevice,stream);
            wubu_cuda_rms_norm(B,fwd_T,D_MODEL,d_cur,d_norm_w,1e-6f,d_np,stream);
            if(pga_enabled||nested_ssm_enabled)
                cudaMemcpy(saved_normed+l*N*D_MODEL,d_np,fwd_N*D_MODEL*sizeof(float),cudaMemcpyDeviceToHost);

            if(model.layers[l].is_ssm){
                if(nested_ssm_enabled){
                    // CPU detour for Nested SSM
                    cudaStreamSynchronize(stream);
                    float *cpu_in=(float*)malloc(fwd_N*D_MODEL*sizeof(float));
                    float *cpu_out=(float*)calloc(fwd_N*D_MODEL,sizeof(float));
                    float *conv_state=NULL;
                    if(d_conv_s[l]){conv_state=(float*)calloc((CONV_KERNEL-1)*CONV_DIM,sizeof(float));}
                    cudaMemcpy(cpu_in,d_np,fwd_N*D_MODEL*sizeof(float),cudaMemcpyDeviceToHost);
                    wubu_nested_ssm_forward(cpu_in,B,fwd_T,&model.layers[l].ssm,&nested_state,
                        conv_state?conv_state:NULL,NULL,cpu_out);
                    cudaMemcpyAsync(d_out,cpu_out,fwd_N*D_MODEL*sizeof(float),cudaMemcpyHostToDevice,stream);
                    free(cpu_in);free(cpu_out);free(conv_state);
                } else if(poincare_R>0.0f){
                    gpu_poincare_ssm_forward(cublas_h,stream,d_np,B,fwd_T,
                        ssm_gpu[l].d_attn_qkv,ssm_gpu[l].d_attn_gate,
                        ssm_gpu[l].d_ssm_beta,ssm_gpu[l].d_ssm_alpha,
                        ssm_gpu[l].d_ssm_dt_bias,ssm_gpu[l].d_ssm_a,
                        ssm_gpu[l].d_ssm_conv1d,ssm_gpu[l].d_ssm_norm,ssm_gpu[l].d_ssm_out,
                        d_ssm_s[l],d_conv_s[l],d_out,
                        d_qkv,d_z,d_beta,d_alpha,d_bs,d_ab,d_gate,
                        d_ci,d_co,d_qc,d_kc,d_vc,d_qn,d_kn,d_do,d_zs,poincare_R);
                } else {
                    gpu_ssm_forward_save(cublas_h,stream,d_np,B,fwd_T,
                        ssm_gpu[l].d_attn_qkv,ssm_gpu[l].d_attn_gate,
                        ssm_gpu[l].d_ssm_beta,ssm_gpu[l].d_ssm_alpha,
                        ssm_gpu[l].d_ssm_dt_bias,ssm_gpu[l].d_ssm_a,
                        ssm_gpu[l].d_ssm_conv1d,ssm_gpu[l].d_ssm_norm,ssm_gpu[l].d_ssm_out,
                        d_ssm_s[l],d_conv_s[l],NULL,d_out,
                        d_qkv,d_z,d_beta,d_alpha,d_bs,d_ab,d_gate,
                        d_ci,d_co,d_qc,d_kc,d_vc,d_qn,d_kn,d_do,d_zs);
                }
            } else {
                if(pga_enabled){
                    cudaStreamSynchronize(stream);
                    float *cpu_in=(float*)malloc(fwd_N*D_MODEL*sizeof(float));
                    float *cpu_out=(float*)calloc(fwd_N*D_MODEL,sizeof(float));
                    float pR=poincare_R>0?poincare_R:10.0f;
                    cudaMemcpy(cpu_in,d_np,fwd_N*D_MODEL*sizeof(float),cudaMemcpyDeviceToHost);
                    wubu_poincare_gqa_forward_save(cpu_in,B,fwd_T,&model.layers[l].gqa,pR,cpu_out,&pga_save[l]);
                    cudaMemcpyAsync(d_out,cpu_out,fwd_N*D_MODEL*sizeof(float),cudaMemcpyHostToDevice,stream);
                    free(cpu_in);free(cpu_out);
                } else {
                    gpu_gqa_forward_save(cublas_h,stream,d_np,B,fwd_T,
                        gqa_gpu[l].d_attn_q,gqa_gpu[l].d_attn_k,gqa_gpu[l].d_attn_v,
                        gqa_gpu[l].d_attn_out_w,gqa_gpu[l].d_q_norm_w,gqa_gpu[l].d_k_norm_w,
                        d_out,d_gqa_Qf,d_gqa_K,d_gqa_V,d_gqa_scr,NULL,NULL,d_gqa_K,d_gqa_scr);
                }
            }

            if(pga_enabled)
                cudaMemcpy(saved_attn_out+l*N*D_MODEL,d_out,fwd_N*D_MODEL*sizeof(float),cudaMemcpyDeviceToHost);
            float a=1.0f; cublasSaxpy(cublas_h,fwd_N*D_MODEL,&a,d_out,1,d_cur,1);

            cudaMemcpyAsync(d_norm_w,model.layers[l].post_attn_norm_weight,D_MODEL*sizeof(float),cudaMemcpyHostToDevice,stream);
            wubu_cuda_rms_norm(B,fwd_T,D_MODEL,d_cur,d_norm_w,1e-6f,d_np,stream);
            // Async D→H copy: sync happens right before MoE uses the data
            cudaMemcpyAsync(saved_normed2+l*N*D_MODEL,d_np,fwd_N*D_MODEL*sizeof(float),cudaMemcpyDeviceToHost,stream);

            // MoE
            if(moe_enabled&&moe_qdata[l].q_gate_exps){
                // Sync normed2 before CPU reads it for MoE
                cudaStreamSynchronize(stream);
                lmoe_t*mc=&moe_cache[l];
                const float*n2=saved_normed2+l*N*D_MODEL;
                float*ffn_out=saved_ffn_out+l*N*D_MODEL;

                if(!mc->dgi){
                    mc->dgi=(float*)malloc((int64_t)D_MODEL*N_EXPERTS*sizeof(float));
                    gguf_dequantize(moe_qdata[l].q_gate_inp,moe_qdata[l].ty_gi,(int64_t)D_MODEL*N_EXPERTS,mc->dgi);
                }

                float*scores=(float*)malloc(fwd_N*N_EXPERTS*sizeof(float));
                if(nested_moe_enabled&&poc_router.loaded){
                    wubu_poincare_router_forward(n2,B,fwd_T,&poc_router,scores);
                } else {
                    for(int s=0;s<fwd_N;s++)
                        for(int e=0;e<N_EXPERTS;e++){
                            double sum=0;
                            for(int k=0;k<D_MODEL;k++)
                                sum+=(double)n2[s*D_MODEL+k]*(double)mc->dgi[e*D_MODEL+k];
                            scores[s*N_EXPERTS+e]=(float)sum;
                        }
                }

                int topk_idx[64*8]; float topk_wt[64*8];
                mc->n_u=0;
                for(int s=0;s<fwd_N;s++){
                    float*sc=scores+s*N_EXPERTS;
                    float mx=sc[0];for(int e=1;e<N_EXPERTS;e++)if(sc[e]>mx)mx=sc[e];
                    float se=0;for(int e=0;e<N_EXPERTS;e++)se+=expf(sc[e]-mx);
                    float inv=1.0f/(se+1e-30f);
                    float sm[256];for(int e=0;e<N_EXPERTS;e++)sm[e]=expf(sc[e]-mx)*inv;
                    int*is=topk_idx+s*N_ACTIVE_EXPTS;
                    float*ws=topk_wt+s*N_ACTIVE_EXPTS;
                    for(int k=0;k<N_ACTIVE_EXPTS;k++){
                        int bi=-1;float bv=-1e30f;
                        for(int e=0;e<N_EXPERTS;e++){
                            int used=0;for(int pk=0;pk<k;pk++)if(is[pk]==e){used=1;break;}
                            if(!used&&sm[e]>bv){bv=sm[e];bi=e;}
                        }
                        is[k]=bi;ws[k]=bv;
                    }
                    float sw=0;for(int k=0;k<N_ACTIVE_EXPTS;k++)sw+=ws[k];
                    if(sw>1e-30f){float iv=1.0f/sw;for(int k=0;k<N_ACTIVE_EXPTS;k++)ws[k]*=iv;}
                    for(int k=0;k<N_ACTIVE_EXPTS;k++){
                        if(is[k]<0)continue;
                        int seen=0;for(int u=0;u<mc->n_u;u++)if(mc->uid[u]==is[k]){seen=1;break;}
                        if(!seen)mc->uid[mc->n_u++]=is[k];
                    }
                }
                free(scores);

                if(!mc->gs&&moe_qdata[l].q_gate_shexp){
                    mc->gs=(float*)malloc((int64_t)D_MODEL*SHARED_D_FF*sizeof(float));
                    mc->us=(float*)malloc((int64_t)D_MODEL*SHARED_D_FF*sizeof(float));
                    mc->ds=(float*)malloc((int64_t)SHARED_D_FF*D_MODEL*sizeof(float));
                    gguf_dequantize(moe_qdata[l].q_gate_shexp,moe_qdata[l].ty_gs,(int64_t)D_MODEL*SHARED_D_FF,mc->gs);
                    gguf_dequantize(moe_qdata[l].q_up_shexp,moe_qdata[l].ty_gs,(int64_t)D_MODEL*SHARED_D_FF,mc->us);
                    gguf_dequantize(moe_qdata[l].q_down_shexp,moe_qdata[l].ty_gs,(int64_t)SHARED_D_FF*D_MODEL,mc->ds);
                }

                moe_weights_t mw;memset(&mw,0,sizeof(mw));
                mw.ffn_gate_inp=mc->dgi;mw.ffn_gate_shexp=mc->gs;
                mw.ffn_up_shexp=mc->us;mw.ffn_down_shexp=mc->ds;mw.loaded=true;
                // Per-expert IQ2_XXS dequant + transpose into interleaved arrays (cached in lmoe_t)
                if(!mc->ge_persist){
                    mc->ge_persist=(float*)calloc((int64_t)N_EXPERTS*D_MODEL*D_FF,sizeof(float));
                    mc->ue_persist=(float*)calloc((int64_t)N_EXPERTS*D_MODEL*D_FF,sizeof(float));
                    mc->de_persist=(float*)calloc((int64_t)N_EXPERTS*D_FF*D_MODEL,sizeof(float));
                }
                float*ge=mc->ge_persist; float*ue=mc->ue_persist; float*de=mc->de_persist;
                int64_t per_exp_elems=(int64_t)D_MODEL*D_FF;
                int64_t per_exp_raw=gguf_raw_size(moe_qdata[l].ty_ge,per_exp_elems);
                int64_t per_exp_raw_dn=gguf_raw_size(moe_qdata[l].ty_ge,(int64_t)D_FF*D_MODEL);
                // Zero only the active expert regions (not full 3GB)
                for(int u=0;u<mc->n_u;u++){
                    int eid=mc->uid[u];
                    memset(ge+(int64_t)eid*D_MODEL*D_FF,0,(int64_t)D_MODEL*D_FF*sizeof(float));
                    memset(ue+(int64_t)eid*D_MODEL*D_FF,0,(int64_t)D_MODEL*D_FF*sizeof(float));
                    memset(de+(int64_t)eid*D_FF*D_MODEL,0,(int64_t)D_FF*D_MODEL*sizeof(float));
                }
                for(int u=0;u<mc->n_u;u++){
                    int eid=mc->uid[u];
                    // Gate/Up: raw dequant → temp[ff][model] → transpose → ge[model][ff]
                    float*temp=(float*)malloc(per_exp_elems*sizeof(float));
                    gguf_dequantize(moe_qdata[l].q_gate_exps+(int64_t)eid*per_exp_raw,
                        moe_qdata[l].ty_ge,per_exp_elems,temp);
                    for(int md=0;md<D_MODEL;md++)for(int ff=0;ff<D_FF;ff++)
                        ge[(int64_t)eid*D_MODEL*D_FF+md*D_FF+ff]=temp[ff*D_MODEL+md];
                    gguf_dequantize(moe_qdata[l].q_up_exps+(int64_t)eid*per_exp_raw,
                        moe_qdata[l].ty_ge,per_exp_elems,temp);
                    for(int md=0;md<D_MODEL;md++)for(int ff=0;ff<D_FF;ff++)
                        ue[(int64_t)eid*D_MODEL*D_FF+md*D_FF+ff]=temp[ff*D_MODEL+md];
                    free(temp);
                    // Down: raw dequant → temp[model][ff] → transpose → de[ff][model]
                    temp=(float*)malloc(per_exp_elems*sizeof(float));
                    gguf_dequantize(moe_qdata[l].q_down_exps+(int64_t)eid*per_exp_raw_dn,
                        moe_qdata[l].ty_ge,per_exp_elems,temp);
                    for(int ff=0;ff<D_FF;ff++)for(int md=0;md<D_MODEL;md++)
                        de[(int64_t)eid*D_FF*D_MODEL+ff*D_MODEL+md]=temp[md*D_FF+ff];
                    free(temp);
                    // Point backprop cache into interleaved arrays
                    mc->eg[u]=ge+(int64_t)eid*D_MODEL*D_FF;
                    mc->eu[u]=ue+(int64_t)eid*D_MODEL*D_FF;
                    mc->ed[u]=de+(int64_t)eid*D_FF*D_MODEL;
                }
                mw.ffn_gate_exps=ge;mw.ffn_up_exps=ue;mw.ffn_down_exps=de;
                wubu_moe_forward(n2,B,fwd_T,&mw,ffn_out, NULL, N_ACTIVE_EXPTS, N_EXPERTS, D_MODEL, D_FF);
                cudaMemcpyAsync(d_np,ffn_out,fwd_N*D_MODEL*sizeof(float),cudaMemcpyHostToDevice,stream);
            } else {
                memcpy(saved_ffn_out+l*N*D_MODEL,saved_normed2+l*N*D_MODEL,fwd_N*D_MODEL*sizeof(float));
            }
            cublasSaxpy(cublas_h,fwd_N*D_MODEL,&a,d_np,1,d_cur,1);
            cudaStreamSynchronize(stream);
        }

        cudaMemcpyAsync(d_norm_w,model.norm_weight,D_MODEL*sizeof(float),cudaMemcpyHostToDevice,stream);
        wubu_cuda_rms_norm(B,fwd_T,D_MODEL,d_cur,d_norm_w,1e-6f,d_np,stream);
        cudaMemcpy(d_cur,d_np,fwd_N*D_MODEL*sizeof(float),cudaMemcpyDeviceToDevice);
        cudaMemcpy(hidden,d_cur,fwd_N*D_MODEL*sizeof(float),cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream);

        // GPU output projection: logits[N,V] = hidden[N,D_MODEL] @ output_weight[V,D_MODEL]^T
        // cublas: C[V,N] = op(A)[D_MODEL,V]^T @ op(B)[D_MODEL,N]
        // opA=T: A[D_MODEL,V]^T → [V,D_MODEL], opB=N: B[D_MODEL,N]
        // Result at d_logits as [V,N] col-major = [N,V] row-major
        { const float alpha=1.0f, beta=0.0f;
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                V, fwd_N, D_MODEL, &alpha,
                d_output_weight, D_MODEL, d_cur, D_MODEL, &beta, d_logits, V);
            cudaMemcpy(logits, d_logits, fwd_N * V * sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Loss
        float loss=0;
        if(use_tst&&n_tst_bags>0){
            tst_compute_mce_loss(logits,tst_targets_buf,B,fwd_T,V,tst_s,&loss);
        } else {
            for(int i=0;i<fwd_N;i++){
                float*log_i=logits+i*V;
                softmax_inplace(log_i,V);
                int tgt=targets[i];if(tgt<0||tgt>=V)tgt=0;
                loss-=logf(fmaxf(log_i[tgt],1e-30f));
            }
            loss/=fwd_N;
        }

        // MTP loss (multi-token prediction): predict t+2 from hidden[t]
        // Reuses already-computed logits (same output_weight, mtp_use_dedicated_embeddings=false)
        float mtp_loss=0;
        if(!use_tst && fwd_N>=2){
            for(int i=0;i<fwd_N-1;i++){
                float*log_i=logits+i*V;  // reuse host-side logits (already softmax'd)
                int tgt=(start_idx+i+2<total_tokens)?tokens[start_idx+i+2]:0;
                if(tgt<0||tgt>=V)tgt=0;
                mtp_loss-=logf(fmaxf(log_i[tgt],1e-30f));
            }
            mtp_loss/=fwd_N-1;
            loss+=0.3f*mtp_loss;  // MTP weight 0.3
        }

        // Gradient w.r.t. logits
        memset(dW,0,(int64_t)D_MODEL*V*sizeof(float));
        float*d_hidden=(float*)calloc(fwd_N*D_MODEL,sizeof(float));
        int *mtp_targets = NULL;
        if(!use_tst && fwd_N>=2){
            mtp_targets=(int*)malloc((fwd_N-1)*sizeof(int));
            for(int i=0;i<fwd_N-1;i++){
                int tgt=(start_idx+i+2<total_tokens)?tokens[start_idx+i+2]:0;
                if(tgt<0||tgt>=V)tgt=0;
                mtp_targets[i]=tgt;
            }
        }
        for(int i=0;i<fwd_N;i++){
            float*log_i=logits+i*V;
            if(!use_tst){
                int tgt=targets[i];if(tgt<0||tgt>=V)tgt=0;
                float mtp_w = (mtp_targets && i<fwd_N-1) ? 0.3f : 0.0f;
                int tmtp = mtp_targets ? mtp_targets[i] : 0;
                for(int j=0;j<V;j++){
                    float dL_dlog=(log_i[j]-(j==tgt?1.0f:0.0f))
                                 + mtp_w*(log_i[j]-(j==tmtp?1.0f:0.0f));
                    for(int k=0;k<D_MODEL;k++){
                        d_hidden[i*D_MODEL+k]+=dL_dlog*output_weight[j*D_MODEL+k];
                        dW[j*D_MODEL+k]+=dL_dlog*hidden[i*D_MODEL+k]/fwd_N;
                    }
                }
            }
        }
        free(mtp_targets);

        // === PGA Backward: propagate gradient through PGA GQA layers ===
        if(pga_enabled){
            float *d_cur_bwd = (float*)malloc(fwd_N*D_MODEL*sizeof(float));
            memcpy(d_cur_bwd, d_hidden, fwd_N*D_MODEL*sizeof(float));

            for(int l=model.n_layers-1; l>=0; l--){
                if(model.layers[l].is_ssm) continue;

                float pR = poincare_R>0 ? poincare_R : 10.0f;
                gqa_layer_weights *gw = &model.layers[l].gqa;
                
                int64_t wQ = D_MODEL * gqa_q_dim * 2;
                int64_t wK = D_MODEL * kv_dim;
                int64_t wV = D_MODEL * kv_dim;
                int64_t wOut = gqa_q_dim * D_MODEL;
                
                float *d_x_l = (float*)calloc(fwd_N*D_MODEL, sizeof(float));
                float *d_q_w = (float*)calloc(wQ, sizeof(float));
                float *d_k_w = (float*)calloc(wK, sizeof(float));
                float *d_v_w = (float*)calloc(wV, sizeof(float));
                float *d_qn_w = (float*)calloc(GQA_HEAD_DIM, sizeof(float));
                float *d_kn_w = (float*)calloc(GQA_HEAD_DIM, sizeof(float));
                float *d_out_w = (float*)calloc(wOut, sizeof(float));
                
                wubu_poincare_gqa_backward(B, fwd_T,
                    saved_normed + l*N*D_MODEL,
                    pga_save[l].Q_norm, pga_save[l].Q_raw,
                    pga_save[l].K_norm, pga_save[l].K_raw,
                    pga_save[l].V,
                    pga_save[l].Q_ball, pga_save[l].K_ball, pga_save[l].V_ball,
                    pga_save[l].gate, pga_save[l].gate_sig,
                    pga_save[l].attn_out_pre_gate,
                    saved_attn_out + l*N*D_MODEL,
                    d_cur_bwd, gw, pR,
                    d_x_l,
                    d_q_w, d_k_w, d_v_w,
                    d_qn_w, d_kn_w, d_out_w);
                
                // Clip + update weights (lr_gqa = lr * 0.001 for PGA stability)
                float lr_gqa = lr * 0.001f;
                float max_g = 0;
                for(int64_t i=0; i<wQ; i++){float v=fabsf(d_q_w[i]);if(v>max_g)max_g=v;}
                for(int64_t i=0; i<wK; i++){float v=fabsf(d_k_w[i]);if(v>max_g)max_g=v;}
                for(int64_t i=0; i<wV; i++){float v=fabsf(d_v_w[i]);if(v>max_g)max_g=v;}
                for(int64_t i=0; i<wOut; i++){float v=fabsf(d_out_w[i]);if(v>max_g)max_g=v;}
                float clip = max_g > 1.0f ? 1.0f/max_g : 1.0f;

                for(int64_t i=0; i<wQ; i++) gw->attn_q_weight[i] -= lr_gqa * d_q_w[i] * clip;
                for(int64_t i=0; i<wK; i++) gw->attn_k_weight[i] -= lr_gqa * d_k_w[i] * clip;
                for(int64_t i=0; i<wV; i++) gw->attn_v_weight[i] -= lr_gqa * d_v_w[i] * clip;
                for(int64_t i=0; i<wOut; i++) gw->attn_output_weight[i] -= lr_gqa * d_out_w[i] * clip;
                for(int i=0; i<GQA_HEAD_DIM; i++){
                    gw->attn_q_norm_weight[i] -= lr_gqa * d_qn_w[i] * clip;
                    gw->attn_k_norm_weight[i] -= lr_gqa * d_kn_w[i] * clip;
                }
                
                memcpy(d_cur_bwd, d_x_l, fwd_N*D_MODEL*sizeof(float));
                
                free(d_x_l);
                free(d_q_w); free(d_k_w); free(d_v_w);
                free(d_qn_w); free(d_kn_w); free(d_out_w);
            }
            free(d_cur_bwd);
        }

        // Weight update
        float max_g_out=0;
        for(int64_t idx=0;idx<(int64_t)D_MODEL*V;idx++){
            float v=fabsf(dW[idx]);if(v>max_g_out)max_g_out=v;
        }
        float clip_out=max_g_out>1.0f?1.0f/max_g_out:1.0f;
        for(int64_t idx=0;idx<(int64_t)D_MODEL*V;idx++)
            output_weight[idx]-=lr*dW[idx]*clip_out;

        // RSGD for Poincaré embedding cache
        if(rsgd_enabled){
            float R = poincare_R>0?poincare_R:0.956f;
            for(int i=0;i<fwd_N;i++){
                int id=tokens[start_idx+i];
                if(id<0||id>=model.vocab_size)continue;
                float*es=poincare_embd_cache+(int64_t)id*D_MODEL;
                float lr_adj=lr*0.1f/(float)fwd_N;
                // Apply RSGD to this single vector
                rsgd_step(es, d_hidden+i*D_MODEL, 1, D_MODEL, lr_adj, R, 1.0f);
            }
        }

        qlearner_step(&ql,loss);
        cudaStreamSynchronize(stream);
        double step_time=now_sec()-t0;
        total_time+=step_time;

        printf("Step %3d: loss=%.4f (%.3fs, %.1f tok/s)%s%s%s%s%s%s\n",
               step+1,loss,step_time,fwd_N/step_time,
               use_tst?" TST":"",rsgd_enabled?" RSGD":"",
               pga_enabled?" PGA":"",nested_ssm_enabled?" NSSM":"",
               nested_moe_enabled?" NMOE":"",
               mtp_loss>0?" MTP":"");
        fflush(stdout);
        free(d_hidden);
    }

    gguf_close(gguf_moe);
    free(tokens);free(output_weight);free(saved_normed);
    free(saved_attn_out);free(saved_normed2);free(saved_ffn_out);
    free(hidden);free(logits);free(dW);
    if(rsgd_enabled){free(poincare_embd_cache);}
    if(nested_ssm_enabled)wubu_nested_ssm_free(&nested_state);
    if(poc_centroids)free(poc_centroids);

    printf("\n=== TRAINING COMPLETE ===\n");
    printf("Avg: %.3fs/step (%.1f tok/s)\n",total_time/n_steps,N/(total_time/n_steps));
    printf("TST=%d RSGD=%d PGA=%d NSSM=%d NMOE=%d\n",
           tst_enabled,rsgd_enabled,pga_enabled,nested_ssm_enabled,nested_moe_enabled);

    // Free PGA save structs
    if(pga_save){
        for(int l=0;l<model.n_layers;l++){
            if(!model.layers[l].is_ssm){
                free(pga_save[l].Q_ball); free(pga_save[l].K_ball); free(pga_save[l].V_ball);
                free(pga_save[l].Q_norm); free(pga_save[l].Q_raw);
                free(pga_save[l].K_norm); free(pga_save[l].K_raw);
                free(pga_save[l].V); free(pga_save[l].gate); free(pga_save[l].gate_sig);
                free(pga_save[l].attn_out_pre_gate);
            }
        }
        free(pga_save);
    }

    // Free MoE cache persistent buffers
    for(int l=0;l<model.n_layers;l++){
        free(moe_cache[l].ge_persist);
        free(moe_cache[l].ue_persist);
        free(moe_cache[l].de_persist);
    }
    // Free GPU project buffers
    wubu_cuda_free(d_output_weight);
    wubu_cuda_free(d_logits);

    return 0;
}
