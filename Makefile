CC = gcc
CXX = g++
NVCC = /usr/local/cuda-13.3/bin/nvcc
CUDA_HOME = /usr/local/cuda-13.3
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize -Wall -Wextra -Wno-unused-parameter -I include -I$(CUDA_HOME)/targets/x86_64-linux/include -fopenmp
# Policy-only gates that must not use aligned AVX stack stores on MinGW.
CFLAGS_NOAVX = $(filter-out -mavx2 -mfma -mavx -mf16c,$(CFLAGS))
LDFLAGS = -lm -fopenmp
NVCC_FLAGS = -O3 -I include -arch=sm_86
CUDA_LIBS = -lcublas -lcudart
CUDA_INC = -I$(CUDA_HOME)/targets/x86_64-linux/include

.PHONY: all clean

# GPU targets are optional: hosts without nvcc get a full CPU build.
NVCC_PRESENT := $(shell test -x $(NVCC) && echo yes)
ifeq ($(NVCC_PRESENT),yes)
GPU_ALL := infer_vision_gpu infer_poincare test_kv_cache train_integrated
else
GPU_ALL :=
endif

all: test_nested_ssm test_nested_ssm_backward load_model test_model test_cpu_timing infer_moe infer_moe_lazy infer_unified infer_vision test_256k infer_vision_text test_poincare_gqa test_tst test_moe_hyperbolic test_mobius_linear test_hyperbolic_output_proj test_chunked_ssm api_server $(GPU_ALL)

api_server: tools/api_server.c
	$(CC) -O2 -g -Wall -I include -o $@ $< -lssl -lcrypto -lm

test_sse: tools/api_server.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Object files
CORE_OBJ = src/wubu_ssm.o src/wubu_ssm_chunked.o src/wubu_mobius.o src/wubu_nested_ssm.o src/wubu_nested_ssm_backward.o src/wubu_moe.o src/wubu_moe_backward.o src/wubu_moe_hyperbolic.o src/wubu_poincare_ssm_backward.o src/wubu_poincare_gqa.o src/wubu_poincare_gqa_backward.o src/wubu_mobius_linear.o src/wubu_hyperbolic_output_proj.o src/wubu_vision.o src/wubu_dims.o src/wubu_dims_gpu_stub.o src/wubu_model_arch.o src/wubu_rotate.o src/wubu_ops.o src/wubu_dense_ffn.o src/gguf_reader.o src/qlearner.o src/rsgd.o src/wubu_tst.o src/dequant_iq2_xxs.o src/quantized_matmul.o src/quantized_dot_generic.o src/wubu_vecdot_dispatch.o src/wubu_vram_plan.o src/wubu_neuron_cluster.o src/wubu_offload_plan.o src/wubu_ssd_row_reuse.o src/wubu_radix_cache.o src/wubu_overlap_sched.o src/wubu_admission.o src/wubu_grammar_fsm.o src/wubu_sampler_chain.o src/wubu_pretoken.o src/wubu_chat_template.o src/wubu_sleep_alloc.o src/wubu_lora_slots.o src/wubu_hicache.o src/wubu_recompute_sched.o src/wubu_expert_trace.o src/wubu_tree_verify.o src/wubu_skip_softmax.o src/wubu_eagle3.o src/wubu_spec_size.o src/wubu_dynact.o src/wubu_isq_plan.o src/wubu_kv_lazy.o src/wubu_blob_store.o src/wubu_fp8_attn.o src/wubu_capacity_sched.o src/wubu_ext_kv.o src/wubu_fp8_kv.o src/wubu_unified_arena.o src/wubu_page_radix.o src/wubu_agent_state.o src/wubu_invalid_kv.o src/wubu_spec_search.o src/wubu_tv_accept.o src/wubu_draft_kv.o src/wubu_sparse_indexer.o src/wubu_lowrank_kv.o src/wubu_sq_agreement.o src/wubu_mx8.o src/wubu_zero_logits.o src/wubu_markov_w4.o src/wubu_hostkv.o src/wubu_bitnest.o src/wubu_readkv.o src/wubu_ssd_expert.o src/wubu_seqmoe.o src/wubu_spice.o src/wubu_rocketkv.o src/wubu_proxy_route.o src/wubu_uno.o src/wubu_dflash.o src/wubu_frspec.o src/wubu_specprefill.o src/wubu_online_sd.o src/wubu_layerskip.o src/wubu_blockverify.o src/wubu_dspark.o src/wubu_specquant.o src/wubu_memspec.o src/wubu_leanstream.o src/wubu_tensorfold.o src/wubu_titan.o src/wubu_strata.o src/wubu_goal.o src/wubu_goal_steer.o src/wubu_goal_resample.o src/wubu_harness_bandit.o src/wubu_ttc_budget.o src/wubu_kv_exact.o src/wubu_swa_reclaim.o src/wubu_spec_budget.o src/wubu_spec_seed.o src/wubu_spec_constrained.o src/wubu_kernel_spec.o src/wubu_kv_qfmt.o src/wubu_kv_fetch.o src/wubu_simd_quant.o src/wubu_diff_sampler.o src/wubu_state_version.o
MODEL_OBJ = src/wubu_model.o $(CORE_OBJ)
CORE_OBJ += src/wubu_ev_depth.o
CORE_OBJ += src/wubu_adapt_len.o
CORE_OBJ += src/wubu_tree_accept.o
CORE_OBJ += src/wubu_prefetch_feedback.o
CORE_OBJ += src/wubu_reclaim_margin.o
CORE_OBJ += src/wubu_replay_ssm.o
CORE_OBJ += src/wubu_keyed_sample.o
CORE_OBJ += src/wubu_accept_table.o
CORE_OBJ += src/wubu_two_level_verify.o
CORE_OBJ += src/wubu_kv_rollback.o
CORE_OBJ += src/wubu_early_expert.o
CORE_OBJ += src/wubu_batch_h2d.o
CORE_OBJ += src/wubu_layer_probe.o
CORE_OBJ += src/wubu_exact_cert.o
CORE_OBJ += src/wubu_dynamic_tree.o
CORE_OBJ += src/wubu_conv_store.o
CORE_OBJ += src/wubu_logits_proc.o
CORE_OBJ += src/wubu_ngram_pool.o
CORE_OBJ += src/wubu_ngram_map.o
CORE_OBJ += src/wubu_candidate_selector.o
CORE_OBJ += src/wubu_index_reuse.o
CORE_OBJ += src/wubu_expert_sub.o
CORE_OBJ += src/wubu_fly_defer.o
CORE_OBJ += src/wubu_pod_plan.o
CORE_OBJ += src/wubu_cond_mem.o
CORE_OBJ += src/wubu_hadamard.o
CORE_OBJ += src/wubu_streamk.o
CORE_OBJ += src/wubu_vc_attn.o
CORE_OBJ += src/wubu_dfly.o
CORE_OBJ += src/wubu_sparse_retrieve.o
CORE_OBJ += src/wubu_suffix_spec.o
CORE_OBJ += src/wubu_token_recycle.o
CORE_OBJ += src/wubu_expert_cache.o
CORE_OBJ += src/wubu_dag_verify.o
CORE_OBJ += src/wubu_relaxed_verify.o
CORE_OBJ += src/wubu_compact_session.o
CORE_OBJ += src/wubu_model_surgery.o
CORE_OBJ += src/wubu_varbit_vq.o
CORE_OBJ += src/wubu_gqa_index.o
CORE_OBJ += src/wubu_expert_stage.o
CORE_OBJ += src/wubu_aot_pack.o
CUDA_OBJ = src/cuda_kernels.o src/gpu_output_proj.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o
GPU_OBJ = src/wubu_model_gpu.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/gpu_moe_kernel.o src/gpu_ssm_recurrence.o
RSGD_OBJ = src/rsgd.o

src/qlearner.o: src/qlearner.c include/qlearner.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ssm.o: src/wubu_ssm.c include/wubu_ssm.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_vecdot_dispatch.o: src/wubu_vecdot_dispatch.c include/wubu_vecdot_dispatch.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_vram_plan.o: src/wubu_vram_plan.c include/wubu_vram_plan.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_neuron_cluster.o: src/wubu_neuron_cluster.c include/wubu_neuron_cluster.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_offload_plan.o: src/wubu_offload_plan.c include/wubu_offload_plan.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ssd_row_reuse.o: src/wubu_ssd_row_reuse.c include/wubu_ssd_row_reuse.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_radix_cache.o: src/wubu_radix_cache.c include/wubu_radix_cache.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_overlap_sched.o: src/wubu_overlap_sched.c include/wubu_overlap_sched.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_admission.o: src/wubu_admission.c include/wubu_admission.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_grammar_fsm.o: src/wubu_grammar_fsm.c include/wubu_grammar_fsm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_sampler_chain.o: src/wubu_sampler_chain.c include/wubu_sampler_chain.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_pretoken.o: src/wubu_pretoken.c include/wubu_pretoken.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_chat_template.o: src/wubu_chat_template.c include/wubu_chat_template.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_sleep_alloc.o: src/wubu_sleep_alloc.c include/wubu_sleep_alloc.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_lora_slots.o: src/wubu_lora_slots.c include/wubu_lora_slots.h include/wubu_lora.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_hicache.o: src/wubu_hicache.c include/wubu_hicache.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_recompute_sched.o: src/wubu_recompute_sched.c include/wubu_recompute_sched.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_expert_trace.o: src/wubu_expert_trace.c include/wubu_expert_trace.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_tree_verify.o: src/wubu_tree_verify.c include/wubu_tree_verify.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_skip_softmax.o: src/wubu_skip_softmax.c include/wubu_skip_softmax.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_eagle3.o: src/wubu_eagle3.c include/wubu_eagle3.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_spec_size.o: src/wubu_spec_size.c include/wubu_spec_size.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_dynact.o: src/wubu_dynact.c include/wubu_dynact.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_isq_plan.o: src/wubu_isq_plan.c include/wubu_isq_plan.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_kv_lazy.o: src/wubu_kv_lazy.c include/wubu_kv_lazy.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_blob_store.o: src/wubu_blob_store.c include/wubu_blob_store.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_fp8_attn.o: src/wubu_fp8_attn.c include/wubu_fp8_attn.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_capacity_sched.o: src/wubu_capacity_sched.c include/wubu_capacity_sched.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ext_kv.o: src/wubu_ext_kv.c include/wubu_ext_kv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_fp8_kv.o: src/wubu_fp8_kv.c include/wubu_fp8_kv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_unified_arena.o: src/wubu_unified_arena.c include/wubu_unified_arena.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_page_radix.o: src/wubu_page_radix.c include/wubu_page_radix.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_agent_state.o: src/wubu_agent_state.c include/wubu_agent_state.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_invalid_kv.o: src/wubu_invalid_kv.c include/wubu_invalid_kv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_spec_search.o: src/wubu_spec_search.c include/wubu_spec_search.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_tv_accept.o: src/wubu_tv_accept.c include/wubu_tv_accept.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_draft_kv.o: src/wubu_draft_kv.c include/wubu_draft_kv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_sparse_indexer.o: src/wubu_sparse_indexer.c include/wubu_sparse_indexer.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_lowrank_kv.o: src/wubu_lowrank_kv.c include/wubu_lowrank_kv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_sq_agreement.o: src/wubu_sq_agreement.c include/wubu_sq_agreement.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mx8.o: src/wubu_mx8.c include/wubu_mx8.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_zero_logits.o: src/wubu_zero_logits.c include/wubu_zero_logits.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_markov_w4.o: src/wubu_markov_w4.c include/wubu_markov_w4.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_hostkv.o: src/wubu_hostkv.c include/wubu_hostkv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_bitnest.o: src/wubu_bitnest.c include/wubu_bitnest.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_readkv.o: src/wubu_readkv.c include/wubu_readkv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ssd_expert.o: src/wubu_ssd_expert.c include/wubu_ssd_expert.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_seqmoe.o: src/wubu_seqmoe.c include/wubu_seqmoe.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_spice.o: src/wubu_spice.c include/wubu_spice.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_rocketkv.o: src/wubu_rocketkv.c include/wubu_rocketkv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_proxy_route.o: src/wubu_proxy_route.c include/wubu_proxy_route.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_uno.o: src/wubu_uno.c include/wubu_uno.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_dflash.o: src/wubu_dflash.c include/wubu_dflash.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_frspec.o: src/wubu_frspec.c include/wubu_frspec.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_specprefill.o: src/wubu_specprefill.c include/wubu_specprefill.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_online_sd.o: src/wubu_online_sd.c include/wubu_online_sd.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_layerskip.o: src/wubu_layerskip.c include/wubu_layerskip.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_blockverify.o: src/wubu_blockverify.c include/wubu_blockverify.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_dspark.o: src/wubu_dspark.c include/wubu_dspark.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_specquant.o: src/wubu_specquant.c include/wubu_specquant.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_memspec.o: src/wubu_memspec.c include/wubu_memspec.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_leanstream.o: src/wubu_leanstream.c include/wubu_leanstream.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_tensorfold.o: src/wubu_tensorfold.c include/wubu_tensorfold.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_titan.o: src/wubu_titan.c include/wubu_titan.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_strata.o: src/wubu_strata.c include/wubu_strata.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ev_depth.o: src/wubu_ev_depth.c include/wubu_ev_depth.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_adapt_len.o: src/wubu_adapt_len.c include/wubu_adapt_len.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_tree_accept.o: src/wubu_tree_accept.c include/wubu_tree_accept.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_prefetch_feedback.o: src/wubu_prefetch_feedback.c include/wubu_prefetch_feedback.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_reclaim_margin.o: src/wubu_reclaim_margin.c include/wubu_reclaim_margin.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_replay_ssm.o: src/wubu_replay_ssm.c include/wubu_replay_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_keyed_sample.o: src/wubu_keyed_sample.c include/wubu_keyed_sample.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_accept_table.o: src/wubu_accept_table.c include/wubu_accept_table.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_two_level_verify.o: src/wubu_two_level_verify.c include/wubu_two_level_verify.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_kv_rollback.o: src/wubu_kv_rollback.c include/wubu_kv_rollback.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_early_expert.o: src/wubu_early_expert.c include/wubu_early_expert.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_batch_h2d.o: src/wubu_batch_h2d.c include/wubu_batch_h2d.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_layer_probe.o: src/wubu_layer_probe.c include/wubu_layer_probe.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_exact_cert.o: src/wubu_exact_cert.c include/wubu_exact_cert.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_dynamic_tree.o: src/wubu_dynamic_tree.c include/wubu_dynamic_tree.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_conv_store.o: src/wubu_conv_store.c include/wubu_conv_store.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_logits_proc.o: src/wubu_logits_proc.c include/wubu_logits_proc.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ngram_pool.o: src/wubu_ngram_pool.c include/wubu_ngram_pool.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ngram_map.o: src/wubu_ngram_map.c include/wubu_ngram_map.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_candidate_selector.o: src/wubu_candidate_selector.c include/wubu_candidate_selector.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_index_reuse.o: src/wubu_index_reuse.c include/wubu_index_reuse.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_expert_sub.o: src/wubu_expert_sub.c include/wubu_expert_sub.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_fly_defer.o: src/wubu_fly_defer.c include/wubu_fly_defer.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_pod_plan.o: src/wubu_pod_plan.c include/wubu_pod_plan.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_cond_mem.o: src/wubu_cond_mem.c include/wubu_cond_mem.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_streamk.o: src/wubu_streamk.c include/wubu_streamk.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_vc_attn.o: src/wubu_vc_attn.c include/wubu_vc_attn.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_dfly.o: src/wubu_dfly.c include/wubu_dfly.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_sparse_retrieve.o: src/wubu_sparse_retrieve.c include/wubu_sparse_retrieve.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_suffix_spec.o: src/wubu_suffix_spec.c include/wubu_suffix_spec.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_token_recycle.o: src/wubu_token_recycle.c include/wubu_token_recycle.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_expert_cache.o: src/wubu_expert_cache.c include/wubu_expert_cache.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_dag_verify.o: src/wubu_dag_verify.c include/wubu_dag_verify.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_relaxed_verify.o: src/wubu_relaxed_verify.c include/wubu_relaxed_verify.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_compact_session.o: src/wubu_compact_session.c include/wubu_compact_session.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_model_surgery.o: src/wubu_model_surgery.c include/wubu_model_surgery.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_varbit_vq.o: src/wubu_varbit_vq.c include/wubu_varbit_vq.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_gqa_index.o: src/wubu_gqa_index.c include/wubu_gqa_index.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_expert_stage.o: src/wubu_expert_stage.c include/wubu_expert_stage.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_aot_pack.o: src/wubu_aot_pack.c include/wubu_aot_pack.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ssm_chunked.o: src/wubu_ssm_chunked.c include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/qlearner.o: src/qlearner.c include/qlearner.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_nested_ssm.o: src/wubu_nested_ssm.c include/wubu_nested_ssm.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_nested_ssm_backward.o: src/wubu_nested_ssm_backward.c include/wubu_nested_ssm.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mobius.o: src/wubu_mobius.c include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe.o: src/wubu_moe.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe_hyperbolic.o: src/wubu_moe_hyperbolic.c include/wubu_moe_hyperbolic.h include/wubu_moe.h include/wubu_mobius.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe_backward.o: src/wubu_moe_backward.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_poincare_ssm_backward.o: src/wubu_poincare_ssm_backward.c include/wubu_ssm.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_poincare_gqa.o: src/wubu_poincare_gqa.c include/wubu_poincare_gqa.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_poincare_gqa_backward.o: src/wubu_poincare_gqa_backward.c include/wubu_poincare_gqa.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mobius_linear.o: src/wubu_mobius_linear.c include/wubu_mobius_linear.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mobius_gyrate.o: src/wubu_mobius_gyrate.c include/wubu_mobius.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_moe_hyperbolic_backward.o: src/wubu_moe_hyperbolic_backward.c include/wubu_moe_hyperbolic.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_hyperbolic_output_proj.o: src/wubu_hyperbolic_output_proj.c include/wubu_hyperbolic_output_proj.h include/wubu_mobius_linear.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_vision.o: src/wubu_vision.c include/wubu_vision.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/gguf_reader.o: src/gguf_reader.c include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_model.o: src/wubu_model.c include/wubu_model.h include/wubu_ssm.h include/wubu_moe.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/cuda_kernels.o: src/cuda_kernels.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_output_proj.o: src/gpu_output_proj.cu include/gpu_output_proj.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/wubu_model_gpu.o: src/wubu_model_gpu.cu include/wubu_model.h include/cuda_kernels.h include/bench.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_quant_matmul.o: src/gpu_quant_matmul.cu include/gpu_quant_matmul.h include/gguf_reader.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_quant_matmul_row_major.o: src/gpu_quant_matmul_row_major.cu include/gpu_quant_matmul.h include/gguf_reader.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_moe_kernel.o: src/gpu_moe_kernel.cu include/gpu_moe_kernel.h include/gguf_reader.h src/iq2xxs_grid_data.inc src/iq3xxs_grid.inc
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_ssm_recurrence.o: src/gpu_ssm_recurrence.cu include/gpu_ssm_recurrence.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/flash_attn_q4_0_opt.o: src/flash_attn_q4_0_opt.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/flash_attn_q4_0_prefill_opt.o: src/flash_attn_q4_0_prefill_opt.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_gemma4.o: src/gpu_gemma4.cu include/gpu_gemma4.h include/wubu_gemma4.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/gpu_gemma4_forward.o: src/gpu_gemma4_forward.cu include/gpu_gemma4.h include/wubu_gemma4.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/rsgd.o: src/rsgd.c include/rsgd.h include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_tst.o: src/wubu_tst.c include/wubu_tst.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/bench.o: src/bench.c include/bench.h include/cuda_kernels.h include/wubu_ssm.h
	$(CC) $(CFLAGS) $(CUDA_INC) -c -o $@ $<

src/dequant_iq2_xxs.o: src/dequant_iq2_xxs.c include/gguf_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Test binaries

test_chunked_ssm: tools/test_chunked_ssm.c src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o,$(CORE_OBJ))
	$(CC) $(CFLAGS) -o $@ tools/test_chunked_ssm.c src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o,$(CORE_OBJ)) $(LDFLAGS)

test_decode_path: tools/test_decode_path.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_nested_ssm: tools/test_nested_ssm.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_nested_ssm_backward: tools/test_nested_ssm_backward.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


test_poincare_gqa: tools/test_poincare_gqa.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_poincare_kv_cache: tools/test_poincare_kv_cache.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_pga_backward: tools/test_pga_backward.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_mobius_linear: tools/test_mobius_linear.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_hyperbolic_output_proj: tools/test_hyperbolic_output_proj.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gpu_layers: tools/test_gpu_layers.c $(CORE_OBJ) $(CUDA_OBJ) src/bench.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

test_gyrate: tools/test_gyrate.c src/wubu_mobius.o src/wubu_mobius_gyrate.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_poincare_router_backward: tools/test_poincare_router_backward.c $(CORE_OBJ) src/wubu_moe_hyperbolic_backward.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_nested_moe_router_backward: tools/test_nested_moe_router_backward.c $(CORE_OBJ) src/wubu_moe_hyperbolic_backward.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_tokenizer: tools/test_tokenizer.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_magnet: tools/test_magnet.c src/wubu_magnet.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_sha1: tools/test_sha1.c src/wubu_sha1.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_torrent: tools/test_torrent.c src/wubu_torrent.c src/wubu_sha1.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_dht: tools/test_dht.c src/wubu_dht.c src/wubu_sha1.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_dht_table: tools/test_dht_table.c src/wubu_dht_table.c src/wubu_dht.c src/wubu_sha1.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_multiaddr: tools/test_multiaddr.c src/wubu_multiaddr.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_cid: tools/test_cid.c src/wubu_cid.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_bitswap: tools/test_bitswap.c src/wubu_bitswap.c src/wubu_cid.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_multistream: tools/test_multistream.c src/wubu_multistream.c src/wubu_cid.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_yamux: tools/test_yamux.c src/wubu_yamux.c
	$(CC) $(CFLAGS) -Iinclude -o $@ $^ $(LDFLAGS)
	./$@

test_model: tools/test_model.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_moe: tools/test_moe.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_moe_hyperbolic: tools/test_moe_hyperbolic.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_full_moe: tools/test_full_moe.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_rope_t2: tools/test_rope_t2.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gen_text: tools/gen_text.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ tools/gen_text.c $(MODEL_OBJ) src/wubu_tokenizer.o $(LDFLAGS)

# CPU-only gen_text (recompiles wubu_model + wubu_moe without GPU_SUPPORT)
gen_text_cpu: CFLAGS_FILTERED = $(filter-out -I/usr/local/cuda-13.1/include,$(CFLAGS))
gen_text_cpu: src/wubu_model_cpu.o src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o,$(CORE_OBJ)) src/wubu_tokenizer.o
	$(CC) $(CFLAGS_FILTERED) -o $@ tools/gen_text.c src/wubu_model_cpu.o src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o,$(CORE_OBJ)) src/wubu_tokenizer.o $(LDFLAGS)
	@echo "gen_text_cpu built (CPU-only, no GPU support)"

src/wubu_model_cpu.o: src/wubu_model.c include/wubu_model.h include/wubu_ssm.h include/wubu_moe.h include/gguf_reader.h
	$(CC) $(CFLAGS) -o $@ -c $<

# CPU-only wubu_moe (no GPU_SUPPORT)
src/wubu_moe_cpu.o: src/wubu_moe.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -o $@ -c $<

run_bos: tools/run_bos.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ tools/run_bos.c $(MODEL_OBJ) $(LDFLAGS)

gen_text_mtp: tools/gen_text_mtp.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS)

gen_text_gpu: tools/gen_text.c $(MODEL_OBJ) src/wubu_tokenizer.o $(CUDA_OBJ) $(GPU_OBJ)
	$(CXX) $(CFLAGS) -DGPU_SUPPORT -o $@ tools/gen_text.c $(MODEL_OBJ) src/wubu_tokenizer.o $(CUDA_OBJ) $(GPU_OBJ) $(LDFLAGS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcublas -lcudart

test_tok_debug: tools/test_tok_debug.c src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ref_dumper: tools/ref_dumper.cpp
	$(CXX) $(CFLAGS) -std=c++17 -I $(HOME)/llama.cpp/include -I $(HOME)/llama.cpp/ggml/include -o $@ $^ $(LDFLAGS) $(HOME)/llama.cpp/build/bin/libllama.so $(HOME)/llama.cpp/build/bin/libggml.so $(HOME)/llama.cpp/build/bin/libggml-cpu.so $(HOME)/llama.cpp/build/bin/libggml-base.so -Wl,-rpath,$(HOME)/llama.cpp/build/bin

ref_dumper_mtp: tools/ref_dumper_mtp.cpp
	$(CXX) $(CFLAGS) -std=c++17 -I $(HOME)/llama.cpp/include -I $(HOME)/llama.cpp/src -I $(HOME)/llama.cpp/ggml/include -o $@ $^ $(LDFLAGS) $(HOME)/llama.cpp/build/bin/libllama.so $(HOME)/llama.cpp/build/bin/libggml.so $(HOME)/llama.cpp/build/bin/libggml-cpu.so $(HOME)/llama.cpp/build/bin/libggml-base.so -Wl,-rpath,$(HOME)/llama.cpp/build/bin

test_quantized_matmul: tools/test_quantized_matmul.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_vec_dot_types: tools/test_vec_dot_types.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_iq_dot: tools/test_iq_dot.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

load_model: tools/load_model_layer.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gpu: tools/test_gpu.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

bench_e2e: tools/bench_e2e.c src/bench.o $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

test_parallel_scan: tools/test_parallel_scan.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

test_fused: tools/test_fused.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

test_fused_vs_old: tools/test_fused_vs_old.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

debug_beta_layout: tools/debug_beta_layout.c src/gguf_reader.o src/dequant_iq2_xxs.o src/cuda_kernels.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

verify_phase26: tools/verify_phase26_fusions.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

train_integrated: tools/train_integrated.c $(MODEL_OBJ) src/wubu_tokenizer.o $(CUDA_OBJ) src/bench.o $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

infer_text: tools/infer_text.c $(MODEL_OBJ) src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

infer_text_gpu: tools/infer_text_gpu.c $(MODEL_OBJ) src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ) $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

test_cuda_kernels: tools/test_cuda_kernels.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

# Compare our logits vs llama.cpp
compare_logits: tools/compare_logits.c $(MODEL_OBJ) src/wubu_tokenizer.o
	g++ -std=c++11 -O2 -I include -I /home/wubu/llama.cpp/include -I /home/wubu/llama.cpp/ggml/include \
		-o $@ $^ \
		-L /home/wubu/llama.cpp/build/bin -lllama -lggml-base -lggml-cpu -lggml \
		-lm -fopenmp -Wl,-rpath,/home/wubu/llama.cpp/build/bin

# Per-layer cos-sim comparison tool
layer_cos_sim: tools/layer_cos_sim.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Hidden state dumper (standalone, no llama deps)
dump_hidden: tools/dump_hidden.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Training & tools
train_stub: tools/train_stub.c
	$(CC) -O0 -g -Wall -Wextra -Wno-unused-parameter -I include -fopenmp -o $@ $< -lm -fopenmp

# Inference engines
infer_moe: tools/infer_moe.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_moe_lazy: tools/infer_moe_lazy.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_unified: tools/infer_unified.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_vision: tools/infer_vision.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_vision_text: tools/infer_vision_text.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_vision_real: tools/test_vision_real.c $(MODEL_OBJ) $(CUDA_OBJ) $(GPU_OBJ)
	$(CXX) $(CFLAGS) -DGPU_SUPPORT -o $@ tools/test_vision_real.c $(MODEL_OBJ) $(CUDA_OBJ) $(GPU_OBJ) $(LDFLAGS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcublas -lcudart
	@echo "test_vision_real built (GPU vision + text)"

infer_vision_text_gpu: tools/infer_vision_text_gpu_nvcc.o $(MODEL_OBJ) $(CUDA_OBJ) src/cuda_vision.o $(GPU_OBJ)
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcublas -lcudart -lstdc++

tools/infer_vision_text_gpu_nvcc.o: tools/infer_vision_text_gpu.cu include/cuda_vision.h include/wubu_vision.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

infer_poincare: tools/infer_poincare.c src/bench.o $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

tailslayer: tools/tailslayer.c $(MODEL_OBJ) src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

infer_vision_gpu: tools/infer_vision_gpu.o $(CORE_OBJ) src/cuda_vision.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

tools/infer_vision_gpu.o: tools/infer_vision_gpu.cu include/cuda_vision.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/cuda_vision.o: src/cuda_vision.cu include/cuda_vision.h include/wubu_vision.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

test_256k: tools/test_256k.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_256k_context: tools/test_256k_context.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_cache: tools/test_kv_cache.c $(CORE_OBJ) $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

tokenize_corpus: tools/tokenize_corpus.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_real: tools/train_real.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_backprop: tools/train_backprop.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_gpu: tools/train_gpu.c src/bench.o $(MODEL_OBJ) $(CUDA_OBJ) src/wubu_tokenizer.o $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

dump_mmproj: tools/dump_mmproj.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

verify_iq2s: tools/verify_iq2s.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_iq2xxs_stride: tools/check_iq2xxs_stride.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

verify_dequant: tools/verify_dequant.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_iq2_dequant: tools/test_iq2_dequant.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_dequant: tools/test_dequant.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_forward: tools/check_forward.c $(MODEL_OBJ) src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_iq2_xxs_dot: tools/test_iq2_xxs_dot.c src/gguf_reader.o src/dequant_iq2_xxs.o src/wubu_moe.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Test runners
test: test_ssm
	./test_ssm

test_nested_ssm_run: test_nested_ssm
	./test_nested_ssm

test_poincare_gqa_run: test_poincare_gqa
	./test_poincare_gqa

test_tst: tools/test_tst.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_tst_run: test_tst
	./test_tst

test_gpu_run: test_gpu
	./test_gpu

bench_e2e_run: bench_e2e
	./bench_e2e

train_stub_run: train_stub
	./train_stub

test_regression: tools/test_regression.c $(MODEL_OBJ) src/wubu_tokenizer.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

test_gpu_poincare: tools/test_gpu_poincare.c $(CORE_OBJ) $(CUDA_OBJ) src/bench.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lstdc++

test_rsgd: tools/test_rsgd.c $(RSGD_OBJ) src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_backward: tools/test_backward.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_bwd_model: tools/test_bwd_model.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_backward_simple: tools/test_backward_simple.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# CPU timing + hedged spec tests (from tailslayer pattern)
test_cpu_timing: tools/test_cpu_timing.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

check_weights: tools/check_weights.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_ssm_a: tools/check_ssm_a.c $(MODEL_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Gemma 4 12B test binaries
test_gemma4: tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o
	$(CC) $(CFLAGS) -o $@ tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o $(LDFLAGS)

test_goal: tools/test_goal.c src/wubu_goal.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_goal_steer: tools/test_goal_steer.c src/wubu_goal_steer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_goal_resample: tools/test_goal_resample.c src/wubu_goal_resample.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-01: runtime per-type vec_dot table + multi-row GEMV.
test_eg01_vecdot: tools/test_eg01_vecdot.c src/wubu_vecdot_dispatch.c src/quantized_dot_generic.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-13: pre-tokenizer family registry + NFC fallback.
test_eg13_pretok: tools/test_eg13_pretok.c src/wubu_pretoken.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-14: Jinja-subset chat template renderer.
test_eg14_template: tools/test_eg14_template.c src/wubu_chat_template.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-15: disaggregated prefill/decode handoff.
test_eg15_pd: tools/test_eg15_pd.c src/wubu_pd_serve.c src/wubu_pd_split.c src/wubu_kv_transfer.c src/wubu_win.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-16: tagged sleep/wake allocator with partial restore.
test_eg16_sleep: tools/test_eg16_sleep.c src/wubu_sleep_alloc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-17: multi-adapter LoRA slots + grouped apply.
test_eg17_lora_slots: tools/test_eg17_lora_slots.c src/wubu_lora_slots.c src/wubu_lora.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-18: three-tier cache with async write-back + deadline prefetch.
test_eg18_hicache: tools/test_eg18_hicache.c src/wubu_hicache.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-19: KV restore-vs-recompute scheduler.
test_eg19_recompute: tools/test_eg19_recompute.c src/wubu_recompute_sched.c
	$(CC) $(CFLAGS_NOAVX) -o $@ $^ $(LDFLAGS)

# EG-20: MoE expert activation trace cache.
test_eg20_expert: tools/test_eg20_expert.c src/wubu_expert_trace.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-21: Medusa-style tree verification.
test_eg21_tree: tools/test_eg21_tree.c src/wubu_tree_verify.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-23: calibrated attention block pruning.
test_eg23_skipsoftmax: tools/test_eg23_skipsoftmax.c src/wubu_skip_softmax.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-24: EAGLE-3 tri-layer fusion + direct prediction.
test_eg24_eagle3: tools/test_eg24_eagle3.c src/wubu_eagle3.c src/wubu_tree_verify.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-25a: Sequoia-style draft-length sizing.
test_eg25_specfam: tools/test_eg25_specfam.c src/wubu_spec_size.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-26: per-token dynamic activation quantization.
test_eg26_dynact: tools/test_eg26_dynact.c src/wubu_dynact.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-27: ISQ quantize-on-CPU before device load.
test_eg27_isq: tools/test_eg27_isq.c src/wubu_isq_plan.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-28: lazy per-layer KV quantization.
test_eg28_kvquant: tools/test_eg28_kvquant.c src/wubu_kv_lazy.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-29: SHA-256 content-addressed blob store.
test_eg29_blob: tools/test_eg29_blob.c src/wubu_blob_store.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-30: FP8 attention with F32 accumulation.
test_eg30_fp8attn: tools/test_eg30_fp8attn.c src/wubu_fp8_attn.c src/wubu_fp8.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-32a: capacity + micro-batch scheduler.
test_eg32_capacity: tools/test_eg32_capacity.c src/wubu_capacity_sched.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-35: UMA-corrected KV sizing after warmup.
test_eg35_extkv: tools/test_eg35_extkv.c src/wubu_ext_kv.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-36a: OCP FP8 KV round-trip.
test_eg36_fp8kv: tools/test_eg36_fp8kv.c src/wubu_fp8_kv.c src/wubu_fp8.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-37: unified arena with compaction and whole-request admission.
test_eg37_arena: tools/test_eg37_arena.c src/wubu_unified_arena.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-38: page-owning radix with draft-state rollback.
test_eg38_pages: tools/test_eg38_pages.c src/wubu_page_radix.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-39: persistent per-agent state with prefix-digest resume.
test_eg39_agentstate: tools/test_eg39_agentstate.c src/wubu_agent_state.c src/wubu_blob_store.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-40: invalidated-KV draft verification with preemption.
test_eg40_invalidkv: tools/test_eg40_invalidkv.c src/wubu_invalid_kv.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-41: hardware/memory-aware speculative config search.
test_eg41_specsearch: tools/test_eg41_specsearch.c src/wubu_spec_search.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-42: TV acceptance identity and calibration.
test_eg42_tv: tools/test_eg42_tv.c src/wubu_tv_accept.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-43: K4V4 quantized draft KV.
test_eg43_draftkv: tools/test_eg43_draftkv.c src/wubu_draft_kv.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-44: sparse-attention indexer accuracy gate.
test_eg44_indexer: tools/test_eg44_indexer.c src/wubu_sparse_indexer.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-45a: low-rank key approximation.
test_eg45_lowrank: tools/test_eg45_lowrank.c src/wubu_lowrank_kv.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-101: SQ build agreement/KL gate.
test_eg101_sq_agreement: tools/test_eg101_sq_agreement.c src/wubu_sq_agreement.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-102: MX8 block microscaling.
test_eg102_mx8: tools/test_eg102_mx8.c src/wubu_mx8.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-103: zero-logits D2H handoff.
test_eg103_zero_logits: tools/test_eg103_zero_logits.c src/wubu_zero_logits.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-104: W=4 Markov beam-tree branch selection.
test_eg104_markov_tree: tools/test_eg104_markov_tree.c src/wubu_markov_w4.c src/wubu_tree_verify.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-47: INT8 host-tier KV row-local record.
test_eg47_hostkv: tools/test_eg47_hostkv.c src/wubu_hostkv.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-48: nested W4 draft inside W8 target.
test_eg48_bitnest: tools/test_eg48_bitnest.c src/wubu_bitnest.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-49: progressive KV code: 8-bit store / 4-bit fetch.
test_eg49_readkv: tools/test_eg49_readkv.c src/wubu_readkv.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-50: concurrent SSD expert reads + async H2D.
test_eg50_ssdexpert: tools/test_eg50_ssdexpert.c src/wubu_ssd_expert.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-52: sequence-model expert residency.
test_eg52_seqmoe: tools/test_eg52_seqmoe.c src/wubu_seqmoe.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-51: SPICE confidence routing + accuracy-gated surrogate.
test_eg51_spice: tools/test_eg51_spice.c src/wubu_spice.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-53: RocketKV training-free prompt-KV eviction.
test_eg53_rocketkv: tools/test_eg53_rocketkv.c src/wubu_rocketkv.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-54: block-sparse proxy routes.
test_eg54_proxy: tools/test_eg54_proxy.c src/wubu_proxy_route.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-55: UNO gated-LoRA drafter + lossless Psi-Spec sampling.
test_eg55_uno: tools/test_eg55_uno.c src/wubu_uno.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-56: DFlash block-diffusion drafting.
test_eg56_dflash: tools/test_eg56_dflash.c src/wubu_dflash.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-57: FR-Spec frequency-ranked draft head.
test_eg57_frspec: tools/test_eg57_frspec.c src/wubu_frspec.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-58: SpecPrefill draft-attention prompt compression.
test_eg58_specprefill: tools/test_eg58_specprefill.c src/wubu_specprefill.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-59: online speculative decoding.
test_eg59_online: tools/test_eg59_online.c src/wubu_online_sd.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-60: LayerSkip layer dropout + exit-cache remainder.
test_eg60_layerskip: tools/test_eg60_layerskip.c src/wubu_layerskip.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-61: block verification with distribution-exact correction.
test_eg61_blockverify: tools/test_eg61_blockverify.c src/wubu_blockverify.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-62: DSpark correction/confidence/budget.
test_eg62_dspark: tools/test_eg62_dspark.c src/wubu_dspark.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-63: SpecQuant precision routing.
test_eg63_specquant: tools/test_eg63_specquant.c src/wubu_specquant.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-64: MemSpec memory-aware draft scheduling.
test_eg64_memspec: tools/test_eg64_memspec.c src/wubu_memspec.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-65: LeanStream partial-result streaming.
test_eg65_leanstream: tools/test_eg65_leanstream.c src/wubu_leanstream.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-66a: TensorFold disk prefix snapshot round-trip.
test_eg66_tensorfold: tools/test_eg66_tensorfold.c src/wubu_tensorfold.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-67: titan-engine tiered expert planning.
test_eg67_titan: tools/test_eg67_titan.c src/wubu_titan.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-68: Strata host/launch/split-KV gate.
test_eg68_strata: tools/test_eg68_strata.c src/wubu_strata.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-87: expected-value speculative depth gate.
test_eg87_ev_depth: tools/test_eg87_ev_depth.c src/wubu_ev_depth.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-88: adaptive candidate length gate.
test_eg88_adapt_len: tools/test_eg88_adapt_len.c src/wubu_adapt_len.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-91: pointer-jumping tree acceptance gate.
test_eg91_tree_accept: tools/test_eg91_tree_accept.c src/wubu_tree_accept.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-95: closed-loop prefetch depth gate.
test_eg95_prefetch_feedback: tools/test_eg95_prefetch_feedback.c src/wubu_prefetch_feedback.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-97: overflow-class memory margin gate.
test_eg97_reclaim_margin: tools/test_eg97_reclaim_margin.c src/wubu_reclaim_margin.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-69: checkpoint+input-buffer SSM rollback gate.
test_eg69_replayssm: tools/test_eg69_replayssm.c src/wubu_replay_ssm.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-89: position-keyed exponential-race sampling gate.
test_eg89_keyed_sample: tools/test_eg89_keyed_sample.c src/wubu_keyed_sample.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-94: empirical tree-ordering acceptance table gate.
test_eg94_accept_table: tools/test_eg94_accept_table.c src/wubu_accept_table.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-92: two-level quantized pre-verify then exact commit gate.
test_eg92_two_level_verify: tools/test_eg92_two_level_verify.c src/wubu_two_level_verify.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-93: bounded-checkpoint KV prefix rollback gate.
test_eg93_kv_rollback: tools/test_eg93_kv_rollback.c src/wubu_kv_rollback.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-96: cross-layer early expert read gate.
test_eg96_early_expert: tools/test_eg96_early_expert.c src/wubu_early_expert.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-98: batched expert H2D upload gate.
test_eg98_batch_h2d: tools/test_eg98_batch_h2d.c src/wubu_batch_h2d.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-100: measured layer probe pricing gate.
test_eg100_layer_probe: tools/test_eg100_layer_probe.c src/wubu_layer_probe.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-70: output-exactness certificate gate.
test_eg70_cert: tools/test_eg70_cert.c src/wubu_exact_cert.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-90: EAGLE-2 confidence-shaped dynamic tree gate.
test_eg90_dynamic_tree: tools/test_eg90_dynamic_tree.c src/wubu_dynamic_tree.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-99: content-addressed conversation blocks gate.
test_eg99_conv_blocks: tools/test_eg99_conv_blocks.c src/wubu_conv_store.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-34: typical-p + watermark logits processors gate.
test_eg34_logits: tools/test_eg34_logits.c src/wubu_logits_proc.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-71: shared rolling-hash n-gram pool gate.
test_eg71_ngram_pool: tools/test_eg71_ngram_pool.c src/wubu_ngram_pool.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-72: k4v n-gram map gate.
test_eg72_ngram_map: tools/test_eg72_ngram_map.c src/wubu_ngram_map.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-74: candidate selector with p-min cut gate.
test_eg74_selector: tools/test_eg74_selector.c src/wubu_candidate_selector.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-76: cross-layer DSA index reuse gate.
test_eg76_index_reuse: tools/test_eg76_index_reuse.c src/wubu_index_reuse.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-78: constant novice expert substitution gate.
test_eg78_expert_sub: tools/test_eg78_expert_sub.c src/wubu_expert_sub.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-80: entropy-gated rejection deferral gate.
test_eg80_fly: tools/test_eg80_fly.c src/wubu_fly_defer.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-83: POD split-K workspace and aux reuse gate.
test_eg83_pod_plan: tools/test_eg83_pod_plan.c src/wubu_pod_plan.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-82: Engram conditional-memory gate.
test_eg82_engram: tools/test_eg82_engram.c src/wubu_cond_mem.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-85: stateless Sylvester-Hadamard rotation gate (existing module).
test_eg85_hadamard: tools/test_eg85_hadamard.c src/wubu_hadamard.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-84: AWQ uint4 group scale/zero-point gate.
test_eg84_awq: tools/test_eg84_awq.c src/wubu_awq.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-79: Stream-K partition and partial-sum fixup gate.
test_eg79_streamk: tools/test_eg79_streamk.c src/wubu_streamk.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-77: V-residual attention gate.
test_eg77_vc_attn: tools/test_eg77_vc_attn.c src/wubu_vc_attn.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-81: checkpoint-native shared-KV drafting gate.
test_eg81_dfly: tools/test_eg81_dfly.c src/wubu_dfly.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-75: sparse-KV retrieval lifecycle gate.
test_eg75_sparse_retrieve: tools/test_eg75_sparse_retrieve.c src/wubu_sparse_retrieve.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-73: suffix decoding from finished-request corpus gate.
test_eg73_suffix_spec: tools/test_eg73_suffix_spec.c src/wubu_suffix_spec.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-25b: token recycling from verified chains.
test_eg25_recycle: tools/test_eg25_recycle.c src/wubu_token_recycle.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-36b: MoE expert LRFU/ClassLRU cache gate.
test_eg36_expert_cache: tools/test_eg36_expert_cache.c src/wubu_expert_cache.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-25c: DAG verification of shared draft prefixes.
test_eg25_dag_verify: tools/test_eg25_dag_verify.c src/wubu_dag_verify.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-25d: relaxed speculative verification gate.
test_eg25_relaxed: tools/test_eg25_relaxed.c src/wubu_relaxed_verify.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-32b: long-lived compact session gate.
test_eg32_compact_session: tools/test_eg32_compact_session.c src/wubu_compact_session.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-39b: model surgery planner gate.
test_eg39_surgery: tools/test_eg39_surgery.c src/wubu_model_surgery.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-45b: variable-bit packed VQ gate.
test_eg45_varbit_vq: tools/test_eg45_varbit_vq.c src/wubu_varbit_vq.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-44b: GQA grouping for sparse index gate.
test_eg44_gqa_index: tools/test_eg44_gqa_index.c src/wubu_gqa_index.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-45c: coordinated expert layer staging gate.
test_eg45_expert_stage: tools/test_eg45_expert_stage.c src/wubu_expert_stage.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-66c: AOT kernel pack selector gate.
test_eg66_aot_pack: tools/test_eg66_aot_pack.c src/wubu_aot_pack.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-12: composable sampler chain.
test_eg12_sampler: tools/test_eg12_sampler.c src/wubu_sampler_chain.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-11: grammar automaton with batched masks and jump-forward.
test_eg11_grammar: tools/test_eg11_grammar.c src/wubu_grammar_fsm.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-10: shared admission validator with token budget.
test_eg10_admit: tools/test_eg10_admit.c src/wubu_admission.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-09: overlapping scheduler with a one-step deferred result queue.
test_eg09_overlap: tools/test_eg09_overlap.c src/wubu_overlap_sched.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-08: RadixAttention-style prefix tree with LRU eviction.
test_eg08_radix: tools/test_eg08_radix.c src/wubu_radix_cache.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-07: SSD sliding-window row reuse with coalesced reads.
test_eg07_row_reuse: tools/test_eg07_row_reuse.c src/wubu_ssd_row_reuse.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-06: FlexGen-style placement solver + zigzag schedule.
test_eg06_plan: tools/test_eg06_plan.c src/wubu_offload_plan.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-05: neuron-cluster predictor + cluster I/O accounting.
test_eg05_neuron: tools/test_eg05_neuron.c src/wubu_neuron_cluster.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-04: attention sinks pin the first N KV positions in SWA reclaim.
test_eg04_attention_sink: tools/test_eg04_attention_sink.c src/wubu_swa_reclaim.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-03: metadata-first VRAM planner.
test_eg03_vram_plan: tools/test_eg03_vram_plan.c src/wubu_vram_plan.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# EG-02: GGUF tokenizer metadata completeness. Fixture generated by
# tools/make_tok_meta_fixture.py; gate runs from the repo root.
test_eg02_tok_meta: tools/test_eg02_tok_meta.c src/wubu_gguf_tokenizer.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Engine aggregation gates: vLLM-style iteration-level batching and
# SGLang-style cross-request prefix KV reuse. Sources are listed directly so
# the gate does not depend on an implicit object rule.
test_contbatch: tools/test_continuous_batching.c src/wubu_scheduler.c src/wubu_continuous_batching.c src/wubu_contbatch.c src/wubu_paged_kv.c src/wubu_prefix_cache.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_prefix_reuse: tools/test_prefix_reuse.c src/wubu_prefix_cache.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_harness_bandit: tools/test_harness_bandit.c src/wubu_harness_bandit.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_ttc_budget: tools/test_ttc_budget.c src/wubu_ttc_budget.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_alloc_exact: tools/test_kv_alloc_exact.c src/wubu_kv_exact.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_swa_reclaim: tools/test_swa_reclaim.c src/wubu_swa_reclaim.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_spec_budget: tools/test_spec_budget.c src/wubu_spec_budget.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_spec_seed: tools/test_spec_seed.c src/wubu_spec_seed.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_spec_constrained: tools/test_spec_constrained.c src/wubu_spec_constrained.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kernel_identity: tools/test_kernel_identity.c src/wubu_kernel_spec.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_qfmt: tools/test_kv_qfmt.c src/wubu_kv_qfmt.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_kv_fetch: tools/test_kv_fetch.c src/wubu_kv_fetch.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_simd_quant: tools/test_simd_quant.c src/wubu_simd_quant.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_diffusion_sampler: tools/test_diffusion_sampler.c src/wubu_diff_sampler.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_state_version: tools/test_state_version.c src/wubu_state_version.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gemma4_gpu: tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o $(LDFLAGS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcublas -lcudart -lstdc++

gen_text_gemma4: tools/gen_text_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o
	$(CC) $(CFLAGS) -o $@ tools/gen_text_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o $(LDFLAGS)

gen_text_gemma4_gpu: tools/gen_text_gemma4.c src/wubu_gemma4_model.o src/gguf_reader.o src/quantized_matmul.o src/quantized_dot_generic.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ tools/gen_text_gemma4.c src/wubu_gemma4_model.o src/gguf_reader.o src/quantized_matmul.o src/quantized_dot_generic.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o $(LDFLAGS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcublas -lcudart -lstdc++

clean:
	rm -f test_nested_ssm test_poincare_ssm test_poincare_gqa load_model test_model test_gpu tokenize_corpus test_moe test_moe_hyperbolic train_real bench_e2e verify_iq2s inspect_iq2s inspect_model train_backprop train_gpu test_gpu_poincare test_rsgd test_backward test_cpu_timing infer_moe infer_moe_lazy infer_unified infer_vision infer_poincare infer_vision_gpu test_256k test_kv_cache test_tst test_nested_moe_router_backward tailslayer test_iq2_dequant test_iq2_xxs_dot test_gemma4 test_gemma4_gpu gen_text_gemma4 gen_text_gemma4_gpu src/*.o tools/*.o
