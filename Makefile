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
CORE_OBJ = src/wubu_ssm.o src/wubu_ssm_chunked.o src/wubu_mobius.o src/wubu_nested_ssm.o src/wubu_nested_ssm_backward.o src/wubu_moe.o src/wubu_moe_backward.o src/wubu_moe_hyperbolic.o src/wubu_poincare_ssm_backward.o src/wubu_poincare_gqa.o src/wubu_poincare_gqa_backward.o src/wubu_mobius_linear.o src/wubu_hyperbolic_output_proj.o src/wubu_vision.o src/wubu_dims.o src/wubu_dims_gpu_stub.o src/wubu_model_arch.o src/wubu_rotate.o src/wubu_ops.o src/wubu_dense_ffn.o src/gguf_reader.o src/qlearner.o src/rsgd.o src/wubu_tst.o src/dequant_iq2_xxs.o src/quantized_matmul.o src/quantized_dot_generic.o src/wubu_vecdot_dispatch.o src/wubu_vram_plan.o src/wubu_neuron_cluster.o src/wubu_offload_plan.o src/wubu_ssd_row_reuse.o src/wubu_radix_cache.o src/wubu_overlap_sched.o src/wubu_admission.o src/wubu_grammar_fsm.o src/wubu_sampler_chain.o src/wubu_pretoken.o src/wubu_chat_template.o src/wubu_sleep_alloc.o src/wubu_lora_slots.o src/wubu_hicache.o src/wubu_recompute_sched.o src/wubu_expert_trace.o src/wubu_tree_verify.o src/wubu_skip_softmax.o src/wubu_eagle3.o src/wubu_spec_size.o src/wubu_dynact.o src/wubu_isq_plan.o src/wubu_kv_lazy.o src/wubu_blob_store.o src/wubu_fp8_attn.o src/wubu_capacity_sched.o src/wubu_ext_kv.o src/wubu_fp8_kv.o src/wubu_unified_arena.o src/wubu_goal.o src/wubu_goal_steer.o src/wubu_goal_resample.o src/wubu_harness_bandit.o src/wubu_ttc_budget.o src/wubu_kv_exact.o src/wubu_swa_reclaim.o src/wubu_spec_budget.o src/wubu_spec_seed.o src/wubu_spec_constrained.o src/wubu_kernel_spec.o src/wubu_kv_qfmt.o src/wubu_kv_fetch.o src/wubu_simd_quant.o src/wubu_diff_sampler.o src/wubu_state_version.o
MODEL_OBJ = src/wubu_model.o $(CORE_OBJ)
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
