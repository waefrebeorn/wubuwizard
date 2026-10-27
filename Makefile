CC = gcc
CXX = g++
NVCC = /usr/local/cuda-13.3/bin/nvcc
CUDA_HOME = /usr/local/cuda-13.3
CFLAGS = -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize -Wall -Wextra -Wno-unused-parameter -I include -I$(CUDA_HOME)/targets/x86_64-linux/include -fopenmp
LDFLAGS = -lm -fopenmp
NVCC_FLAGS = -O3 -I include -arch=sm_86
CUDA_LIBS = -lcublas -lcudart
CUDA_INC = -I$(CUDA_HOME)/targets/x86_64-linux/include

.PHONY: all clean

all: test_nested_ssm test_nested_ssm_backward load_model test_model test_cpu_timing infer_moe infer_moe_lazy infer_unified infer_vision infer_poincare infer_vision_gpu test_256k test_kv_cache infer_vision_text test_poincare_gqa test_tst test_moe_hyperbolic test_mobius_linear test_hyperbolic_output_proj train_integrated test_chunked_ssm api_server

api_server: tools/api_server.c
	$(CC) -O2 -g -Wall -I include -o $@ $< -lssl -lcrypto -lm

# Object files
CORE_OBJ = src/wubu_ssm.o src/wubu_ssm_chunked.o src/wubu_mobius.o src/wubu_nested_ssm.o src/wubu_nested_ssm_backward.o src/wubu_moe.o src/wubu_moe_backward.o src/wubu_moe_hyperbolic.o src/wubu_poincare_ssm_backward.o src/wubu_poincare_gqa.o src/wubu_poincare_gqa_backward.o src/wubu_mobius_linear.o src/wubu_hyperbolic_output_proj.o src/wubu_vision.o src/gguf_reader.o src/qlearner.o src/rsgd.o src/wubu_tst.o src/dequant_iq2_xxs.o src/quantized_matmul.o src/quantized_dot_generic.o
MODEL_OBJ = src/wubu_model.o $(CORE_OBJ)
CUDA_OBJ = src/cuda_kernels.o src/gpu_output_proj.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o
GPU_OBJ = src/wubu_model_gpu.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/gpu_moe_kernel.o src/gpu_ssm_recurrence.o
RSGD_OBJ = src/rsgd.o

src/qlearner.o: src/qlearner.c include/qlearner.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_ssm.o: src/wubu_ssm.c include/wubu_ssm.h include/wubu_mobius.h
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

test_gemma4_gpu: tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o $(LDFLAGS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcublas -lcudart -lstdc++

gen_text_gemma4: tools/gen_text_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o
	$(CC) $(CFLAGS) -o $@ tools/gen_text_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o $(LDFLAGS)

gen_text_gemma4_gpu: tools/gen_text_gemma4.c src/wubu_gemma4_model.o src/gguf_reader.o src/quantized_matmul.o src/quantized_dot_generic.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ tools/gen_text_gemma4.c src/wubu_gemma4_model.o src/gguf_reader.o src/quantized_matmul.o src/quantized_dot_generic.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o $(LDFLAGS) -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcublas -lcudart -lstdc++

clean:
	rm -f test_nested_ssm test_poincare_ssm test_poincare_gqa load_model test_model test_gpu tokenize_corpus test_moe test_moe_hyperbolic train_real bench_e2e verify_iq2s inspect_iq2s inspect_model train_backprop train_gpu test_gpu_poincare test_rsgd test_backward test_cpu_timing infer_moe infer_moe_lazy infer_unified infer_vision infer_poincare infer_vision_gpu test_256k test_kv_cache test_tst test_nested_moe_router_backward tailslayer test_iq2_dequant test_iq2_xxs_dot test_gemma4 test_gemma4_gpu gen_text_gemma4 gen_text_gemma4_gpu src/*.o tools/*.o
