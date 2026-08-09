CC = gcc
CXX = g++
# ccache for faster rebuilds (research 066-C5). CCACHE=0 to disable.
# Uses CCACHE=1 (default) only if ccache binary exists; falls back to bare CC.
ifneq ($(CCACHE),0)
  ifneq ($(shell command -v ccache 2>/dev/null),)
    CC := ccache $(CC)
    CXX := ccache $(CXX)
  endif
endif
WUBU_VERSION = 1.0.0
# CUDA layout on this WSL2 box (2026-08-03, verified): the GPU is an
# NVIDIA GeForce RTX 4050 Laptop (sm_89, 6GB) exposed via WSL /dev/dxg.
# nvcc lives in the partial .run install at /usr/local/cuda-13.1/bin;
# its include/ and lib64/ are SYMLINKED to the apt toolkit
# (/usr/include/cuda + /usr/lib/x86_64-linux-gnu) -- see
# tools/check_cuda_env.sh. The WSL GPU passthrough libcuda.so.1 lives in
# /usr/lib/wsl/lib and is NOT on the default linker path, so it is added
# as an rpath so GPU binaries find it at load time without the caller
# exporting LD_LIBRARY_PATH.
NVCC = $(or $(shell which nvcc 2>/dev/null),/usr/bin/nvcc)
CUDA_HOME = $(shell d=$(NVCC); d=$${d%/*}; d=$${d%/*}; echo $$d)
CUDA_INC = -I$(CUDA_HOME)/include
CUDA_LIBDIR = $(shell if [ -d $(CUDA_HOME)/lib/x86_64-linux-gnu ]; then echo $(CUDA_HOME)/lib/x86_64-linux-gnu; else echo $(CUDA_HOME)/lib64; fi)
WSL_LIB = /usr/lib/wsl/lib
CFLAGS = -O3 -march=native -funroll-loops -fno-fast-math -ffp-contract=fast -ftree-vectorize -Wall -Wextra -Wno-unused-parameter -I include $(CUDA_INC) -fopenmp
LDFLAGS = -lm -fopenmp -L$(CUDA_LIBDIR) -L$(WSL_LIB) -Wl,-rpath,$(WSL_LIB) -lcudart -lcublas -lpthread -lssl -lcrypto
NVCC_FLAGS = -O3 -I include $(CUDA_INC) -arch=sm_89
CUDA_INCS = $(CUDA_INC)
CUDA_LIBS = -L$(CUDA_LIBDIR) -lcublas -lcudart
CUDA_LIB = -L$(CUDA_LIBDIR) -lcudart

# ---- CUDA sanity guard (so a broken CUDA layout can never silently
# produce CPU-only binaries again). Runs before any GPU target. ----
cuda_check:
	@if [ ! -x "$(NVCC)" ]; then echo "FATAL: nvcc not found ($(NVCC)). Fix: apt install nvidia-cuda-toolkit OR restore /usr/local/cuda-13.1"; exit 1; fi
	@if [ ! -d "$(CUDA_HOME)/include" ]; then echo "FATAL: CUDA headers missing ($(CUDA_HOME)/include). Fix: sudo ln -s /usr/include/cuda $(CUDA_HOME)/include"; exit 1; fi
	@if [ ! -f "$(CUDA_LIBDIR)/libcublas.so" ] && [ ! -f "$(CUDA_LIBDIR)/libcublas.so.12" ]; then echo "FATAL: libcublas missing in $(CUDA_LIBDIR). Fix: sudo ln -s /usr/lib/x86_64-linux-gnu $(CUDA_HOME)/lib64"; exit 1; fi
	@if [ ! -f "$(WSL_LIB)/libcuda.so.1" ]; then echo "FATAL: WSL GPU passthrough missing ($(WSL_LIB)/libcuda.so.1). Is the Windows NVIDIA driver installed?"; exit 1; fi
	@echo "CUDA OK: nvcc=$(NVCC) inc=$(CUDA_HOME)/include libdir=$(CUDA_LIBDIR) gpu=$(WSL_LIB)/libcuda.so.1"

.PHONY: all clean help ninja test_all

# EXTRA_OBJ — the 39 orphaned-but-clean modules (the module registry,
# MODULES.md): compiled into the build so nothing sits in the toolbox.
# Collision-free vs CORE_OBJ (verified by nm 2026-08-09).
EXTRA_OBJ = src/gaad_nesting_llm.o src/lfm2_attn.o src/lfm2_conv.o src/lfm2_ffn.o src/lfm2_forward.o src/lfm2_load.o src/lfm2_math.o src/thread_pool.o src/wubu_agi.o src/wubu_ambig.o src/wubu_bear_mlp.o src/wubu_colonel.o src/wubu_credit_sft.o src/wubu_dbstate.o src/wubu_dedup.o src/wubu_dense_ffn.o src/wubu_dsv4_layer.o src/wubu_epcap.o src/wubu_eval.o src/wubu_fmt.o src/wubu_gaad.o src/wubu_h3_norm.o src/wubu_kernel_budget.o src/wubu_masked_ce.o src/wubu_mix.o src/wubu_mmrope.o src/wubu_passk.o src/wubu_patch_embed.o src/wubu_priority.o src/wubu_recency.o src/wubu_rollout.o src/wubu_seed.o src/wubu_spawn_win.o src/wubu_traj_grpo.o src/wubu_traj_sft.o src/wubu_user_sim.o src/wubu_uuid.o src/wubu_vision_moondream.o src/wubu_width.o src/wubu_win.o src/quantized_matmul_fixed.o

# Build every EXTRA module (objects only — their tools link them on demand)
$(EXTRA_OBJ): src/%.o: src/%.c
	$(CC) $(CFLAGS) -I include -c -o $@ $<

# the module-registry gate: all EXTRA objects compile
.PHONY: module-registry
module-registry: $(EXTRA_OBJ)
	@echo "=== MODULE REGISTRY: $(words $(EXTRA_OBJ)) EXTRA modules compiled ==="

all: module-registry test_nested_ssm test_nested_ssm_backward load_model test_model test_cpu_timing test_model_adapter infer_moe infer_moe_lazy infer_unified infer_poincare test_256k test_kv_cache test_poincare_gqa test_tst test_moe_hyperbolic test_mobius_linear test_hyperbolic_output_proj train_integrated test_chunked_ssm api_server

# Generate compile_commands.json for clangd / IDE / agent tooling (research 066-C1).
compile_commands.json:
	@python3 tools/gen_compile_commands.py

# Generate symbol index for agents (research 066-C3).
symbols.json:
	@python3 tools/gen_symbols.py > docs/symbols.json
	@echo "Wrote $$(python3 -c "import json;print(len(json.load(open('docs/symbols.json'))))") symbols"

# Generate build.ninja from compile_commands.json (research 066-C5).
ninja: compile_commands.json
	@python3 tools/gen_ninja.py


help:
	@echo "wubuwizard build system (WUBU_VERSION=$(WUBU_VERSION))"
	@echo ""
	@echo "  all          - build all tools and test binaries"
	@echo "  test_all     - build + run the full test gate"
	@echo "  clean        - remove all build artifacts"
	@echo "  help         - show this help"
	@echo ""
	@echo "  Individual test targets (build+run):"
	@echo "    test_nested_ssm test_nested_ssm_backward test_poincare_gqa"
	@echo "    test_lfm test_megakernel test_multiteach test_kvfs"
	@echo "    test_kv_styx test_kivi_roundtrip test_kv_bw test_decode_path"
	@echo "    test_chunked_ssm test_polarquant test_ternary test_compress"
	@echo "    test_compress2 test_eagle test_pga_backward test_mobius_linear"
	@echo "    test_hyperbolic_output_proj test_rsgd test_backward test_tst"
	@echo "    test_moe_hyperbolic test_mobius_linear test_256k test_kv_cache"
	@echo "    test_model test_model_adapter test_cpu_timing"
	@echo ""
	@echo "  Individual tool targets:"
	@echo "    gen_text wubu_cli api_server infer_moe infer_moe_lazy infer_unified"
	@echo "    infer_poincare infer_vision_gpu load_model"
	@echo "    train_integrated test_chunked_ssm test_btl3_lora"
	@echo ""
	@echo "  Agent workflow:"
	@echo "    make help          - this help"
	@echo "    make test_all      - build + run all tests (gate before committing)"
	@echo "    make test_<name>   - build + run one test"
	@echo "    make <tool>        - build one tool binary"
	@echo "    make compile_commands.json - generate clangd compile_commands.json"
	@echo "    make symbols.json  - generate docs/symbols.json symbol index"
	@echo "    make ninja         - generate build.ninja from compile_commands.json"
	@echo "    make CCACHE=0      - build without ccache"
	@echo "    make clean && make all - full rebuild from scratch"
	@echo ""
	@echo "  Architecture docs:"
	@echo "    docs/TOPOLOGY.md  - master map of both repos"
	@echo "    docs/adr/         - architecture decision records"
	@echo "    research/INDEX.md - gap ledger (open vs wired)"
	@echo "    AGENTS.md         - this repo's agent onboarding"

api_server: tools/api_server.c
	$(CC) -O2 -g -Wall -DWUBU_TOOL_VERSION=\"$(WUBU_VERSION)\" -I include -o $@ $< -lssl -lcrypto -lm

# Object files
CORE_OBJ = src/wubu_model.o src/wubu_model_format.o src/wubu_model_format_gguf.o src/wubu_model_format_st.o src/wubu_model_format_onnx.o src/wubu_dims.o src/wubu_dims_gpu.o src/wubu_ops.o src/wubu_plugin.o src/wubu_ssm.o src/wubu_ssm_workspace.o src/wubu_ssm_chunked.o src/wubu_mobius.o src/wubu_nested_ssm.o src/wubu_nested_ssm_backward.o src/wubu_moe.o src/wubu_moe_backward.o src/wubu_moe_hyperbolic.o src/wubu_poincare_ssm_backward.o src/wubu_poincare_gqa.o src/wubu_poincare_gqa_backward.o src/wubu_mobius_linear.o src/wubu_hyperbolic_output_proj.o src/wubu_vision.o src/gguf_reader.o src/qlearner.o src/rsgd.o src/wubu_tst.o src/dequant_iq2_xxs.o src/quantized_matmul.o src/quantized_dot_generic.o src/wubu_weight.o src/wubu_graph.o src/safetensors_reader.o src/wubu_repetition.o src/wubu_lora.o src/wubu_model_adapter.o src/wubu_safetensors_shard.o src/wubu_ssd_moe.o src/wubu_gemm.o src/wubu_kvcache_quant.o src/wubu_ssm_scan.o src/wubu_roofline.o src/wubu_kv_select.o src/wubu_kv_runtime.o src/wubu_gemv_tune.o src/wubu_affinity.o src/wubu_rotate.o src/wubu_flashdecode.o src/wubu_kvvq.o src/wubu_spec_decode.o src/wubu_generate.o src/wubu_ternary.o src/wubu_smoothquant.o src/wubu_arena.o src/wubu_mem_budget.o src/wubu_prefix_cache.o src/wubu_paged_kv.o src/wubu_q4k_m.o src/wubu_delta_net.o src/wubu_scheduler.o src/wubu_ngram.o src/wubu_self_cascade.o src/wubu_spec_cascade.o src/wubu_spawn.o src/wubu_kv_styx.o src/wubu_kv_tier.o src/wubu_kvfs.o src/wubu_kv_compress.o src/wubu_kv_embedding.o src/wubu_tokenizer_hf.o src/wubu_gguf_tokenizer.o src/wubu_attn_gate.o src/wubu_layer_skip.o src/wubu_kv_adaptive.o src/wubu_awq.o src/wubu_gptq.o src/wubu_soa.o src/wubu_flash_prefill.o src/wubu_kv_cacheline.o src/wubu_rope_prefetch.o src/wubu_numerical_audit.o src/wubu_mla.o src/wubu_expert_choice.o src/wubu_chunked_prefill.o src/wubu_smt_check.o src/wubu_lmcache.o src/wubu_kernel.o src/wubu_kernel_backends.o src/wubu_fast_attn.o src/wubu_4kv.o src/wubu_polarquant.o src/wubu_eagle.o src/wubu_kv_evict.o src/wubu_thread_spec.o src/wubu_early_exit.o src/wubu_hwcaps.o src/wubu_tandem.o src/wubu_rambus.o src/wubu_gamebud.o src/wubu_fp8.o src/wubu_ecs.o src/wubu_nvfp4.o src/wubu_hadamard.o src/wubu_expert_allreduce.o src/wubu_equiv_check.o src/wubu_integrate.o src/wubu_capzero.o src/wubu_latency.o src/wubu_ctxvm.o src/wubu_safekern.o src/wubu_loopguard.o src/wubu_planediv.o src/wubu_coord.o src/wubu_metagame.o src/wubu_credit.o src/wubu_metagame2.o src/wubu_resource.o src/wubu_worldmodel.o src/wubu_agentauth.o src/wubu_vecsearch.o src/wubu_causal.o src/wubu_symbolic.o src/wubu_dgm.o src/wubu_tooluse.o src/wubu_synth.o src/wubu_evolve.o src/wubu_codeexec.o src/wubu_sandbox_safekern.o src/wubu_codesynth.o src/wubu_verify.o src/wubu_experibuf.o src/wubu_ewc.o src/wubu_taskbd.o src/wubu_distill.o src/wubu_imgenc.o src/wubu_audio.o src/wubu_mm_align.o src/wubu_mm_adapter.o src/wubu_mm_kv.o src/wubu_bft.o src/wubu_threshsig.o src/wubu_agentid.o src/wubu_semcons.o src/wubu_fraud.o src/wubu_symreg.o src/wubu_sindy.o src/wubu_cegis.o src/wubu_prover.o src/wubu_invariant.o src/wubu_gp.o src/wubu_acq.o src/wubu_bo.o src/wubu_uq.o src/wubu_active.o src/wubu_bandit.o src/wubu_reinforce.o src/wubu_policy.o src/wubu_actor_critic.o src/wubu_ppo.o src/wubu_dqn.o src/wubu_value.o src/wubu_specdec.o src/wubu_pagedkv.o src/wubu_moeroute.o src/wubu_contbatch.o src/wubu_medusa.o src/wubu_quantkv.o src/wubu_hashrouter.o src/wubu_dsa.o src/wubu_tensor_store.o src/wubu_mhc_mh.o src/wubu_mxfp4.o src/wubu_linear_attn.o src/wubu_enc_h3.o src/wubu_dsv4.o src/wubu_lfm.o src/wubu_megakernel.o src/wubu_multiteach.o src/wubu_dequant_fp4.o src/wubu_dequant_nf4.o src/wubu_backend.o src/wubu_gguf_names.o src/wubu1_block_bridge.o src/wubu_hive.o src/wubu_ecosystem.o src/wubu_gravity.o src/wubu_orbits.o src/wubu_router.o src/wubu_router_physics.o src/wubu_scale.o
MODEL_OBJ = $(CORE_OBJ)
CUDA_OBJ = src/cuda_kernels.o src/gpu_output_proj.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o src/wubu_kernel_cuda.o
GPU_OBJ = src/wubu_model_gpu.o src/wubu_gpu_weight_cache.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/gpu_moe_kernel.o src/gpu_ssm_recurrence.o src/wubu_kv_runtime.o src/wubu_gemv_tune.o src/wubu_affinity.o src/wubu_rotate.o src/wubu_flashdecode.o src/wubu_kvvq.o src/wubu_spec_decode.o src/wubu_generate.o src/wubu_ternary.o src/wubu_smoothquant.o src/wubu_arena.o src/wubu_backend_cuda.o
RSGD_OBJ = src/rsgd.o

# New Colonel-model support (safetensors + LoRA + repetition + adapter)
NEW_OBJ = src/safetensors_reader.o src/wubu_repetition.o src/wubu_lora.o \
          src/wubu_model_adapter.o src/wubu_model_safetensors_bridge.o \
          src/wubu_safetensors_shard.o src/wubu_ssd_moe.o

# WuBuOS EDR layer (linked directly into the agent gauntlet; no daemon).
# Path to the WuBuOS checkout; override on the command line if checked out elsewhere.
WUBUOS ?= /home/wubu/wubuos
EDR_INC = -I$(WUBUOS)/src/runtime -I$(WUBUOS)/src/runtime/edr
EDR_SRC  = $(WUBUOS)/src/runtime/wubu_edr.c \
           $(WUBUOS)/src/runtime/edr/edr_core.c \
           $(WUBUOS)/src/runtime/edr/edr_proc_pin.c \
           $(WUBUOS)/src/runtime/edr/edr_fanotify.c \
           $(WUBUOS)/src/runtime/edr/edr_poller.c

src/wubu_ssd_moe.o: src/wubu_ssd_moe.c include/wubu_ssd_moe.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/safetensors_reader.o: src/safetensors_reader.c include/safetensors_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_safetensors_shard.o: src/wubu_safetensors_shard.c include/wubu_safetensors_shard.h include/safetensors_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_dims.o: src/wubu_dims.c include/wubu_dims.h
	$(CC) $(CFLAGS) -c $< -o $@

src/wubu_dims_gpu.o: src/wubu_dims_gpu.cu include/wubu_dims.h
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

src/wubu_kernel_cuda.o: src/wubu_kernel_cuda.cu include/wubu_kernel.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_backend_cuda.o: src/wubu_backend_cuda.c include/wubu_backend.h include/wubu_model_gpu.h include/wubu_kvfs.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_kernel_backends.o: src/wubu_kernel_backends.c include/wubu_kernel.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_model_safetensors_bridge.o: src/wubu_model_safetensors_bridge.c include/wubu_model_safetensors_bridge.h include/wubu_model.h include/wubu_lora.h include/wubu_model_adapter.h include/wubu_affinity.h include/wubu_rotate.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_repetition.o: src/wubu_repetition.c include/wubu_repetition.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_lora.o: src/wubu_lora.c include/wubu_lora.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_model_adapter.o: src/wubu_model_adapter.c include/wubu_model_adapter.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_safetensors_model.o: src/wubu_safetensors_model.c include/wubu_safetensors_model.h include/safetensors_reader.h include/wubu_model_adapter.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/qlearner.o: src/qlearner.c include/qlearner.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_ssm.o: src/wubu_ssm.c include/wubu_ssm.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_ssm_chunked.o: src/wubu_ssm_chunked.c include/wubu_ssm.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_nested_ssm.o: src/wubu_nested_ssm.c include/wubu_nested_ssm.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_nested_ssm_backward.o: src/wubu_nested_ssm_backward.c include/wubu_nested_ssm.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_mobius.o: src/wubu_mobius.c include/wubu_mobius.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_moe.o: src/wubu_moe.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_moe_hyperbolic.o: src/wubu_moe_hyperbolic.c include/wubu_moe_hyperbolic.h include/wubu_moe.h include/wubu_mobius.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_moe_backward.o: src/wubu_moe_backward.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_poincare_ssm_backward.o: src/wubu_poincare_ssm_backward.c include/wubu_ssm.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_poincare_gqa.o: src/wubu_poincare_gqa.c include/wubu_poincare_gqa.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_poincare_gqa_backward.o: src/wubu_poincare_gqa_backward.c include/wubu_poincare_gqa.h include/wubu_ssm.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_prefix_cache.o: src/wubu_prefix_cache.c include/wubu_prefix_cache.h include/wubu_paged_kv.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_mem_budget.o: src/wubu_mem_budget.c include/wubu_mem_budget.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_stream_kv.o: src/wubu_stream_kv.c include/wubu_stream_kv.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_capacity_wall.o: src/wubu_capacity_wall.c include/wubu_capacity_wall.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_hugepage.o: src/wubu_hugepage.c include/wubu_hugepage.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_kv_budget.o: src/wubu_kv_budget.c include/wubu_kv_budget.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_wm_kv.o: src/wubu_wm_kv.c include/wubu_wm_kv.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_spec_tuner.o: src/wubu_spec_tuner.c include/wubu_spec_tuner.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_quant_selector.o: src/wubu_quant_selector.c include/wubu_quant_selector.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_kv_compress.o: src/wubu_kv_compress.c include/wubu_kv_compress.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_lruk.o: src/wubu_lruk.c include/wubu_lruk.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_sparse_attn.o: src/wubu_sparse_attn.c include/wubu_sparse_attn.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_attn_tune.o: src/wubu_attn_tune.c include/wubu_attn_tune.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_kv_shield.o: src/wubu_kv_shield.c include/wubu_kv_shield.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_ctx_manage.o: src/wubu_ctx_manage.c include/wubu_ctx_manage.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_lookahead.o: src/wubu_lookahead.c include/wubu_lookahead.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_sys_tune.o: src/wubu_sys_tune.c include/wubu_sys_tune.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_lm_infinite.o: src/wubu_lm_infinite.c include/wubu_lm_infinite.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_spec_variants.o: src/wubu_spec_variants.c include/wubu_spec_variants.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_more_spec.o: src/wubu_more_spec.c include/wubu_more_spec.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_misc_gaps.o: src/wubu_misc_gaps.c include/wubu_misc_gaps.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_bf16_gemv.o: src/wubu_bf16_gemv.c include/wubu_bf16_gemv.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_attn_kernels.o: src/wubu_attn_kernels.c include/wubu_attn_kernels.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_db_cross.o: src/wubu_db_cross.c include/wubu_db_cross.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_kv2026.o: src/wubu_kv2026.c include/wubu_kv2026.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_kv2026b.o: src/wubu_kv2026b.c include/wubu_kv2026b.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_ttc.o: src/wubu_ttc.c include/wubu_ttc.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_kv2026c.o: src/wubu_kv2026c.c include/wubu_kv2026c.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_sys2026.o: src/wubu_sys2026.c include/wubu_sys2026.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_linear_attn.o: src/wubu_linear_attn.c include/wubu_linear_attn.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_ternary.o: src/wubu_ternary.c include/wubu_ternary.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_agentic_kv.o: src/wubu_agentic_kv.c include/wubu_agentic_kv.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_dn2.o: src/wubu_dn2.c include/wubu_dn2.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_parallel_spec.o: src/wubu_parallel_spec.c include/wubu_parallel_spec.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_moe_rag.o: src/wubu_moe_rag.c include/wubu_moe_rag.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_eval_qat.o: src/wubu_eval_qat.c include/wubu_eval_qat.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_pd_serve.o: src/wubu_pd_serve.c include/wubu_pd_serve.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_integrate.o: src/wubu_integrate.c include/wubu_integrate.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_bonzi2.o: src/wubu_bonzi2.c include/wubu_bonzi2.h
src/wubu_bridge2.o: src/wubu_bridge2.c include/wubu_bridge2.h
src/wubu_bonzi.o: src/wubu_bonzi.c include/wubu_bonzi.h
src/wubu_metagame2.o: src/wubu_metagame2.c include/wubu_metagame2.h
src/wubu_metacog.o: src/wubu_metacog.c include/wubu_metacog.h
src/wubu_bridge.o: src/wubu_bridge.c include/wubu_bridge.h
src/wubu_hopfield2.o: src/wubu_hopfield2.c include/wubu_hopfield2.h
src/wubu_fuzz2.o: src/wubu_fuzz2.c include/wubu_fuzz2.h
src/wubu_vision.o: src/wubu_vision.c include/wubu_vision.h
src/wubu_moondream.o: src/wubu_moondream.c include/wubu_moondream.h
src/wubu_hybrid.o: src/wubu_hybrid.c include/wubu_hybrid.h
src/wubu_compress2.o: src/wubu_compress2.c include/wubu_compress2.h
src/wubu_compress.o: src/wubu_compress.c include/wubu_compress.h
src/wubu_fuzz.o: src/wubu_fuzz.c include/wubu_fuzz.h
src/wubu_neurom.o: src/wubu_neurom.c include/wubu_neurom.h
src/wubu_rsi.o: src/wubu_rsi.c include/wubu_rsi.h
src/wubu_linattn2.o: src/wubu_linattn2.c include/wubu_linattn2.h
src/wubu_linattn.o: src/wubu_linattn.c include/wubu_linattn.h
src/wubu_token2.o: src/wubu_token2.c include/wubu_token2.h
src/wubu_token.o: src/wubu_token.c include/wubu_token.h
src/wubu_pim2.o: src/wubu_pim2.c include/wubu_pim2.h
src/wubu_pim.o: src/wubu_pim.c include/wubu_pim.h
src/wubu_serve2.o: src/wubu_serve2.c include/wubu_serve2.h
src/wubu_serve.o: src/wubu_serve.c include/wubu_serve.h
src/wubu_pref2.o: src/wubu_pref2.c include/wubu_pref2.h
src/wubu_pref.o: src/wubu_pref.c include/wubu_pref.h
src/wubu_hopfield3.o: src/wubu_hopfield3.c include/wubu_hopfield3.h
src/wubu_hopfield4.o: src/wubu_hopfield4.c include/wubu_hopfield4.h include/wubu_hopfield3.h
src/wubu_evict2026c.o: src/wubu_evict2026c.c include/wubu_evict2026c.h
src/wubu_evict2026b.o: src/wubu_evict2026b.c include/wubu_evict2026b.h
src/wubu_evict2026.o: src/wubu_evict2026.c include/wubu_evict2026.h
src/wubu_freeenergy.o: src/wubu_freeenergy.c include/wubu_freeenergy.h
src/wubu_align.o: src/wubu_align.c include/wubu_align.h
src/wubu_hopfield.o: src/wubu_hopfield.c include/wubu_hopfield.h
src/wubu_si.o: src/wubu_si.c include/wubu_si.h
src/wubu_der.o: src/wubu_der.c include/wubu_der.h
src/wubu_reverify.o: src/wubu_reverify.c include/wubu_reverify.h
src/wubu_energy.o: src/wubu_energy.c include/wubu_energy.h
src/wubu_agentic_os.o: src/wubu_agentic_os.c include/wubu_agentic_os.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_agentic_mem.o: src/wubu_agentic_mem.c include/wubu_agentic_mem.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_capzero.o: src/wubu_capzero.c include/wubu_capzero.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_latency.o: src/wubu_latency.c include/wubu_latency.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_ctxvm.o: src/wubu_ctxvm.c include/wubu_ctxvm.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_safekern.o: src/wubu_safekern.c include/wubu_safekern.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_loopguard.o: src/wubu_loopguard.c include/wubu_loopguard.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_planediv.o: src/wubu_planediv.c include/wubu_planediv.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_coord.o: src/wubu_coord.c include/wubu_coord.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_metagame.o: src/wubu_metagame.c include/wubu_metagame.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_credit.o: src/wubu_credit.c include/wubu_credit.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_metagame2.o: src/wubu_metagame2.c include/wubu_metagame2.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_resource.o: src/wubu_resource.c include/wubu_resource.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_worldmodel.o: src/wubu_worldmodel.c include/wubu_worldmodel.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_agentauth.o: src/wubu_agentauth.c include/wubu_agentauth.h
	$(CC) $(CFLAGS) -I include -c -o $@ $<

src/wubu_paged_kv.o: src/wubu_paged_kv.c include/wubu_paged_kv.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_kv_tier.o: src/wubu_kv_tier.c include/wubu_kv_tier.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_q4k_m.o: src/wubu_q4k_m.c include/wubu_q4k_m.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_delta_net.o: src/wubu_delta_net.c include/wubu_delta_net.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_ngram.o: src/wubu_ngram.c include/wubu_ngram.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_spec_cascade.o: src/wubu_spec_cascade.c include/wubu_spec_cascade.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_scheduler.o: src/wubu_scheduler.c include/wubu_scheduler.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_mobius_linear.o: src/wubu_mobius_linear.c include/wubu_mobius_linear.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_mobius_gyrate.o: src/wubu_mobius_gyrate.c include/wubu_mobius.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_moe_hyperbolic_backward.o: src/wubu_moe_hyperbolic_backward.c include/wubu_moe_hyperbolic.h include/wubu_mobius.h include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_hyperbolic_output_proj.o: src/wubu_hyperbolic_output_proj.c include/wubu_hyperbolic_output_proj.h include/wubu_mobius_linear.h include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_vision.o: src/wubu_vision.c include/wubu_vision.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/gguf_reader.o: src/gguf_reader.c include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_generate.o: src/wubu_generate.c include/wubu_generate.h include/wubu_model.h include/wubu_spec_decode.h
src/wubu_kvvq.o: src/wubu_kvvq.c include/wubu_kvvq.h
src/wubu_arena.o: src/wubu_arena.c include/wubu_arena.h
src/wubu_smoothquant.o: src/wubu_smoothquant.c include/wubu_smoothquant.h
src/wubu_ternary.o: src/wubu_ternary.c include/wubu_ternary.h
src/wubu_spec_decode.o: src/wubu_spec_decode.c include/wubu_spec_decode.h
src/wubu_flashdecode.o: src/wubu_flashdecode.c include/wubu_flashdecode.h
src/wubu_rotate.o: src/wubu_rotate.c include/wubu_rotate.h
src/wubu_model.o: src/wubu_model.c include/wubu_model.h include/wubu_ssm.h include/wubu_moe.h include/gguf_reader.h include/wubu_kv_select.h include/wubu_kv_runtime.h include/wubu_affinity.h include/wubu_rotate.h include/wubu_gguf_names.h
src/wubu_gemm.o: src/wubu_gemm.c include/wubu_gemm.h
src/wubu_gemv_tune.o: src/wubu_gemv_tune.c include/wubu_gemv_tune.h include/wubu_roofline.h
src/quantized_matmul.o: src/quantized_matmul.c include/wubu_gemm.h include/wubu_gemv_tune.h include/gguf_reader.h include/wubu_ssm.h include/wubu_safetensors_shard.h
src/wubu_kv_runtime.o: src/wubu_kv_runtime.c include/wubu_kv_runtime.h include/wubu_kv_select.h include/wubu_roofline.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/cuda_kernels.o: src/cuda_kernels.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/gpu_output_proj.o: src/gpu_output_proj.cu include/gpu_output_proj.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_model_gpu.o: src/wubu_model_gpu.cu include/wubu_model.h include/cuda_kernels.h include/bench.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_gpu_weight_cache.o: src/wubu_gpu_weight_cache.cu include/wubu_kernel.h include/gguf_reader.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/gpu_quant_matmul.o: src/gpu_quant_matmul.cu include/gpu_quant_matmul.h include/gguf_reader.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/gpu_quant_matmul_row_major.o: src/gpu_quant_matmul_row_major.cu include/gpu_quant_matmul.h include/gguf_reader.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/gpu_moe_kernel.o: src/gpu_moe_kernel.cu include/gpu_moe_kernel.h include/gguf_reader.h src/iq2xxs_grid_data.inc src/iq3xxs_grid.inc
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/gpu_ssm_recurrence.o: src/gpu_ssm_recurrence.cu include/gpu_ssm_recurrence.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/flash_attn_q4_0_opt.o: src/flash_attn_q4_0_opt.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/flash_attn_q4_0_prefill_opt.o: src/flash_attn_q4_0_prefill_opt.cu include/cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/gpu_gemma4.o: src/gpu_gemma4.cu include/gpu_gemma4.h include/wubu_gemma4.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/gpu_gemma4_forward.o: src/gpu_gemma4_forward.cu include/gpu_gemma4.h include/wubu_gemma4.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/rsgd.o: src/rsgd.c include/rsgd.h include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/wubu_tst.o: src/wubu_tst.c include/wubu_tst.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/bench.o: src/bench.c include/bench.h include/cuda_kernels.h include/wubu_ssm.h
	$(CC) $(CFLAGS) $(CUDA_INC) -c -o $@ $<

src/dequant_iq2_xxs.o: src/dequant_iq2_xxs.c include/gguf_reader.h
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

# Test binaries

test_chunked_ssm: tools/test_chunked_ssm.c src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o src/wubu_model.o,$(CORE_OBJ) src/wubu_dense_ffn.o)
	$(CC) $(CFLAGS) -o $@ tools/test_chunked_ssm.c src/wubu_moe_cpu.o $(filter-out src/wubu_moe.o src/wubu_model.o,$(CORE_OBJ)) $(LDFLAGS)

test_decode_path: tools/test_decode_path.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_nested_ssm: tools/test_nested_ssm.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
test_nested_ssm_backward: tools/test_nested_ssm_backward.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


test_poincare_gqa: tools/test_poincare_gqa.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_poincare_kv_cache: tools/test_poincare_kv_cache.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_pga_backward: tools/test_pga_backward.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_mobius_linear: tools/test_mobius_linear.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_hyperbolic_output_proj: tools/test_hyperbolic_output_proj.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gpu_layers: tools/test_gpu_layers.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ) src/bench.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_gyrate: tools/test_gyrate.c src/wubu_mobius.o src/wubu_mobius_gyrate.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_poincare_router_backward: tools/test_poincare_router_backward.c $(CORE_OBJ) src/wubu_dense_ffn.o src/wubu_moe_hyperbolic_backward.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_nested_moe_router_backward: tools/test_nested_moe_router_backward.c $(CORE_OBJ) src/wubu_dense_ffn.o src/wubu_moe_hyperbolic_backward.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ---- New Colonel-model unit tests ----
gen_fixture_safetensors: tools/gen_fixture_safetensors.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_kv_styx: tools/test_kv_styx.c src/wubu_kv_styx.o src/wubu_kvfs.o src/wubu_kv_compress.o src/wubu_spawn.o
	$(CC) $(CFLAGS) -o $@ tools/test_kv_styx.c src/wubu_kv_styx.o src/wubu_kvfs.o src/wubu_kv_compress.o src/wubu_spawn.o $(LDFLAGS)
test_kv_styx_g2: tools/test_kv_styx_g2.c src/wubu_kv_styx.o src/wubu_kvfs.o src/wubu_kv_compress.o src/wubu_spawn.o
	$(CC) $(CFLAGS) -o $@ tools/test_kv_styx_g2.c src/wubu_kv_styx.o src/wubu_kvfs.o src/wubu_kv_compress.o src/wubu_spawn.o $(LDFLAGS)
	./test_kv_styx_g2
test_kvfs: tools/test_kvfs.c src/wubu_kvfs.o
	$(CC) $(CFLAGS) -o $@ tools/test_kvfs.c src/wubu_kvfs.o $(LDFLAGS)
	./test_kvfs

src/wubu_kv_embedding.o: src/wubu_kv_embedding.c include/wubu_kv_embedding.h include/wubu_kvfs.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_kv_embedding.c

src/wubu_coherence_reward.o: src/wubu_coherence_reward.c include/wubu_coherence_reward.h include/wubu_kv_embedding.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_coherence_reward.c

src/wubu_fs_dataset.o: src/wubu_fs_dataset.c include/wubu_fs_dataset.h include/wubu_kv_embedding.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_fs_dataset.c

test_kv_embedding: tools/test_kv_embedding.c src/wubu_kv_embedding.o src/wubu_kvfs.o src/wubu_coherence_reward.o src/wubu_fs_dataset.o src/wubu_grow_kv.o src/wubu_tokenizer_hf.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_embedding.c src/wubu_kv_embedding.o src/wubu_kvfs.o src/wubu_coherence_reward.o src/wubu_fs_dataset.o src/wubu_grow_kv.o src/wubu_tokenizer_hf.o $(LDFLAGS)
	./test_kv_embedding

# G3 (single encoder): the embedding bridge encodes with the SAME BPE as
# the trainer/pipe — one encoder, end to end. 16 assertions.
test_kv_g3: tools/test_kv_g3.c src/wubu_kv_embedding.o src/wubu_kvfs.o src/wubu_tokenizer_hf.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_g3.c src/wubu_kv_embedding.o src/wubu_kvfs.o src/wubu_tokenizer_hf.o $(LDFLAGS)
	./test_kv_g3

# G5 (Styx 9P export): the mind materializes /n/kv/ as a real directory
# tree the WuBuOS Styx host serves — the body can ls the mind. 22 asserts.
test_kv_styx_g5: tools/test_kv_styx_g5.c src/wubu_kv_styx.o src/wubu_kvfs.o src/wubu_kv_compress.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_styx_g5.c src/wubu_kv_styx.o src/wubu_kvfs.o src/wubu_kv_compress.o $(LDFLAGS)
	./test_kv_styx_g5

# P3: KV immune audit — the immune system watches every seam (load/save/
# forward/mutate). NaN/Inf via IEEE bit checks (survive -ffast-math).
test_kv_audit: tools/test_kv_audit.c src/wubu_kv_audit.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_audit.c src/wubu_kv_audit.o $(LDFLAGS)
	./test_kv_audit

# P3: hyperbolic KV addressing — Poincaré-ball coordinates per region;
# retrieval/routing by geodesic distance (boundary amplification).
test_poincare_kv: tools/test_poincare_kv.c src/wubu_poincare_kv.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_poincare_kv.c src/wubu_poincare_kv.o src/wubu_mobius.o $(LDFLAGS)
	./test_poincare_kv

# P-adapt GRPO (2505.07527): Kalman-adaptive reward baseline — damps
# noisy coherence rewards, blends with classic group normalization.
test_grpo_kalman: tools/test_grpo_kalman.c src/wubu_grpo_kalman.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_grpo_kalman.c src/wubu_grpo_kalman.o $(LDFLAGS)
	./test_grpo_kalman

# LRU-K (DB buffer pools 2512.22995): sequential-flooding guard on KV
# eviction — first touch half-warms, repeat access fully warms.
test_kv_evict_lruk: tools/test_kv_evict_lruk.c src/wubu_kv_evict.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_evict_lruk.c src/wubu_kv_evict.o $(LDFLAGS)
	./test_kv_evict_lruk

src/wubu_grow_kv.o: src/wubu_grow_kv.c include/wubu_grow_kv.h include/wubu_kv_embedding.h include/wubu_kvfs.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_grow_kv.c

src/wubu_kv_hierarchy.o: src/wubu_kv_hierarchy.c include/wubu_kv_hierarchy.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_kv_hierarchy.c

src/wubu_kv_semantic_router.o: src/wubu_kv_semantic_router.c include/wubu_kv_semantic_router.h include/wubu_kv_hierarchy.h include/wubu_mobius.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_kv_semantic_router.c

src/wubu_kv_shrink.o: src/wubu_kv_shrink.c include/wubu_kv_shrink.h include/wubu_kvfs.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_kv_shrink.c

src/wubu_kv_coherence_diag.o: src/wubu_kv_coherence_diag.c include/wubu_kv_coherence_diag.h include/wubu_kv_embedding.h include/wubu_coherence_reward.h include/wubu_grow_kv.h include/wubu_kv_shrink.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_kv_coherence_diag.c

src/wubu_kv_shell.o: src/wubu_kv_shell.c include/wubu_kv_shell.h include/wubu_kv_embedding.h include/wubu_fs_dataset.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_kv_shell.c

src/wubu_kv_tiering.o: src/wubu_kv_tiering.c include/wubu_kv_tiering.h include/wubu_kv_embedding.h include/gguf_reader.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_kv_tiering.c

test_kv_tiering: tools/test_kv_tiering.c src/wubu_kv_tiering.o src/wubu_kv_embedding.o src/wubu_kvfs.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_tiering.c src/wubu_kv_tiering.o src/wubu_kv_embedding.o src/wubu_tokenizer_hf.o src/wubu_kvfs.o $(LDFLAGS)
	./test_kv_tiering

test_scalable_model: tools/test_scalable_model.c src/wubu_model_scalable.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_scalable_model.c src/wubu_model_scalable.o $(LDFLAGS)
	./test_scalable_model

src/wubu_density_planner.o: src/wubu_density_planner.c include/wubu_density_planner.h include/wubu_kv_embedding.h include/wubu_coherence_reward.h include/wubu_model_scalable.h include/wubu_kv_tiering.h include/wubu_kv_shrink.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_density_planner.c

test_density_planner: tools/test_density_planner.c src/wubu_density_planner.o src/wubu_kv_embedding.o src/wubu_kvfs.o src/wubu_model_scalable.o src/wubu_kv_shrink.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_density_planner.c src/wubu_density_planner.o src/wubu_kv_embedding.o src/wubu_tokenizer_hf.o src/wubu_kvfs.o src/wubu_model_scalable.o src/wubu_kv_shrink.o $(LDFLAGS)
	./test_density_planner

src/wubu_adapter_registry.o: src/wubu_adapter_registry.c include/wubu_adapter.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_adapter_registry.c

src/wubu_adapter_compat.o: src/wubu_adapter_compat.c include/wubu_adapter_compat.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_adapter_compat.c

test_adapter: tools/test_adapter.c src/wubu_adapter_registry.o src/wubu_adapter_compat.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_adapter.c src/wubu_adapter_registry.o src/wubu_adapter_compat.o $(LDFLAGS)
	./test_adapter

test_kv_shell: tools/test_kv_shell.c src/wubu_kv_shell.o src/wubu_kv_embedding.o src/wubu_kvfs.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_shell.c src/wubu_kv_shell.o src/wubu_kv_embedding.o src/wubu_tokenizer_hf.o src/wubu_kvfs.o $(LDFLAGS)
	./test_kv_shell

test_kv_diag: tools/test_kv_diag.c src/wubu_kv_coherence_diag.o src/wubu_kv_embedding.o src/wubu_coherence_reward.o src/wubu_grow_kv.o src/wubu_kv_shrink.o src/wubu_kvfs.o src/wubu_kv_hierarchy.o src/wubu_mobius.o src/wubu_kv_semantic_router.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_diag.c src/wubu_kv_coherence_diag.o src/wubu_kv_embedding.o src/wubu_tokenizer_hf.o src/wubu_coherence_reward.o src/wubu_grow_kv.o src/wubu_kv_shrink.o src/wubu_kvfs.o src/wubu_kv_hierarchy.o src/wubu_mobius.o src/wubu_kv_semantic_router.o $(LDFLAGS)
	./test_kv_diag

test_kv_hierarchy: tools/test_kv_hierarchy.c src/wubu_kv_hierarchy.o src/wubu_mobius.o src/wubu_kv_semantic_router.o src/wubu_kv_shrink.o src/wubu_kvfs.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_kv_hierarchy.c src/wubu_kv_hierarchy.o src/wubu_mobius.o src/wubu_kv_semantic_router.o src/wubu_kv_shrink.o src/wubu_kvfs.o $(LDFLAGS)
	./test_kv_hierarchy

test_backend_dispatch: tools/test_backend_dispatch.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./test_backend_dispatch

test_enc_h3: tools/test_enc_h3.c src/wubu_enc_h3.o src/wubu_rotate.o src/wubu_fp8.o src/wubu_nvfp4.o
	$(CC) $(CFLAGS) -o $@ tools/test_enc_h3.c src/wubu_enc_h3.o src/wubu_rotate.o src/wubu_fp8.o src/wubu_nvfp4.o $(LDFLAGS)
	./test_enc_h3

# SD1.5 txt2img (SD-port agent, d8668bb) — quantized-only inference.
# Wired into the shared build: both agents, one trunk.
txt2img_sd: tools/txt2img_sd.c src/wubu_sd_clip.o src/wubu_sd_unet.o src/wubu_sd_vae.o src/wubu_sd_ops.o src/gguf_reader.o
	$(CC) $(CFLAGS) -I include -o $@ tools/txt2img_sd.c src/wubu_sd_clip.o src/wubu_sd_unet.o src/wubu_sd_vae.o src/wubu_sd_ops.o src/gguf_reader.o $(LDFLAGS)

src/wubu_sd_clip.o: src/wubu_sd_clip.c include/wubu_sd_clip.h include/wubu_sd_ops.h include/gguf_reader.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_sd_clip.c

src/wubu_sd_unet.o: src/wubu_sd_unet.c include/wubu_sd_unet.h include/wubu_sd_ops.h include/gguf_reader.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_sd_unet.c

src/wubu_sd_vae.o: src/wubu_sd_vae.c include/wubu_sd_vae.h include/wubu_sd_ops.h include/gguf_reader.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_sd_vae.c

src/wubu_sd_ops.o: src/wubu_sd_ops.c include/wubu_sd_ops.h include/gguf_reader.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_sd_ops.c

# AN21/22 Phase 8 gate: THE MODEL TRAINS ON THE FILESYSTEM —
# files -> batches -> coherence reward -> amoeba grow (wubu_fs_trainer).
test_fs_trainer: tools/test_fs_trainer.c src/wubu_fs_trainer.o src/wubu_fs_dataset.o src/wubu_kv_embedding.o src/wubu_grow_kv.o src/wubu_coherence_reward.o src/wubu_kvfs.o src/wubu.o src/wubu_runtime_dims.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/wubu_train.o src/wubu_backprop.o src/wubu_tokenizer_hf.o
	$(CC) $(CFLAGS) -I include -fopenmp -o $@ $^ -lm
	./$@

src/wubu_fs_trainer.o: src/wubu_fs_trainer.c include/wubu_fs_trainer.h include/wubu_fs_dataset.h include/wubu_kv_embedding.h include/wubu_grow_kv.h include/wubu_train.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_fs_trainer.c

test_ops: tools/test_ops.c src/wubu_ops.o
	$(CC) $(CFLAGS) -I include -I tools/include -o $@ tools/test_ops.c src/wubu_ops.o -lm
	./test_ops

test_dsv4: tools/test_dsv4.c src/wubu_dsv4.o src/wubu_hashrouter.o src/wubu_mxfp4.o src/wubu_dsa.o
	$(CC) $(CFLAGS) -o $@ tools/test_dsv4.c src/wubu_dsv4.o src/wubu_hashrouter.o src/wubu_mxfp4.o src/wubu_dsa.o $(LDFLAGS) -lm
	./test_dsv4

test_lfm: tools/test_lfm.c src/wubu_lfm.o src/wubu_linear_attn.o
	$(CC) $(CFLAGS) -o $@ tools/test_lfm.c src/wubu_lfm.o src/wubu_linear_attn.o $(LDFLAGS) -lm
	./test_lfm

test_multiteach: tools/multiteach_selftest.c src/wubu_multiteach.o
	$(CC) $(CFLAGS) -o $@ tools/multiteach_selftest.c src/wubu_multiteach.o $(LDFLAGS) -lm
	./test_multiteach

test_megakernel: tools/test_megakernel.c src/wubu_megakernel.o
	$(CC) $(CFLAGS) -o $@ tools/test_megakernel.c src/wubu_megakernel.o $(LDFLAGS) -lm
	./test_megakernel

test_kivi_roundtrip: tools/test_kivi_roundtrip.c src/wubu_kvcache_quant.o
	$(CC) $(CFLAGS) -o test_kivi_roundtrip $< src/wubu_kvcache_quant.o $(LDFLAGS) -lm
	./test_kivi_roundtrip

test_kv_bw: tools/test_kv_bw.c src/wubu_kvcache_quant.o
	$(CC) $(CFLAGS) -o test_kv_bw $< src/wubu_kvcache_quant.o $(LDFLAGS) -lm
	./test_kv_bw

test_4kv: tools/test_4kv.c src/wubu_4kv.o
	$(CC) $(CFLAGS) -o $@ $< src/wubu_4kv.o $(LDFLAGS) -lm
	./$@

test_safetensors: tools/test_safetensors.c src/safetensors_reader.o src/wubu_dequant_nf4.o gen_fixture_safetensors
	$(CC) $(CFLAGS) -o $@ $< src/safetensors_reader.o $(LDFLAGS)
	./gen_fixture_safetensors
	./$@

test_repetition: tools/test_repetition.c src/wubu_repetition.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_lora: tools/test_lora.c src/wubu_lora.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_model_adapter: tools/test_model_adapter.c src/wubu_model_adapter.o $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_new_models: test_safetensors test_repetition test_lora test_model_adapter test_st_bridge
	@echo "=== new Colonel-model unit tests PASSED ==="

gen_fixture_safetensors_model: tools/gen_fixture_safetensors_model.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

gen_fixture_btl3_lora: tools/gen_fixture_btl3_lora.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_st_bridge: tools/test_st_bridge.c $(MODEL_OBJ) src/wubu_dense_ffn.o gen_fixture_safetensors_model
	$(CC) $(CFLAGS) -o $@ tools/test_st_bridge.c $(MODEL_OBJ) $(LDFLAGS)
	./gen_fixture_safetensors_model
	./$@

test_universal_weight: tools/test_universal_weight.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ tools/test_universal_weight.c $(MODEL_OBJ) $(LDFLAGS)
	@echo "== universal weight descriptor vs real model-mixed.gguf =="
	./$@ models/wubu/model-mixed.gguf

test_wubu_graph: tools/test_wubu_graph.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ tools/test_wubu_graph.c $(MODEL_OBJ) $(LDFLAGS)
	./$@

test_btl3_lora: tools/test_btl3_lora.c $(MODEL_OBJ) src/wubu_dense_ffn.o gen_fixture_safetensors_model gen_fixture_btl3_lora
	$(CC) $(CFLAGS) -o $@ tools/test_btl3_lora.c $(MODEL_OBJ) $(LDFLAGS)
	./gen_fixture_safetensors_model
	./gen_fixture_btl3_lora
	./$@

test_real_load: tools/test_real_load.c src/wubu_model_adapter.o $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_ssd_moe: tools/test_ssd_moe.c src/wubu_ssd_moe.o src/wubu_safetensors_shard.o src/safetensors_reader.o src/wubu_dequant_nf4.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ── 100-improvement modules (Areas A/B/C/D/F/H/I/J/K) ───────────────
test_spec_decode: tools/test_spec_decode.c src/wubu_spec_decode.o src/wubu_ngram.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_kvquant: tools/test_kvquant.c src/wubu_kvquant.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_paged_kv: tools/test_paged_kv.c src/wubu_paged_kv.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_moe_grouped: tools/test_moe_grouped.c src/wubu_moe_grouped.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_ssm_scan: tools/test_ssm_scan.c src/wubu_ssm_scan.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_q8: tools/test_q8.c src/wubu_q8.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_cuda_graph: tools/test_cuda_graph.c src/wubu_cuda_graph.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# ── Round-2 (cross-disciplinary) modules (Areas L/M/N/O) ───────
test_roofline: tools/test_roofline.c src/wubu_roofline.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_cache_advice: tools/test_cache_advice.c src/wubu_cache_advice.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_kereq: tools/test_kereq.c src/wubu_kereq.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_pd_split: tools/test_pd_split.c src/wubu_pd_split.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# ── Round-3 (new-model architecture: hybrid/recurrent, mHC, CLA, MEGA, YaRN) ──
test_delta_net: tools/test_delta_net.c src/wubu_delta_net.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_mhc: tools/test_mhc.c src/wubu_mhc.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_hashrouter: tools/test_hashrouter.c src/wubu_hashrouter.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_dsa: tools/test_dsa.c src/wubu_dsa.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_tensor_store: tools/test_tensor_store.c src/wubu_tensor_store.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_gguf_tq: tools/test_gguf_tq.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_gguf_load: tools/test_gguf_load.c src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# P0 gate (COHESIVE_DIRECTION.md Step 1): role-based loader resolves TRUE
# dims on our own WuBu-35M GGUF (12 layers, 448 dim, 16384 vocab — not the
# old 1-layer/2048/248320 fallback that segfaulted).
test_role_load: tools/test_role_load.c src/wubu_gguf_names.o src/gguf_reader.o src/dequant_iq2_xxs.o src/wubu_dequant_fp4.o src/wubu_dequant_nf4.o src/quantized_dot_generic.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./test_role_load models/wubu/model-mixed.gguf

# P1 gate (COHESIVE_DIRECTION.md Step 2+4): canonical wubu1_block_t +
# lossless bridge from the loaded layer struct (config-as-data header
# validated, every GQA layer converts with identity weight pointers).
test_wubu1_block: tools/test_wubu1_block.c $(CORE_OBJ) src/wubu_dense_ffn.o src/wubu1_block_bridge.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./test_wubu1_block models/wubu/model-mixed.gguf

# P2 gate (COHESIVE_DIRECTION.md Step 3): role-tagged .st v3 header —
# per-tensor role + quant + declared dims (header-as-data, not bare blob).
test_st_v3: tools/test_st_v3.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./test_st_v3 models/wubu/model-mixed.gguf

test_wubu_mhc_mh: tools/test_wubu_mhc_mh.c src/wubu_mhc_mh.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_cla: tools/test_cla.c src/wubu_cla.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_mega: tools/test_mega.c src/wubu_mega.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_yarn: tools/test_yarn.c src/wubu_yarn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# ── Round-4 (Kimi K3: KDA, AttnRes, Stable LatentMoE, MXFP4/MXFP8) ──
test_kda: tools/test_kda.c src/wubu_kda.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_attnres: tools/test_attnres.c src/wubu_attnres.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_latentmoe: tools/test_latentmoe.c src/wubu_latentmoe.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_mxfp4: tools/test_mxfp4.c src/wubu_mxfp4.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_scheduler: tools/test_scheduler.c src/wubu_scheduler.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_affinity: tools/test_affinity.c src/wubu_affinity.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# Aggregate of ALL 400 improvement unit tests (Round-1 + Round-2 + Round-3 + Round-4).
test_400: test_spec_decode test_kvquant test_paged_kv test_moe_grouped \
          test_ssm_scan test_q8 test_cuda_graph test_scheduler test_affinity \
          test_roofline test_cache_advice test_kereq test_pd_split \
          test_delta_net test_mhc test_hashrouter test_dsa test_tensor_store test_diag test_gravity test_orbits test_gguf_tq test_wubu_mhc_mh test_cla test_mega test_yarn \
          test_kda test_attnres test_latentmoe test_mxfp4 \
          test_kv_tier test_kv_tier_evict
	@echo "ALL 400-IMPROVEMENT UNIT TESTS PASSED"

# Aggregate of ALL 300 improvement unit tests (Round-1 + Round-2 + Round-3).
test_300: test_spec_decode test_kvquant test_paged_kv test_moe_grouped \
          test_ssm_scan test_q8 test_cuda_graph test_scheduler test_affinity \
          test_roofline test_cache_advice test_kereq test_pd_split \
          test_delta_net test_mhc test_hashrouter test_dsa test_tensor_store test_wubu_mhc_mh test_cla test_mega test_yarn
	@echo "ALL 300-IMPROVEMENT UNIT TESTS PASSED"

# Aggregate of ALL 200 improvement unit tests (Round-1 + Round-2).
test_200: test_spec_decode test_kvquant test_paged_kv test_moe_grouped \
          test_ssm_scan test_q8 test_cuda_graph test_scheduler test_affinity \
          test_roofline test_cache_advice test_kereq test_pd_split
	@echo "ALL 200-IMPROVEMENT UNIT TESTS PASSED"

test_model_config: tools/test_model_config.c src/wubu_model_adapter.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

test_tokenizer: tools/test_tokenizer.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_model: tools/test_model.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ── Agent tool gauntlet (4 Colonel models × 3 tools, EDR fan-out) ──────────
# Links the wubuwizard engine + WuBuOS EDR layer. The EDR sources are
# standalone (lock-free ring + worker thread); no daemon required.
gauntlet: tools/agent_gauntlet/agent_gauntlet.c tools/agent_gauntlet/gauntlet_run.c \
          tools/agent_gauntlet/agent_gauntlet.h $(MODEL_OBJ) $(EDR_SRC) \
          src/wubu_tokenizer_hf.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) $(EDR_INC) -o $@ tools/agent_gauntlet/agent_gauntlet.c \
		tools/agent_gauntlet/gauntlet_run.c $(MODEL_OBJ) $(EDR_SRC) \
		src/wubu_tokenizer_hf.o src/wubu_tokenizer.o $(LDFLAGS) -lpthread

test_gauntlet: tools/agent_gauntlet/agent_gauntlet.c tools/agent_gauntlet/test_gauntlet.c \
               tools/agent_gauntlet/agent_gauntlet.h $(MODEL_OBJ) $(EDR_SRC) \
               src/wubu_tokenizer_hf.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) $(EDR_INC) -o $@ tools/agent_gauntlet/agent_gauntlet.c \
		tools/agent_gauntlet/test_gauntlet.c $(MODEL_OBJ) $(EDR_SRC) \
		src/wubu_tokenizer_hf.o src/wubu_tokenizer.o $(LDFLAGS) -lpthread
	./$@

# Verify the principled GDN chunkwise-parallel recurrence vs the sequential
# scalar reference (must match to ~1e-2 at every chunk size C).
test_gdn_chunk: tools/agent_gauntlet/test_gdn_chunk.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

# Our-own-kernel GEMM benchmark: baseline scalar vs tiled AVX2/AVX512-FMA.
bench_gemm: tools/bench_gemm.c src/wubu_gemm.o
	$(CC) $(CFLAGS) -mavx2 -mfma -o $@ $^ -lm -fopenmp
	./$@

# WuBuOS-agnostic kernel dispatch: prove device-backend registry works.
test_kernel_dispatch: tools/test_kernel_dispatch.c src/wubu_kernel.o src/wubu_kernel_backends.o src/wubu_kernel_cuda.o
	$(CXX) $(CFLAGS) -DWUBU_ENABLE_CUDA -I include -o $@ tools/test_kernel_dispatch.c src/wubu_kernel.o src/wubu_kernel_backends.o src/wubu_kernel_cuda.o $(LDFLAGS) -L$(CUDA_LIBDIR) -lcudart -lstdc++
	./$@

bench_gemm_run: bench_gemm

# Projection accuracy: quantized_matmul GEMV vs unambiguous scalar oracle
# on REAL Qwen layer-0 gate_proj weights. Catches the transpose-layout bug.
# CPU-only link (exclude CUDA/GPU objects that need -lcuda).
CPU_OBJ = $(filter-out src/wubu_dims_gpu.o src/cuda_kernels.o src/gpu_output_proj.o src/flash_attn_q4_0_opt.o src/flash_attn_q4_0_prefill_opt.o src/wubu_model_gpu.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/gpu_moe_kernel.o src/gpu_ssm_recurrence.o,$(CORE_OBJ)) src/wubu_dims_gpu_stub.o
test_proj_accuracy: tools/test_proj_accuracy.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# KV-cache quantization integration (engine's real kv_cache_read/write_head
# under each scheme) + roofline-driven wubu_kv_select. Three builds:
#  -DWAIT no — builds with KV_CACHE_OUR_Q8, KV_CACHE_KIVI, and default.
test_kv_cache_q8: tools/test_kv_cache_integration.c $(CPU_OBJ) src/wubu_kv_select.o
	$(CC) $(CFLAGS) -DKV_CACHE_OUR_Q8 -I include -o $@ $^ -lm -fopenmp
	./$@

test_kv_cache_kivi: tools/test_kv_cache_integration.c $(CPU_OBJ) src/wubu_kv_select.o
	$(CC) $(CFLAGS) -DKV_CACHE_KIVI -I include -o $@ $^ -lm -fopenmp
	./$@

# KV-cache quant unit test (Q8_0 + KIVI K!=V axes). Round-trip + edge.
# Roofline-driven GEMV auto-tuner: tiled fp32 + int8 variants vs scalar oracle
# + tuner sanity. Real Qwen weight used for one probe (skip-safe if absent).
# B03: int4 weight GEMV vs fp32 oracle + autotune precedence
# doc 018/K01: n-gram speculative decoding generator (equivalence vs plain)
test_generate_spec: tools/test_generate_spec.c src/wubu_generate.o src/wubu_spec_decode.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# G03: n-gram drafter unit test (no model needed)
test_ngram: tools/test_ngram.c src/wubu_ngram.o
	$(CC) $(CFLAGS) -I include -o $@ $< src/wubu_ngram.o -lm
	./$@

# G04: Hive data structure test (linked fixed blocks + skipfield + freelist)
test_hive: tools/test_hive.c src/wubu_hive.o $(CPU_OBJ)
test_diag: tools/test_diag.c src/wubu_diag.o src/wubu_hive.o src/wubu.o src/wubu_train.o src/wubu_backprop.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/wubu_runtime_dims.o gpu_wubu.o
	$(CC) $(CFLAGS) -I include -fopenmp -o $@ $^ -lm $(CUDA_LIBS)
	./$@
test_gravity: tools/test_gravity.c src/wubu_gravity.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@
test_orbits: tools/test_orbits.c src/wubu_orbits.o src/wubu_hive.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@
test_amoeba: tools/test_amoeba.c src/wubu_amoeba.o src/wubu_hive.o src/wubu_moe2.o src/wubu_prover2.o src/wubu_hyper.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_diagnosis: tools/test_diagnosis.c src/wubu_diagnosis.o src/wubu_amoeba.o src/wubu_hive.o src/wubu_moe2.o src/wubu_prover2.o src/wubu_hyper.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_orch: tools/test_orch.c src/wubu_agi.o src/wubu_hive.o src/wubu_moe2.o src/wubu_prover2.o src/wubu_hyper.o src/wubu.o src/wubu_runtime_dims.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/wubu_mobius.o src/wubu_grow.o src/wubu_plateau.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_pref: tools/test_pref.c src/wubu_pref.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_colonel: tools/test_colonel.c src/wubu_colonel.o src/wubu_agentic_os.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_selfimprove: tools/test_selfimprove.c src/wubu_selfimprove.o src/wubu_rsi.o src/wubu_hive.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_lineage: tools/test_lineage.c src/wubu_lineage.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_trajcell: tools/test_trajcell.c src/wubu_trajcell.o src/wubu_hive.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_metadiag: tools/test_metadiag.c src/wubu_metadiag.o src/wubu_hive.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_capgate: tools/test_capgate.c src/wubu_capgate.o src/wubu_hive.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_contracts: tools/test_contracts.c src/wubu_contracts.o src/wubu_hive.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)











test_nest: tools/test_nest.c src/wubu_nest.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_deltanet: tools/test_deltanet.c src/wubu_deltanet.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_prover2: tools/test_prover2.c src/wubu_prover2.o src/wubu_hyper.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 006: arena allocator for per-request + KV buffers
test_arena: tools/test_arena.c src/wubu_arena.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp -lssl -lcrypto
	./$@

# doc 005: SmoothQuant activation outlier migration
test_smoothquant: tools/test_smoothquant.c src/wubu_smoothquant.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp -lssl -lcrypto
	./$@

# doc 004: BitNet 1.58 ternary {-1,0,+1} GEMV
test_ternary: tools/test_ternary.c src/wubu_ternary.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 014: sub-4-bit KV vector quantization (CommVQ/TurboQuant)
test_kvvq: tools/test_kvvq.c src/wubu_kvvq.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 002: KV multi-tier storage (hot/warm/cold)
test_kv_tier: tools/test_kv_tier.c src/wubu_kv_tier.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

test_kv_tier_evict: tools/test_kv_tier_evict.c src/wubu_kv_tier.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc 011: attention-sink-free gated attention
test_attn_gate: tools/test_attn_gate.c src/wubu_attn_gate.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 017: Mixture-of-Depths token-wise layer skip
test_layer_skip: tools/test_layer_skip.c src/wubu_layer_skip.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 012/018: self-cascade speculative decoding (n-gram drafter)
test_self_cascade: tools/test_self_cascade.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 018: CAS-Spec adaptive deferral rule (verify + cascade)
test_spec_cascade: tools/test_spec_cascade.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 012: EAGLE-style self-draft
test_eagle: tools/test_eagle.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 012: MEDUSA guess heads
test_medusa: tools/test_medusa.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 001: Ecco entropy-aware adaptive KV compression
test_kv_adaptive: tools/test_kv_adaptive.c src/wubu_kv_adaptive.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc B05: AWQ activation-aware weight quantization
test_awq: tools/test_awq.c src/wubu_awq.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc B06: GPTQ second-order weight quantization
test_gptq: tools/test_gptq.c src/wubu_gptq.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc 010: cross-request prefix KV reuse
test_prefix_reuse: tools/test_prefix_reuse.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 007: continuous (iteration-level) batching
test_continuous_batching: tools/test_continuous_batching.c src/wubu_scheduler.o src/wubu_continuous_batching.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc I02: SoA activation tensor layout
test_soa: tools/test_soa.c src/wubu_soa.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc H01: FlashAttention-style fused prefill
# Note: compiled WITHOUT -ffast-math — online softmax needs stable expf ordering
src/wubu_flash_prefill.o: src/wubu_flash_prefill.c include/wubu_flash_prefill.h
	$(CC) -O3 -march=native -funroll-loops -ftree-vectorize -Wall -Wextra -Wno-unused-parameter -I include -c -o $@ $<

test_flash_prefill: tools/test_flash_prefill.c src/wubu_flash_prefill.o
	$(CC) -O3 -march=native -funroll-loops -ftree-vectorize -Wall -Wextra -I include -o $@ $^ -lm
	./$@

# doc C03: cache-line-aligned KV page storage
test_kv_cacheline: tools/test_kv_cacheline.c src/wubu_kv_cacheline.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc A10: RoPE-aware KV prefetch
test_rope_prefetch: tools/test_rope_prefetch.c src/wubu_rope_prefetch.o src/wubu_kv_cacheline.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc F03: numerical-stability audit
# Compiled without -ffast-math: isnan/isinf must work (fast-math assumes no NaN)
src/wubu_numerical_audit.o: src/wubu_numerical_audit.c include/wubu_numerical_audit.h
	$(CC) -O3 -march=native -funroll-loops -Wall -Wextra -Wno-unused-parameter -I include -c -o $@ $<

test_numerical_audit: tools/test_numerical_audit.c src/wubu_numerical_audit.o src/wubu_kv_adaptive.o
	$(CC) -O3 -march=native -funroll-loops -Wall -Wextra -I include -o $@ tools/test_numerical_audit.c src/wubu_numerical_audit.o src/wubu_kv_adaptive.o -lm
	./$@

# doc E02: MLA (DeepSeek multi-head latent attention)
test_mla: tools/test_mla.c src/wubu_mla.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc E05: MoE expert choice routing
test_expert_choice: tools/test_expert_choice.c src/wubu_expert_choice.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc 007/D03: disaggregated prefill/decode (separate passes, shared KV)
test_disagg_prefill_decode: tools/test_disagg_prefill_decode.c src/wubu_continuous_batching.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc A07b: priority-based KV eviction (importance + LRU hybrid)
test_kv_evict: tools/test_kv_evict.c src/wubu_kv_evict.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc H02: CPU thread-specialization (prefill/decode pinned pools)
test_thread_spec: tools/test_thread_spec.c src/wubu_thread_spec.o
	$(CC) $(CFLAGS) -I include -pthread -o $@ $^ -lm
	./$@

# doc J03: early-exit + self-speculative verify
test_early_exit: tools/test_early_exit.c src/wubu_early_exit.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc "tandem"/"rambus"/"gamebud": full HW-accel stack (N64 RCP + RDRAM KV + frame budget)
test_tandem_gamebud: tools/test_tandem_gamebud.c src/wubu_hwcaps.o src/wubu_rambus.o src/wubu_tandem.o src/wubu_gamebud.o
	$(CC) $(CFLAGS) -I include -pthread -o $@ $^ -lm
	./$@

# doc "hwaccel": wubu_model_wire_hwaccel() wires the real HW stack into a model
test_model_hwaccel: tools/test_model_hwaccel.c $(CORE_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o src/wubu_tokenizer_hf.o
	$(CXX) $(CFLAGS) -DWUBU_ENABLE_CUDA -I include -pthread -o $@ $^ -lm -L$(CUDA_LIBDIR) -lcudart -lstdc++
	./$@

# doc B07: FP8 E4M3/E5M2 emulation (CPU path)
test_fp8: tools/test_fp8.c src/wubu_fp8.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc C06: ECS-style component store for engine state
test_ecs: tools/test_ecs.c src/wubu_ecs.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_512k_budget: tools/test_512k_budget.c src/wubu_mem_budget.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_stream_kv: tools/test_stream_kv.c src/wubu_stream_kv.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_kv_evict_h2o: tools/test_kv_evict_h2o.c src/wubu_kv_evict.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_capacity_wall: tools/test_capacity_wall.c src/wubu_capacity_wall.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_hugepage: tools/test_hugepage.c src/wubu_hugepage.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_kv_budget: tools/test_kv_budget.c src/wubu_kv_budget.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_wm_kv: tools/test_wm_kv.c src/wubu_wm_kv.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_spec_tuner: tools/test_spec_tuner.c src/wubu_spec_tuner.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_quant_selector: tools/test_quant_selector.c src/wubu_quant_selector.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_kv_compress: tools/test_kv_compress.c src/wubu_kv_compress.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_lruk: tools/test_lruk.c src/wubu_lruk.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_sparse_attn: tools/test_sparse_attn.c src/wubu_sparse_attn.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_attn_tune: tools/test_attn_tune.c src/wubu_attn_tune.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_kv_shield: tools/test_kv_shield.c src/wubu_kv_shield.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_ctx_manage: tools/test_ctx_manage.c src/wubu_ctx_manage.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_lookahead: tools/test_lookahead.c src/wubu_lookahead.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_sys_tune: tools/test_sys_tune.c src/wubu_sys_tune.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_lm_infinite: tools/test_lm_infinite.c src/wubu_lm_infinite.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_spec_variants: tools/test_spec_variants.c src/wubu_spec_variants.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_more_spec: tools/test_more_spec.c src/wubu_more_spec.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_misc_gaps: tools/test_misc_gaps.c src/wubu_misc_gaps.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_bf16_gemv: tools/test_bf16_gemv.c src/wubu_bf16_gemv.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_attn_kernels: tools/test_attn_kernels.c src/wubu_attn_kernels.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_db_cross: tools/test_db_cross.c src/wubu_db_cross.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_kv2026: tools/test_kv2026.c src/wubu_kv2026.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_kv2026b: tools/test_kv2026b.c src/wubu_kv2026b.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_ttc: tools/test_ttc.c src/wubu_ttc.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_kv2026c: tools/test_kv2026c.c src/wubu_kv2026c.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_sys2026: tools/test_sys2026.c src/wubu_sys2026.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_linear_attn: tools/test_linear_attn.c src/wubu_linear_attn.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_ternary: tools/test_ternary.c src/wubu_ternary.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_agentic_kv: tools/test_agentic_kv.c src/wubu_agentic_kv.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_dn2: tools/test_dn2.c src/wubu_dn2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_parallel_spec: tools/test_parallel_spec.c src/wubu_parallel_spec.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_moe_rag: tools/test_moe_rag.c src/wubu_moe_rag.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_eval_qat: tools/test_eval_qat.c src/wubu_eval_qat.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_pd_serve: tools/test_pd_serve.c src/wubu_pd_serve.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_integrate: tools/test_integrate.c src/wubu_integrate.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_agentic_os_mem: tools/test_agentic_os_mem.c src/wubu_agentic_os.o src/wubu_agentic_mem.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_capzero: tools/test_capzero.c src/wubu_capzero.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_agi_os_integration: tools/test_agi_os_integration.c src/wubu_latency.o src/wubu_ctxvm.o src/wubu_safekern.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_loopguard_planediv: tools/test_loopguard_planediv.c src/wubu_loopguard.o src/wubu_planediv.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_metagame_coord: tools/test_metagame_coord.c src/wubu_coord.o src/wubu_metagame.o src/wubu_credit.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_metagame2_resource: tools/test_metagame2_resource.c src/wubu_metagame2.o src/wubu_resource.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_worldmodel_agentauth: tools/test_worldmodel_agentauth.c src/wubu_worldmodel.o src/wubu_agentauth.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_vecsearch: tools/test_vecsearch.c src/wubu_vecsearch.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_causal_symbolic: tools/test_causal_symbolic.c src/wubu_causal.o src/wubu_symbolic.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_ax: tools/test_ax.c src/wubu_dgm.o src/wubu_tooluse.o src/wubu_synth.o src/wubu_evolve.o src/wubu_codeexec.o src/wubu_sandbox_safekern.o src/wubu_spawn.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_axi: tools/test_axi.c src/wubu_dgm.o src/wubu_tooluse.o src/wubu_synth.o src/wubu_evolve.o src/wubu_codeexec.o src/wubu_sandbox_safekern.o src/wubu_codesynth.o src/wubu_verify.o src/wubu_spawn.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_continual: tools/test_continual.c src/wubu_experibuf.o src/wubu_ewc.o src/wubu_taskbd.o src/wubu_distill.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_bi: tools/test_bi.c src/wubu_bi.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_bi.o: src/wubu_bi.c include/wubu_bi.h include/wubu.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_bi.c

test_priority: tools/test_priority.c src/wubu_priority.c
	$(CC) $(CFLAGS) -I include -o $@ tools/test_priority.c src/wubu_priority.c -lm
	./$@

test_multimodal: tools/test_multimodal.c src/wubu_imgenc.o src/wubu_audio.o src/wubu_mm_align.o src/wubu_mm_adapter.o src/wubu_mm_kv.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_multiconsensus: tools/test_multiconsensus.c src/wubu_bft.o src/wubu_threshsig.o src/wubu_agentid.o src/wubu_semcons.o src/wubu_fraud.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_ee: tools/test_ee.c src/wubu_symreg.o src/wubu_sindy.o src/wubu_cegis.o src/wubu_prover.o src/wubu_invariant.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_bonzi2: tools/test_bonzi2.c src/wubu_bonzi2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_bridge2: tools/test_bridge2.c src/wubu_bridge2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_bonzi: tools/test_bonzi.c src/wubu_bonzi.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_metagame2: tools/test_metagame2.c src/wubu_metagame2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_metacog: tools/test_metacog.c src/wubu_metacog.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_bridge: tools/test_bridge.c src/wubu_bridge.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_hopfield2: tools/test_hopfield2.c src/wubu_hopfield2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_fuzz2: tools/test_fuzz2.c src/wubu_fuzz2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_compress2: tools/test_compress2.c src/wubu_compress2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_compress: tools/test_compress.c src/wubu_compress.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# the WuBu corpus pipeline (SD card -> tokens -> trainer)
wubu_tokenc: tools/wubu_tokenc.c src/wubu_tokenizer_hf.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
wubu_pond2tok: tools/wubu_pond2tok.c src/wubu_tokenizer_hf.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm

# the GPU-accelerated trainer (cuBLAS; falls back to CPU without CUDA)
gpu_wubu.o: src/gpu_wubu.cu
	nvcc -O2 -c src/gpu_wubu.cu -o $@ -Xcompiler -fPIC

wubu_train: tools/wubu_train_cli.c src/wubu_runtime_dims.o src/wubu.o src/wubu_train.o src/wubu_backprop.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/wubu_grow.o src/wubu_plateau.o gpu_wubu.o src/wubu_diagnosis.o src/wubu_amoeba.o src/wubu_hive.o src/wubu_prover2.o src/wubu_hyper.o src/wubu_selfimprove.o src/wubu_rsi.o src/wubu_lineage.o src/wubu_contracts.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm $(CUDA_LIBS)

wubu_live_learn: tools/wubu_live_learn.c src/wubu_runtime_dims.o src/wubu.o src/wubu_train.o src/wubu_backprop.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/wubu_tokenizer_hf.o gpu_wubu.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm $(CUDA_LIBS)

wubu_boot: tools/wubu_boot.c src/wubu_runtime_dims.o src/wubu.o src/wubu_train.o src/wubu_backprop.o src/wubu_bi.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/safetensors_writer.o src/wubu_tensor_store.o src/gguf_reader.o gpu_wubu.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm $(CUDA_LIBS)

# the GPU-accelerated trainer: same CLI, weak-linked cuBLAS dispatch.
wubu_train_gpu: tools/wubu_train_cli.c src/wubu_runtime_dims.o src/wubu.o src/wubu_train.o src/wubu_backprop.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/wubu_grow.o src/wubu_plateau.o gpu_wubu.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -L/usr/local/cuda-13.1/lib64 -lcublas -lcudart -Wl,-rpath,/usr/local/cuda-13.1/lib64

test_wubu_save: tools/test_wubu_save.c src/wubu_runtime_dims.o src/wubu.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/safetensors_writer.o src/wubu_save.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@ models/wubu/model.safetensors

test_gpu_matmul: tools/gpu_matmul_check.c gpu_wubu.o
	$(CC) $(CFLAGS) -I include -o $@ tools/gpu_matmul_check.c gpu_wubu.o -lm -L/usr/local/cuda-13.1/lib64 -lcublas -lcudart -Wl,-rpath,/usr/local/cuda-13.1/lib64
	./$@

test_hyper: tools/test_hyper.c src/wubu_hyper.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_moe2: tools/test_moe2.c src/wubu_moe2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

wubu_cli: tools/wubu_cli.c src/wubu_runtime_dims.o src/wubu.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_tokenizer_hf.o src/wubu_dequant_nf4.o gpu_wubu.o
	$(CC) $(CFLAGS) -DWUBU_TOOL_VERSION=\"$(WUBU_VERSION)\" -I include -o $@ $^ -lm $(CUDA_LIBS)

test_wubu: tools/test_wubu.c src/wubu_runtime_dims.o src/wubu.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o gpu_wubu.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm $(CUDA_LIBS)
	./$@ models/wubu/model.safetensors

test_wubu_train: tools/test_wubu_train.c src/wubu_runtime_dims.o src/wubu.o src/wubu_train.o src/wubu_backprop.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o gpu_wubu.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm $(CUDA_LIBS)
	./$@ models/wubu/model.safetensors

src/wubu_backprop.o: include/wubu_backprop.h include/wubu_train.h include/wubu.h
src/wubu_train.o: include/wubu_train.h include/wubu.h include/wubu_backprop.h

test_backprop: tools/test_backprop.c src/wubu_backprop.o src/wubu_runtime_dims.o src/wubu.o src/wubu_train.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o gpu_wubu.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm $(CUDA_LIBS)
	./$@

# the GPU NS5 orthogonalization vs the CPU reference (square/wide/tall)
test_gpu_ns5: tools/test_gpu_ns5.c gpu_wubu.o
	$(CC) $(CFLAGS) -I/usr/local/cuda-13.1/include -fopenmp -I include -o test_gpu_ns5 tools/test_gpu_ns5.c gpu_wubu.o -lm -L/usr/local/cuda-13.1/lib64 -lcublas -lcudart
	./$@

test_gpu_attn: tools/test_gpu_attn.c gpu_wubu.o
	$(CC) $(CFLAGS) -I/usr/local/cuda-13.1/include -fopenmp -I include -o test_gpu_attn tools/test_gpu_attn.c gpu_wubu.o -lm -L/usr/local/cuda-13.1/lib64 -lcublas -lcudart
	./$@

# the folded sin/cos (Silas Lock via Kaze's folded polynomial)
test_foldmath: tools/test_foldmath.c
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# the U-Bus substrate: agnostic dispatch + the roofline selector
test_ubus: tools/test_ubus.c src/wubu_ubus.o gpu_wubu.o
# the agentic-corpus pipeline closes (the Orchard/tau-bench/Hermes wave):
# trajectory-level GRPO (FD-verified), masked-observation SFT, user simulator
test_traj_grpo: tools/test_traj_grpo.c src/wubu_traj_grpo.c include/wubu_traj_grpo.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_traj_grpo.c src/wubu_traj_grpo.c -lm
	./$@
test_grpo_coherence: tools/test_grpo_coherence.c src/wubu_grpo_coherence.c src/wubu_traj_grpo.c src/wubu_grpo_kalman.o include/wubu_grpo_coherence.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_grpo_coherence.c src/wubu_grpo_coherence.c src/wubu_traj_grpo.c src/wubu_grpo_kalman.o -lm
	./test_grpo_coherence
test_traj_sft: tools/test_traj_sft.c src/wubu_traj_sft.c include/wubu_traj_sft.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_traj_sft.c src/wubu_traj_sft.c
	./$@
test_user_sim: tools/test_user_sim.c src/wubu_user_sim.c include/wubu_user_sim.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_user_sim.c src/wubu_user_sim.c
	./$@
test_credit_dbstate: tools/test_credit_dbstate.c src/wubu_credit_sft.c src/wubu_dbstate.c \
	include/wubu_credit_sft.h include/wubu_dbstate.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_credit_dbstate.c src/wubu_credit_sft.c src/wubu_dbstate.c
	./$@
test_width: tools/test_width.c src/wubu_width.c src/wubu.c src/wubu_moe2.c \
	include/wubu_width.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_width.c src/wubu_width.c src/wubu.c src/wubu_moe2.c src/safetensors_reader.c src/wubu_dequant_nf4.c -lm
	./$@
test_seed: tools/test_seed.c src/wubu_seed.c include/wubu_seed.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_seed.c src/wubu_seed.c
	./$@
test_ambig: tools/test_ambig.c src/wubu_ambig.c include/wubu_ambig.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_ambig.c src/wubu_ambig.c -lm
	./$@
test_eval: tools/test_eval.c src/wubu_eval.c src/wubu_dbstate.c src/wubu_passk.c \
	include/wubu_eval.h include/wubu_dbstate.h include/wubu_passk.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_eval.c src/wubu_eval.c src/wubu_dbstate.c src/wubu_passk.c -lm
	./$@
test_recency: tools/test_recency.c src/wubu_recency.c include/wubu_recency.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_recency.c src/wubu_recency.c -lm
	./$@
test_epcap: tools/test_epcap.c src/wubu_epcap.c include/wubu_epcap.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_epcap.c src/wubu_epcap.c
	./$@
test_passk: tools/test_passk.c src/wubu_passk.c include/wubu_passk.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_passk.c src/wubu_passk.c -lm
	./$@
test_rollout: tools/test_rollout.c src/wubu_rollout.c include/wubu_rollout.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_rollout.c src/wubu_rollout.c -lm
	./$@
test_fmt: tools/test_fmt.c src/wubu_fmt.c include/wubu_fmt.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_fmt.c src/wubu_fmt.c
	./$@
test_mix: tools/test_mix.c src/wubu_mix.c include/wubu_mix.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_mix.c src/wubu_mix.c
	./$@
test_dedup: tools/test_dedup.c src/wubu_dedup.c include/wubu_dedup.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_dedup.c src/wubu_dedup.c
	./$@
test_bear_mlp: tools/test_bear_mlp.c src/wubu_bear_mlp.c include/wubu_bear_mlp.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_bear_mlp.c src/wubu_bear_mlp.c -lm
	./$@
test_gaad: tools/test_gaad.c src/wubu_gaad.c include/wubu_gaad.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_gaad.c src/wubu_gaad.c -lm
	./$@
test_uuid: tools/test_uuid.c src/wubu_uuid.c include/wubu_uuid.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_uuid.c src/wubu_uuid.c
	./$@
test_optim: tools/test_optim.c src/wubu_optim.o src/wubu.o src/wubu_train.o src/wubu_backprop.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/wubu_runtime_dims.o src/qlearner.o src/rsgd.o src/wubu_mobius.o src/gguf_reader.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ tools/test_optim.c src/wubu_optim.o src/wubu.o src/wubu_train.o src/wubu_backprop.o src/wubu_moe2.o src/safetensors_reader.o src/wubu_dequant_nf4.o src/wubu_runtime_dims.o src/qlearner.o src/rsgd.o src/wubu_mobius.o src/gguf_reader.o -lm
	./$@
test_fold_sincos8: tools/test_fold_sincos8.c include/wubu_foldmath.h
	$(CC) $(CFLAGS) -march=native -mfma -I include -o $@ tools/test_fold_sincos8.c -lm
	./$@
test_masked_ce: tools/test_masked_ce.c src/wubu_masked_ce.c include/wubu_masked_ce.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_masked_ce.c src/wubu_masked_ce.c -lm
	./$@
test_plateau: tools/test_plateau.c src/wubu_plateau.c include/wubu_plateau.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_plateau.c src/wubu_plateau.c
	./$@
src/wubu_ubus.o: include/wubu_ubus.h src/wubu_ubus.c
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_ubus.c

test_grow: tools/test_grow.c src/wubu_grow.c src/wubu.c src/wubu_backprop.c src/wubu_train.c \
	src/safetensors_reader.c src/wubu_dequant_nf4.c src/wubu_moe2.c src/wubu_ubus.c \
	include/wubu_grow.h include/wubu.h include/wubu_backprop.h include/wubu_train.h
	$(CC) $(CFLAGS) -I include -o $@ tools/test_grow.c src/wubu_grow.c src/wubu.c src/wubu_backprop.c src/wubu_train.c src/safetensors_reader.c src/wubu_dequant_nf4.c src/wubu_moe2.c src/wubu_ubus.c -lm
	./$@

moondream_cli: tools/moondream_cli.c src/wubu_moondream.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm

test_moondream: tools/test_moondream.c src/wubu_moondream.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_hybrid: tools/test_hybrid.c src/wubu_hybrid.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_vision: tools/test_vision.c src/wubu_vision.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_compress2: tools/test_compress2.c src/wubu_compress2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_compress: tools/test_compress.c src/wubu_compress.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_fuzz: tools/test_fuzz.c src/wubu_fuzz.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_neurom: tools/test_neurom.c src/wubu_neurom.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_rsi: tools/test_rsi.c src/wubu_rsi.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_linattn2: tools/test_linattn2.c src/wubu_linattn2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_linattn: tools/test_linattn.c src/wubu_linattn.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_token2: tools/test_token2.c src/wubu_token2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_token: tools/test_token.c src/wubu_token.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_pim2: tools/test_pim2.c src/wubu_pim2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_pim: tools/test_pim.c src/wubu_pim.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_serve2: tools/test_serve2.c src/wubu_serve2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_serve: tools/test_serve.c src/wubu_serve.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_pref2: tools/test_pref2.c src/wubu_pref2.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_pref: tools/test_pref.c src/wubu_pref.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_hopfield4: tools/test_hopfield4.c src/wubu_hopfield4.o src/wubu_hopfield3.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_hopfield3: tools/test_hopfield3.c src/wubu_hopfield3.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_evict2026c: tools/test_evict2026c.c src/wubu_evict2026c.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_evict2026b: tools/test_evict2026b.c src/wubu_evict2026b.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_evict2026: tools/test_evict2026.c src/wubu_evict2026.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_freeenergy: tools/test_freeenergy.c src/wubu_freeenergy.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_align: tools/test_align.c src/wubu_align.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_hopfield: tools/test_hopfield.c src/wubu_hopfield.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_debt: tools/test_debt.c src/wubu_si.o src/wubu_der.o src/wubu_reverify.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_energy: tools/test_energy.c src/wubu_energy.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_ff: tools/test_ff.c src/wubu_gp.o src/wubu_acq.o src/wubu_bo.o src/wubu_uq.o src/wubu_active.o src/wubu_bandit.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_gg: tools/test_gg.c src/wubu_reinforce.o src/wubu_policy.o src/wubu_actor_critic.o src/wubu_ppo.o src/wubu_dqn.o src/wubu_value.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_hh: tools/test_hh.c src/wubu_specdec.o src/wubu_pagedkv.o src/wubu_moeroute.o src/wubu_contbatch.o src/wubu_medusa.o src/wubu_quantkv.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@
# doc B08/H03/E06/F02: remaining CPU-closable cores (NVFP4, Hadamard,
# wide all-reduce, equiv-check). MLA (A08/E02) is in wubu_mla.c/test_mla.
test_more_cores: tools/test_more_cores.c src/wubu_nvfp4.o src/wubu_hadamard.o src/wubu_expert_allreduce.o src/wubu_equiv_check.o src/wubu_fp8.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc 007/D05: localhost KV transfer layer (NIXL/UCX analog)
test_kv_transfer: tools/test_kv_transfer.c src/wubu_kv_transfer.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_kv_mapper: tools/test_kv_mapper.c src/wubu_kv_mapper.o src/wubu_kvfs.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc AN25: cross-model KV transfer ridge mapper
# Compiled WITHOUT -ffast-math — the Cholesky ridge solver needs
# stable floating-point ordering (fast-math reassociation can make
# the factorization numerically unstable, producing wrong maps).
src/wubu_kv_mapper.o: src/wubu_kv_mapper.c include/wubu_kv_mapper.h include/wubu_kvfs.h
	$(CC) -O3 -march=native -funroll-loops -ftree-vectorize -Wall -Wextra -Wno-unused-parameter -I include -c -o $@ $<

wubu_kv_calib: tools/wubu_kv_calib.c src/wubu_kv_mapper.o src/wubu_kvfs.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm

# doc D03/D04: chunked prefill + disaggregated PD
test_chunked_prefill: tools/test_chunked_prefill.c src/wubu_chunked_prefill.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc F02: SMT-style GEMV equivalence checking
test_smt_check: tools/test_smt_check.c src/wubu_smt_check.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc A06: LMCache prefix+PD KV persistence
test_lmcache: tools/test_lmcache.c src/wubu_lmcache.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc 001: adaptive KV hot-path integration test
test_adaptive_hotpath: tools/test_adaptive_hotpath.c src/wubu_kv_adaptive.o src/wubu_kv_runtime.o src/wubu_kv_select.o src/wubu_roofline.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# AGI OS integration test — all modules end-to-end
test_agi_os_integration: tools/test_agi_os_integration.c src/wubu_kv_adaptive.o src/wubu_rope_prefetch.o src/wubu_flash_prefill.o src/wubu_soa.o src/wubu_lmcache.o src/wubu_smt_check.o src/wubu_expert_choice.o src/wubu_chunked_prefill.o src/wubu_mla.o src/wubu_kv_cacheline.o src/wubu_kv_runtime.o src/wubu_kv_select.o src/wubu_roofline.o src/wubu_scheduler.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

# doc 015: FlashDecoding-style parallel KV-load decode attention
test_flashdecode: tools/test_flashdecode.c src/wubu_flashdecode.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 013: QuaRot/Hadamard rotation invariance + outlier-suppression
test_rotate: tools/test_rotate.c src/wubu_rotate.o $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

# doc 009: Bounded kernel equivalence test
test_gemv_equivalence: tools/test_gemv_equivalence.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

test_gemv_int4: tools/test_gemv_int4.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

test_kvcache_quant: tools/test_kvcache_quant.c $(CPU_OBJ)
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -fopenmp
	./$@

test_moe: tools/test_moe.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_moe_hyperbolic: tools/test_moe_hyperbolic.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_full_moe: tools/test_full_moe.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_rope_t2: tools/test_rope_t2.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

gen_text: tools/gen_text.c $(CPU_OBJ) src/wubu_tokenizer.o src/wubu_kernel_cuda.o
	$(CXX) $(CFLAGS) -DWUBU_TOOL_VERSION=\"$(WUBU_VERSION)\" -DWUBU_ENABLE_CUDA -I include -o $@ $< $(CPU_OBJ) src/wubu_tokenizer.o src/wubu_kernel_cuda.o $(LDFLAGS) -L$(CUDA_LIBDIR) -lcudart -lstdc++

# CPU-only gen_text (recompiles wubu_model + wubu_moe without GPU_SUPPORT)
gen_text_cpu: CFLAGS_FILTERED = $(filter-out -I$(CUDA_INC),$(CFLAGS))
gen_text_cpu: src/wubu_model_cpu.o src/wubu_moe_cpu.o src/wubu_dense_ffn.o $(filter-out src/wubu_moe.o src/wubu_model.o,$(CORE_OBJ)) src/wubu_tokenizer.o
	$(CC) $(CFLAGS_FILTERED) -o $@ tools/gen_text.c src/wubu_model_cpu.o src/wubu_moe_cpu.o src/wubu_dense_ffn.o $(filter-out src/wubu_moe.o src/wubu_model.o,$(CORE_OBJ)) src/wubu_tokenizer.o $(LDFLAGS)
	@echo "gen_text_cpu built (CPU-only, no GPU support)"

src/wubu_model_cpu.o: src/wubu_model.c include/wubu_model.h include/wubu_ssm.h include/wubu_moe.h include/gguf_reader.h
	$(CC) $(CFLAGS) -o $@ -c $<

# CPU-only wubu_moe (no GPU_SUPPORT)
src/wubu_moe_cpu.o: src/wubu_moe.c include/wubu_moe.h include/wubu_ssm.h
	$(CC) $(CFLAGS) -o $@ -c $<

run_bos: tools/run_bos.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ tools/run_bos.c $(MODEL_OBJ) $(LDFLAGS)

# Debug build (gdb/ASAN). Compiles gen_text + model objects with -g -O0,
# no GPU_SUPPORT, single-file objects (no _cpu variant clash).
gen_text_dbg: CFLAGS_DBG = -g -O0 -I include $(CUDA_INC) -fopenmp -Wall
gen_text_dbg: tools/gen_text.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS_DBG) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o src/wubu_tokenizer_hf.o $(LDFLAGS)

gen_text_asan: CFLAGS_ASAN = -g -O1 -fsanitize=address -I include $(CUDA_INC) -fopenmp
gen_text_asan: tools/gen_text.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS_ASAN) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o src/wubu_tokenizer_hf.o $(LDFLAGS)

# Probe: load real Qwen3.6-27B (MAX_LAYERS=1) and print layer-0 weight
# pointers + state buffers, to diagnose SSM forward crashes.
test_probe_qwen: tools/test_probe_qwen.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o $(LDFLAGS)

# Verify the ds4-ssd MoE decode bank pages real KAT experts from the source
# checkpoint shards (no sidecar copy).
test_kat_decode_bank: tools/test_kat_decode_bank.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -I include -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o $(LDFLAGS)

# ASAN variant for pinning SSM-forward heap bugs.
test_probe_qwen_asan: CFLAGS_ASAN = -O1 -g -fsanitize=address -mavx2 -mfma -I include $(CUDA_INC) -fopenmp
test_probe_qwen_asan: tools/test_probe_qwen.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS_ASAN) -o $@ $< $(MODEL_OBJ) src/wubu_tokenizer.o $(LDFLAGS)

gen_text_mtp: tools/gen_text_mtp.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $(filter %.c %.o,$^) $(LDFLAGS)

# GPU-enabled kernel_backends (recompiled with WUBU_ENABLE_CUDA so the
# CUDA backend probe+register actually runs).
src/wubu_kernel_backends_gpu.o: src/wubu_kernel_backends.c
	$(CC) $(CFLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

# GPU_OBJ without files already in CORE_OBJ (avoids duplicate symbol errors).
# NOTE: must be defined BEFORE gen_text_gpu uses it — make expands rule
# prerequisites immediately (definition order matters), recipes lazily.
GPU_OBJ_NODUP = src/wubu_model_gpu.o src/wubu_gpu_weight_cache.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/gpu_moe_kernel.o src/gpu_ssm_recurrence.o src/wubu_backend_cuda.o

# gen_text_gpu: link CORE_OBJ but replace wubu_kernel_backends.o with GPU version
# that has CUDA probe enabled. Also adds CUDA kernels + GPU weight cache.
gen_text_gpu: tools/gen_text.c $(filter-out src/wubu_kernel_backends.o,$(CORE_OBJ) src/wubu_dense_ffn.o) src/wubu_kernel_backends_gpu.o src/wubu_tokenizer.o src/wubu_tokenizer_hf.o $(CUDA_OBJ) $(GPU_OBJ_NODUP)
	$(CXX) $(CFLAGS) -DGPU_SUPPORT -DWUBU_ENABLE_CUDA -o $@ tools/gen_text.c $(filter-out src/wubu_kernel_backends.o,$(CORE_OBJ)) src/wubu_kernel_backends_gpu.o src/wubu_tokenizer.o src/wubu_tokenizer_hf.o $(CUDA_OBJ) $(GPU_OBJ_NODUP) $(LDFLAGS) -L$(CUDA_LIBDIR) -lcublas -lcudart

test_tok_debug: tools/test_tok_debug.c src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ref_dumper: tools/ref_dumper.cpp
	$(CXX) $(CFLAGS) -std=c++17 -I $(HOME)/llama.cpp/include -I $(HOME)/llama.cpp/ggml/include -o $@ $^ $(LDFLAGS) $(HOME)/llama.cpp/build/bin/libllama.so $(HOME)/llama.cpp/build/bin/libggml.so $(HOME)/llama.cpp/build/bin/libggml-cpu.so $(HOME)/llama.cpp/build/bin/libggml-base.so -Wl,-rpath,$(HOME)/llama.cpp/build/bin

ref_dumper_mtp: tools/ref_dumper_mtp.cpp
	$(CXX) $(CFLAGS) -std=c++17 -I $(HOME)/llama.cpp/include -I $(HOME)/llama.cpp/src -I $(HOME)/llama.cpp/ggml/include -o $@ $^ $(LDFLAGS) $(HOME)/llama.cpp/build/bin/libllama.so $(HOME)/llama.cpp/build/bin/libggml.so $(HOME)/llama.cpp/build/bin/libggml-cpu.so $(HOME)/llama.cpp/build/bin/libggml-base.so -Wl,-rpath,$(HOME)/llama.cpp/build/bin

test_quantized_matmul: tools/test_quantized_matmul.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_vec_dot_types: tools/test_vec_dot_types.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_iq_dot: tools/test_iq_dot.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

load_model: tools/load_model_layer.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_gpu: tools/test_gpu.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

bench_e2e: tools/bench_e2e.c src/bench.o $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_parallel_scan: tools/test_parallel_scan.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_fused: tools/test_fused.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_fused_vs_old: tools/test_fused_vs_old.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

debug_beta_layout: tools/debug_beta_layout.c src/gguf_reader.o src/dequant_iq2_xxs.o src/cuda_kernels.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

verify_phase26: tools/verify_phase26_fusions.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

train_integrated: tools/train_integrated.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o $(CUDA_OBJ) src/bench.o $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

infer_text: tools/infer_text.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

infer_text_gpu: tools/infer_text_gpu.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ) $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_cuda_kernels: tools/test_cuda_kernels.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

# Compare our logits vs llama.cpp
compare_logits: tools/compare_logits.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o
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
infer_moe: tools/infer_moe.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_moe_lazy: tools/infer_moe_lazy.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_unified: tools/infer_unified.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_vision: tools/infer_vision.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

infer_vision_text: tools/infer_vision_text.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_vision_real: tools/test_vision_real.c $(MODEL_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ) $(GPU_OBJ)
	$(CXX) $(CFLAGS) -DGPU_SUPPORT -o $@ tools/test_vision_real.c $(MODEL_OBJ) $(CUDA_OBJ) $(GPU_OBJ) $(LDFLAGS) -L$(CUDA_LIBDIR) -lcublas -lcudart
	@echo "test_vision_real built (GPU vision + text)"

infer_vision_text_gpu: tools/infer_vision_text_gpu_nvcc.o $(MODEL_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ) src/cuda_vision.o $(GPU_OBJ)
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lcublas -lcudart -lstdc++

tools/infer_vision_text_gpu_nvcc.o: tools/infer_vision_text_gpu.cu include/cuda_vision.h include/wubu_vision.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

infer_poincare: tools/infer_poincare.c src/bench.o $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

tailslayer: tools/tailslayer.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o src/bench.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

infer_vision_gpu: tools/infer_vision_gpu.o $(CORE_OBJ) src/wubu_dense_ffn.o src/cuda_vision.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

tools/infer_vision_gpu.o: tools/infer_vision_gpu.cu include/cuda_vision.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

src/cuda_vision.o: src/cuda_vision.cu include/cuda_vision.h include/wubu_vision.h
	$(NVCC) $(NVCC_FLAGS) -DWUBU_ENABLE_CUDA -c -o $@ $<

test_256k: tools/test_256k.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_256k_context: tools/test_256k_context.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_256k_forward: tools/test_256k_forward.c $(MODEL_OBJ) src/wubu_dense_ffn.o gen_fixture_safetensors_model
	$(CC) $(CFLAGS) -o $@ tools/test_256k_forward.c $(MODEL_OBJ) $(LDFLAGS)
	./gen_fixture_safetensors_model
	./$@

# Chunked 256K prefill proof: builds the binary; run explicitly (heavy 256K).
test_256k_chunked: tools/test_256k_chunked.c $(MODEL_OBJ) src/wubu_dense_ffn.o gen_fixture_safetensors_model
	$(CC) $(CFLAGS) -o $@ tools/test_256k_chunked.c $(MODEL_OBJ) $(LDFLAGS)
	./gen_fixture_safetensors_model

test_kv_cache: tools/test_kv_cache.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

tokenize_corpus: tools/tokenize_corpus.c src/wubu_tokenizer.o src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_real: tools/train_real.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_backprop: tools/train_backprop.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

train_gpu: tools/train_gpu.c src/bench.o $(MODEL_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ) src/wubu_tokenizer.o $(GPU_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

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

test_dequant: tools/test_dequant.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_forward: tools/check_forward.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_iq2_xxs_dot: tools/test_iq2_xxs_dot.c src/gguf_reader.o src/dequant_iq2_xxs.o src/wubu_moe.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# ================================================================
# WuBu-35M mustard-seed engine (the Revolver Doctrine closure S1-S5)
# ================================================================
WUBU_RUNTIME_DIMS_OBJ = src/wubu_runtime_dims.o src/wubu.o \
             src/safetensors_reader.o src/wubu_moe2.o src/wubu_dequant_nf4.o

src/wubu_runtime_dims.o: src/wubu_runtime_dims.c include/wubu_runtime_dims.h include/safetensors_reader.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu.o: src/wubu.c include/wubu.h include/wubu_runtime_dims.h include/safetensors_reader.h include/wubu_moe2.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/wubu_dequant_nf4.o: src/wubu_dequant_nf4.c
	$(CC) $(CFLAGS) -c -o $@ $<

# The 35M gate: load the REAL released checkpoint, probe its geometry
# (probe-don't-assume), forward, generate. MODEL=... to point at a
# checkpoint (default models/wubu/model.safetensors).
test_wubu_arch: tools/test_wubu_arch.c $(WUBU_RUNTIME_DIMS_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "test_wubu_arch built (WuBu-35M revolver gate)"

test_wubu_arch_run: test_wubu_arch
	./test_wubu_arch $(MODEL)

# Theory/08: aligned-geometry gate. Proves the redesigned 35M engine
# geometry (dim=512, heads=8, head_dim=64, ffn=2048) tiles evenly across
# every quant block + SIMD width, so zero-dequant compute applies.
test_wubu_alignment: tools/test_wubu_alignment.c src/wubu_runtime_dims.o src/wubu_dims_stub.o src/wubu_arena.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

# THEORY/10 gate: fractal stacking on the sphere surface (self-contained,
# no engine deps — pure bit-ops + fp math).
test_fractal_index: tools/test_fractal_index.c
	$(CC) $(CFLAGS) -I include -o $@ tools/test_fractal_index.c -lm
	./$@

# THEORY/10 gate: the Ecosystem of Spheres — physics router (potential
# wells, zero learned params), small-active/huge-total, grow/shrink.
test_ecosystem: tools/test_ecosystem.c src/wubu_ecosystem.o src/wubu_mobius.o src/wubu_hive.o src/gguf_reader.o src/wubu_moe_hyperbolic.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_ecosystem.o: src/wubu_ecosystem.c include/wubu_ecosystem.h include/wubu_mobius.h include/wubu_hive.h include/wubu_moe_hyperbolic.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_ecosystem.c

# THE ROUTER SLOT (Revolver): one vtable, any physics registers.
# wubu_moe.o (+ its quantized-matmul chain) is linked so the
# engine-wiring probe proves the slot drives a real MoE forward.
test_router: tools/test_router.c src/wubu_router.o src/wubu_router_physics.o src/wubu_scale.o src/wubu_mem_budget.o src/wubu_hwcaps.o src/wubu_ecosystem.o src/wubu_mobius.o src/wubu_hive.o src/gguf_reader.o src/wubu_moe_hyperbolic.o src/wubu_gravity.o src/wubu_moe.o src/quantized_matmul.o src/quantized_dot_generic.o src/dequant_iq2_xxs.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_router.o: src/wubu_router.c include/wubu_router.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_router.c

src/wubu_router_physics.o: src/wubu_router_physics.c include/wubu_router.h include/wubu_ecosystem.h include/wubu_moe_hyperbolic.h include/wubu_gravity.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_router_physics.c

# THEORY/11 gate: SCALE-TO-FIT — one checkpoint, any hardware (CM4 → super).
test_scale_to_fit: tools/test_scale_to_fit.c src/wubu_scale.o src/wubu_mem_budget.o src/wubu_hwcaps.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_scale.o: src/wubu_scale.c include/wubu_scale.h include/wubu_hwcaps.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_scale.c

# AM01 gate: the symmetric amoeba — shrink operators (depth + width +
# train-state pair). Mirrors wubu_grow.
test_amoeba_shrink: tools/test_amoeba_shrink.c src/wubu_shrink.o src/wubu_grow.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_shrink.o: src/wubu_shrink.c include/wubu_shrink.h include/wubu_grow.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_shrink.c

# WB06/AN06 gate: THE ENCODER SLOT — modality-agnostic base seam.
test_encoder: tools/test_encoder.c src/wubu_encoder.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_encoder.o: src/wubu_encoder.c include/wubu_encoder.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_encoder.c

# WB06 gate: OUR OWN encoder (made, not imported) — imgenc + audio
# wired into the slot with a shared embedding space.
test_encoder_ours: tools/test_encoder_ours.c src/wubu_encoder_impl.o src/wubu_encoder.o src/wubu_imgenc.o src/wubu_audio.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_encoder_impl.o: src/wubu_encoder_impl.c include/wubu_encoder_impl.h include/wubu_encoder.h include/wubu_imgenc.h include/wubu_audio.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_encoder_impl.c

# AN16 G3 gate: THE ENCODER IS A MOUNT — the encoder lives in the
# universal file-system space (KVFS lifts the heavy work).
test_enc_fs: tools/test_enc_fs.c src/wubu_enc_fs.o src/wubu_kvfs.o src/wubu_encoder_impl.o src/wubu_encoder.o src/wubu_imgenc.o src/wubu_audio.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_enc_fs.o: src/wubu_enc_fs.c include/wubu_enc_fs.h include/wubu_kvfs.h include/wubu_encoder.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_enc_fs.c

# research/065 gate: THE USER SPACE IS THE TRAINING DATA — user files
# interpreted in the namespace (chunked + usage ledger + incremental
# + ALL FILE TYPES: images/video/office/pdf via our decoders).
test_userfs: tools/test_userfs.c src/wubu_userfs.o src/wubu_kvfs.o src/wubu_encoder_impl.o src/wubu_encoder.o src/wubu_imgenc.o src/wubu_audio.o src/wubu_png.o src/wubu_jpeg.o src/wubu_video.o src/wubu_ooxml.o src/wubu_pdf.o src/wubu_zip.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -lz
	./$@

src/wubu_userfs.o: src/wubu_userfs.c include/wubu_userfs.h include/wubu_kvfs.h include/wubu_encoder.h include/wubu_audio.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_userfs.c

# THE ALL-FORMATS GATE: images (PNG/JPEG), video (AVI/MJPEG), office
# (docx/xlsx/pptx ZIP+XML), PDF — all decoded by OUR code on zlib.
test_ingest_all: tools/test_ingest_all.c src/wubu_png.o src/wubu_jpeg.o src/wubu_zip.o src/wubu_ooxml.o src/wubu_pdf.o src/wubu_video.o src/wubu_imgenc.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm -lz
	./$@

src/wubu_png.o: src/wubu_png.c include/wubu_png.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_png.c

src/wubu_jpeg.o: src/wubu_jpeg.c include/wubu_jpeg.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_jpeg.c

src/wubu_zip.o: src/wubu_zip.c include/wubu_zip.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_zip.c

src/wubu_ooxml.o: src/wubu_ooxml.c include/wubu_ooxml.h include/wubu_zip.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_ooxml.c

src/wubu_pdf.o: src/wubu_pdf.c include/wubu_pdf.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_pdf.c

src/wubu_video.o: src/wubu_video.c include/wubu_video.h include/wubu_jpeg.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_video.c

# research/066 gate: THE VHF CANVAS — the coordinate-addressable
# latent field in C11 (the user's VHF encoder prior art).
test_canvas: tools/test_canvas.c src/wubu_canvas.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_canvas.o: src/wubu_canvas.c include/wubu_canvas.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_canvas.c

# research/067 gate: THE UNIVERSAL CODEC — the visual audio Kodak in
# C11 (audio <-> image reversible; one encoder for every modality).
test_codec: tools/test_codec.c src/wubu_codec.o src/wubu_imgenc.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_codec.o: src/wubu_codec.c include/wubu_codec.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_codec.c

# AM03 gate: HETEROGENEOUS PRECISION — the Escha ladder (research/046).
test_precision_plan: tools/test_precision_plan.c src/wubu_precision_plan.o src/wubu_hwcaps.o src/wubu_mem_budget.o src/wubu_scale.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

src/wubu_precision_plan.o: src/wubu_precision_plan.c include/wubu_precision_plan.h
	$(CC) $(CFLAGS) -I include -c -o $@ src/wubu_precision_plan.c

# The runtime-dims probe gate (the Revolver Doctrine): proves the loader
# reads REAL geometry from the checkpoint tensor shapes (no fixed geometry).
test_runtime_dims: tools/test_runtime_dims.c $(WUBU_RUNTIME_DIMS_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_runtime_dims_run: test_runtime_dims
	./test_runtime_dims $(MODEL)

# ================================================================
# Test runners
test: test_ssm
	./test_ssm

test_nested_ssm_run: test_nested_ssm
	./test_nested_ssm

test_poincare_gqa_run: test_poincare_gqa
	./test_poincare_gqa

test_tst: tools/test_tst.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_tst_run: test_tst
	./test_tst

test_gpu_run: test_gpu
	./test_gpu

bench_e2e_run: bench_e2e
	./bench_e2e

train_stub_run: train_stub
	./train_stub

test_regression: tools/test_regression.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_tokenizer.o $(CUDA_OBJ)
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_gpu_poincare: tools/test_gpu_poincare.c $(CORE_OBJ) src/wubu_dense_ffn.o $(CUDA_OBJ) src/bench.o
	$(CC) $(CFLAGS) $(CUDA_INC) -o $@ $^ $(LDFLAGS) $(CUDA_LIBS) -L$(CUDA_LIBDIR) -lstdc++

test_rsgd: tools/test_rsgd.c $(RSGD_OBJ) src/gguf_reader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_backward: tools/test_backward.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_bwd_model: tools/test_bwd_model.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_backward_simple: tools/test_backward_simple.c $(CORE_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# CPU timing + hedged spec tests (from tailslayer pattern)
test_cpu_timing: tools/test_cpu_timing.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lpthread

check_weights: tools/check_weights.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

check_ssm_a: tools/check_ssm_a.c $(MODEL_OBJ) src/wubu_dense_ffn.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Gemma 4 12B test binaries
test_gemma4: tools/test_gemma4.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_gemma4_model.o
	$(CC) $(CFLAGS) -o $@ tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o $(LDFLAGS)

test_gemma4_gpu: tools/test_gemma4.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_gemma4_model.o src/gpu_gemma4.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o
	$(CXX) $(CFLAGS) $(CUDA_INC) -DGPU_SUPPORT -o $@ tools/test_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o src/gpu_gemma4.o src/gpu_gemma4_forward.o src/gpu_quant_matmul.o src/gpu_quant_matmul_row_major.o src/cuda_kernels.o $(LDFLAGS) -L$(CUDA_LIBDIR) -lcublas -lcudart -lstdc++

gen_text_gemma4: tools/gen_text_gemma4.c $(MODEL_OBJ) src/wubu_dense_ffn.o src/wubu_gemma4_model.o
	$(CC) $(CFLAGS) -o $@ tools/gen_text_gemma4.c $(MODEL_OBJ) src/wubu_gemma4_model.o $(LDFLAGS)

clean:
	rm -f test_nested_ssm test_poincare_ssm test_poincare_gqa load_model test_model test_gpu tokenize_corpus test_moe test_moe_hyperbolic train_real bench_e2e verify_iq2s inspect_iq2s inspect_model train_backprop train_gpu test_gpu_poincare test_rsgd test_backward test_cpu_timing infer_moe infer_moe_lazy infer_unified infer_poincare test_256k test_kv_cache test_tst test_nested_moe_router_backward tailslayer test_iq2_dequant test_iq2_xxs_dot test_gemma4 test_gemma4_gpu gen_text_gemma4 test_ops src/*.o tools/*.o
	rm -rf plugins/*.so

# Dynamic plugin builds (research 066-G1)
plugins/libonnx_format.so: tools/plugin_onnx.c
	@mkdir -p plugins
	$(CC) -shared -fPIC -O2 -Wall -I include -o plugins/libonnx_format.so tools/plugin_onnx.c -lm
	@echo "Built ONNX format plugin (stub — real impl tracked as gap)"

plugin_onnx: plugins/libonnx_format.so

SRCS += src/wubu_kernel.c
src/wubu_fast_attn.o: src/wubu_fast_attn.c include/wubu_fast_attn.h
	$(CC) $(CFLAGS) -fopenmp -I include -c -o $@ $<

test_fast_attn: tools/test_fast_attn.c src/wubu_fast_attn.o src/wubu_4kv.o src/wubu_polarquant.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_fast_attn_q8: tools/test_fast_attn_q8.c src/wubu_fast_attn.o src/wubu_4kv.o src/wubu_polarquant.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_eagle: tools/test_eagle.c src/wubu_eagle.o
	$(CC) $(CFLAGS) -I include -o $@ tools/test_eagle.c src/wubu_eagle.o -lm
	./$@

test_polarquant: tools/test_polarquant.c src/wubu_polarquant.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_polarquant: tools/test_polarquant.c src/wubu_polarquant.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_polarquant_cache: tools/test_polarquant_cache.c src/wubu_polarquant.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_polarquant_cache: tools/test_polarquant_cache.c src/wubu_polarquant.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_polarquant_scale: tools/test_polarquant_scale.c src/wubu_polarquant.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_polar_pso: tools/test_polar_pso.c src/wubu_polar_pso.o src/wubu_polarquant.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_polarquant_benchmark: tools/test_polarquant_benchmark.c src/wubu_polarquant.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_polarquant_benchmark: tools/test_polarquant_benchmark.c src/wubu_polarquant.o src/wubu_polar_pso.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_q8k_pqv: tools/test_q8k_pqv.c src/wubu_fast_attn.o src/wubu_4kv.o src/wubu_polarquant.o src/wubu_q8.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_splitk: tools/test_splitk.c src/wubu_fast_attn.o src/wubu_4kv.o src/wubu_polarquant.o src/wubu_mobius.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_cross_attn: tools/test_cross_attn.c src/wubu_cross_attn.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@

test_all: test_codec test_canvas test_ingest_all test_userfs test_enc_fs test_precision_plan test_encoder_ours test_encoder test_scale_to_fit test_amoeba_shrink test_router test_ecosystem test_fractal_index test_polarquant test_polarquant_cache test_polar_pso test_polarquant_benchmark test_fast_attn test_fast_attn_q8 test_q8k_pqv test_splitk test_cross_attn test_ring_attn test_nf4 test_4kv test_eagle test_soa test_awq test_gptq test_attn_gate test_rope_prefetch test_kv_cacheline test_scheduler test_mla test_expert_choice test_layer_skip test_smt_check test_self_cascade test_spec_cascade test_lmcache test_kv_adaptive test_delta_net test_chunked_prefill test_disagg_prefill_decode test_kv_transfer test_kv_mapper test_kv_evict test_thread_spec test_early_exit test_tandem_gamebud test_model_hwaccel test_fp8 test_ecs test_more_cores test_512k_budget test_medusa test_numerical_audit test_paged_kv test_smoothquant test_flashdecode test_gemv_int4 test_prefix_reuse test_continuous_batching test_flash_prefill test_ngram test_hive test_stream_kv test_kv_evict_h2o test_capacity_wall test_hugepage test_kv_budget test_wm_kv test_spec_tuner test_quant_selector test_kv_compress test_lruk test_sparse_attn test_attn_tune test_kv_shield test_ctx_manage test_lookahead test_sys_tune test_lm_infinite test_spec_variants test_more_spec test_misc_gaps test_bf16_gemv test_attn_kernels test_db_cross test_kv2026 test_kv2026b test_ttc test_kv2026c test_sys2026 test_linear_attn test_ternary test_agentic_kv test_dn2 test_parallel_spec test_moe_rag test_eval_qat test_pd_serve test_integrate test_agentic_os_mem test_capzero test_loopguard_planediv test_vecsearch test_causal_symbolic test_metagame_coord test_energy test_debt test_hopfield test_align test_freeenergy test_evict2026 test_evict2026b test_hopfield3 test_hopfield2 test_pref test_pref2 test_serve test_serve2 test_pim test_pim2 test_token test_token2 test_linattn test_linattn2 test_rsi test_neurom test_fuzz test_bridge test_metacog test_metagame2 test_bonzi2 test_bridge2 test_bonzi test_worldmodel_agentauth test_ax test_axi test_continual test_multimodal test_multiconsensus test_ee test_ff test_gg test_hh test_backprop test_gpu_ns5 test_ubus test_foldmath test_gpu_attn test_traj_grpo test_grpo_coherence test_traj_sft test_user_sim test_credit_dbstate test_grow test_plateau test_masked_ce test_dedup test_mix test_fmt test_rollout test_passk test_seed test_ambig test_recency test_epcap test_eval test_width test_enc_h3 test_dsv4 test_lfm test_megakernel test_multiteach test_backend_dispatch
	@echo "=== ALL TESTS PASSED ==="

test_nf4: tools/test_nf4.c src/wubu_nf4.o
	$(CC) $(CFLAGS) -I include -o $@ $^ -lm
	./$@

test_ring_attn: tools/test_ring_attn.c src/wubu_ring_attn.o
	$(CC) $(CFLAGS) -fopenmp -I include -o $@ $^ -lm
	./$@
