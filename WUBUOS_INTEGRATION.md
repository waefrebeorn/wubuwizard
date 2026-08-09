# wubuwizard ↔ WuBuOS Integration Plan

## Current State

### wubuwizard (Inference Engine)
- Pure C11 inference engine ("the Colonel")
- Loads GGUF + HuggingFace safetensors
- SSM/GQA/MoE forward paths
- KV-cache, repetition suppression (repeat_penalty + DRY)
- `gen_text` binary for CPU inference
- `api_server` for OpenAI-compatible HTTP

### WuBuOS (Operating System)
- ZealOS kernel + Win98/XP shell + Styx/9P namespace + Arch containers
- `wubu_spawn.c` — dependency-free fork/exec/wait for external programs
- `wubu_exec.c` — universal dispatcher for ELF/PE/Mach-O/WASM/DOS
- `wubu_container.c` — cgroup/seccomp/namespace isolation
- `wubu_archd_daemon.c` — service supervisor
- `wubu_realm.c` — cross-realm verification (DA-2 fail-closed)
- `wubu_selfimprove.c` — self-modification loop (DA-1/2/3)
- `wubu_verifier.h` — integration contract (bytropix-era verifier was folded into the main verifier)

## Integration Goals

### Short Term (WSL Host Path)
1. **wubu_spawn → gen_text**: Use WuBuOS's `wubu_spawn` to launch `gen_text` as a subprocess
   - Replace `popen()` calls in `wubu_tokenizer.c` with `wubu_spawn`
   - Replace `popen()` in `wubu_safetensors_shard.c` with direct opendir/glob or `wubu_spawn`
   - This provides process isolation, observability, and proper exit-code handling

2. **KV-cache as Styx file**: Export the KV-cache arena through `/n/kv/` namespace
   - `wubu_ns_bridge` already exports `/n/services/*`
   - Add `/n/kv/{layer}/{slot}` for live inspection during inference
   - Enables external tools to inspect/decrypt KV state

3. **Model loading through WuBuOS**: Use WuBuOS's file APIs for model loading
   - Replace `mmap()` + `pread()` in shim with `wubu_file_*` calls
   - Enables unified logging/metrics for model loads
   - Allows Styx/9P namespace access to model files

### Medium Term (Container Integration)
4. **gen_text as .wubu container**: Package `gen_text` as a `.wubu` container
   - Use `wubu_exec_container()` to launch inference in isolated namespace
   - Container gets access to `/n/models/` via `/n` mount
   - KV-cache exported to `/n/kv/` for inspection

5. **Inference as a WuBuOS service**: Register `gen_text` daemon with `wubu_archd`
   - Service lifecycle: start/stop/restart/status/reap
   - Auto-restart on crash (like other WuBuOS daemons)
   - Health check via `wubu_archd_svc_super.c`

6. **Unified logging**: Route all inference logs through WuBuOS trace system
   - Use `wubu_trace.c` for structured logging
   - Enable live inspection via `/n/trace/` namespace

### Long Term (Kernel Integration)
7. **Inference in kernel space**: Move inference into ZealOS ring-0
   - Requires `WUBU_BAREMETAL` build
   - Inference as a kernel module (like `wubu_math.c`)
   - Enables zero-copy KV-cache between user/kernel

8. **AGI agent loop in WuBuOS**: Integrate slermes (Hermes agent) with WuBuOS
   - But user says slermes is being handled separately — skip for now
   - Focus on wubuwizard + WuBuOS integration

## Technical Integration Points

### File: `src/runtime/wubu_spawn.h` (WuBuOS)
```c
int wubu_run_program(const char *file, char *const argv[], bool silent);
```
This is the key primitive — dependency-free, can be linked into wubuwizard.

### File: `src/runtime/wubu_exec.h` (WuBuOS)
```c
int64_t wubu_exec_linux_elf(const void *elf_data, size_t elf_size);
int64_t wubu_exec_win_pe(const void *pe_data, size_t pe_size);
int64_t wubu_exec_container(wubu_container_t *ct, const void *payload, size_t payload_size);
```
For launching `gen_text` as a native ELF or container.

### File: `src/runtime/styxfs_server.c` (WuBuOS)
```c
// Pattern for exporting state through 9P
static int styxfs_kv_read(Fid *fid, void *buf, size_t count, off_t offset);
```
Add KV-cache export following `nt_registry_styx_export()` pattern.

## WSL-Specific Considerations

### Hardware Abstraction
- WSL2: 6 P-cores pinned, F16 KV-cache, llvmpipe Vulkan
- No dedicated GPU (`/dev/dri` absent)
- Use `wubu_affinity.c` for CPU pinning
- Use `wubu_cuda_graph.c` conditional compilation for CUDA paths

### Memory Management
- ~13 GB RAM available
- Use BF16 lazy loading (mmap + on-demand dequant)
- ds4-ssd slot-bank for MoE experts
- Keep resident memory under 13 GB

### Build Target
Add to wubuwizard Makefile:
```makefile
# WSL integration target
WSL_OBJS = src/wubu_model.o src/wubu_moe.o src/wubu_ssm.o ...
WSL_LDFLAGS = -lm -fopenmp

gen_text_wsl: tools/gen_text.c $(WSL_OBJS)
	$(CC) $(CFLAGS) -o $@ tools/gen_text.c $(WSL_OBJS) $(WSL_LDFLAGS)
	@echo "gen_text_wsl built (WSL-optimized, no GPU)"
```

## Next Actions

1. ✅ Create `wubu_spawn` wrapper in wubuwizard for external program launch
2. ✅ Replace `popen()` in `wubu_safetensors_shard.c` with direct opendir/glob
3. ✅ Add KV-cache Styx export via `wubu_ns_bridge`
4. ⏳ Package `gen_text` as `.wubu` container
5. ⏳ Register inference as WuBuOS service
6. ⏳ Integrate with WuBuOS trace system for unified logging

## References

- `src/runtime/wubu_exec.c` — WuBuOS exec dispatcher
- `src/runtime/wubu_spawn.c` — dependency-free launcher
- `src/runtime/styxfs_server.c` — 9P export pattern
- `src/runtime/wubu_ns_bridge.c` — namespace bridge
- `tools/gen_text.c` — wubuwizard inference entry point
- `src/wubu_model.c` — wubuwizard model loading
- `include/wubu_model.h` — wubuwizard public API