# CUDA sm_120 (Blackwell RTX 5050) — Hardware Bugs & Workarounds

**Discovered:** May 2026, bytropix GPU kernel development  
**GPU:** NVIDIA RTX 5050, sm_120, CUDA 13.1 toolkit  
**Status:** 3 confirmed bugs with applied workarounds in gpu_moe_kernel.cu v5

---

## Bug 1: `static __shared__` Inside Loops

**Symptom:** Kernels with `__shared__` arrays declared inside a for-loop body hang on sm_120 (Blackwell). The kernel launches but never completes — no error code, no NaN, just infinite hang.

**Affected pattern:**
```cuda
for (...) {
    __shared__ float smem[256];  // BUG: hangs on sm_120
    ...
    __syncthreads();
    ...
}
```

**Fix:** Replace with `extern __shared__` and manual offset calculation:
```cuda
extern __shared__ float smem[];
float *my_smem = smem + blockIdx.x * 256;  // manual partitioning
```

**Workaround cost:** ~20 more lines of pointer arithmetic for multi-block shared memory partitioning.

**Status:** Applied to `gpu_moe_kernel.cu` v5.

---

## Bug 2: `__syncthreads()` After Warp-Leader Shared Write

**Symptom:** Pattern where warp leaders write to shared memory, `__syncthreads()`, then selected threads (idx < NW) read and reduce — hangs on sm_120.

**Affected pattern:**
```cuda
if (threadIdx.x % WARP_SIZE == 0) {
    smem[warp_id] = peak_val;  // warp leader write
}
__syncthreads();               // BUG: hangs on sm_120
if (threadIdx.x < NW) {
    val = smem[threadIdx.x];   // read
}
```

**Fix:** Thread 0 reads all warp peaks and does a serial reduction. Alternative: use a single shared memory atomic max instead of the two-phase reduce.

```cuda
if (threadIdx.x == 0) {
    float max_val = -INFINITY;
    for (int w = 0; w < num_warps; w++) {
        if (smem[w] > max_val) max_val = smem[w];
    }
    result = max_val;
}
```

**Status:** Applied to `gpu_moe_kernel.cu` v5.

---

## Bug 3: `extern __shared__ uint8_t` with `__syncthreads()` In Loops

**Symptom:** `extern __shared__ uint8_t smem_u8[]` combined with `__syncthreads()` in loops causes incorrect code generation on sm_120. The compiler generates wrong code that produces garbage results.

**Affected pattern:**
```cuda
extern __shared__ uint8_t smem_u8[];
for (...) {
    // write to smem_u8
    __syncthreads();
    // read from smem_u8
    __syncthreads();
}
```

**Fix:** Use `extern __shared__ float smem[]` instead of `uint8_t`. Cast to `uint8_t*` only when accessing byte data.

```cuda
extern __shared__ float smem_f[];
uint8_t *smem_u8 = (uint8_t *)smem_f;
```

**Root cause hypothesis:** The compiler's aliasing analysis treats `uint8_t*` and `float*` as non-aliasing types (strict aliasing rule). When the compiler sees a loop with syncthreads, it reorders the uint8_t load before the float store (or vice versa), breaking the `__syncthreads()` ordering guarantee. Using `float*` for the declaration avoids this because the compiler can't reorder around a sync point when the type is uniform.

**Status:** Applied to `gpu_moe_kernel.cu` v5.

---

## Additional sm_120 Notes

### FP8 Tensor Cores
- sm_120 supports native FP8 dot product instructions
- Not yet used in bytropix — current kernels use FP32 only
- Would provide ~2x throughput for batched quant matmul
- Blocked on GPU data-movement problem (H2D/D2H overhead makes any GPU text net-negative)

### Register Pressure
- Blackwell has 65536 registers per block (same as Ada)
- Current kernels use ~32 regs/thread at 512 threads = ~16K registers
- Plenty of headroom for larger blocks or more unrolled loops

### Shared Memory
- 48KB per block on RTX 5050
- Current kernels use ~8KB
- Could increase block size to 1024 threads if shared memory budget allows

### Multi-Block Parallelism
- 32 blocks per SM theoretically possible
- Current kernels use 1 block per SM
- Limiting factor: sequential expert uploads + H2D sync points

### Thermal Throttling
- GPU init heats CPU package, skewing CPU benchmark timing
- Sustained GPU load (>5 min) causes VRAM temp to throttle performance
- Observed: GPU hybrid slower than CPU-only even for compute-heavy tasks
