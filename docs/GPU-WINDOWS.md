# Windows GPU path — RTX 4050 Laptop, CUDA 13.5

Verified 2026-11-22 through the real build entrypoint:

```
mingw32-make.exe -f Makefile.win test_gpu_smoke
```

Output:

```
GPU device=NVIDIA GeForce RTX 4050 Laptop GPU cc=8.9 vram=6438780928 bytes
GPU silu kernel err=no error
GPU silu maxerr=0.00000012
GPU cublas c=[19 22 43 50]
gpu_smoke: PASS
```

## What was broken

1. `Makefile.win` named CUDA `v12.4` and `sm_75`. The installed toolkit is
   `v13.5` (`C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.5`) and the
   device is compute capability 8.9. Both are corrected.
2. `nvcc` on Windows requires the MSVC x64 host environment. `make` on this
   host cannot spawn `cmd.exe` or any recipe containing shell metacharacters
   (`&&`, `. script`), so `vcvars64.bat` cannot be called from a recipe.
   `Makefile.win` now exports `INCLUDE`, `LIB`, `LIBPATH` and a space-free
   `PATH` captured from `vcvars64.bat`, and invokes `nvcc.exe` by its 8.3 path
   with no shell metacharacters.
3. cuBLAS/cuRUNTIME DLLs live in `CUDA/v13.5/bin/x64`, not `bin`; the runtime
   PATH includes both.

`tools/win_nvcc.bat` and `tools/nvcc_win.sh` are retained as diagnostic
wrappers for interactive use; the Makefile does not depend on them.

## Known separate defect

Compiling the whole `src/cuda_kernels.cu` on Windows still fails with 12
errors. The SSM kernels use `SSM_D_STATE`, `SSM_K_HEADS` and `DT_RANK`, which
expand to fields of the runtime `__constant__ WUBU_DIMS_DEV`, as template
arguments and `__shared__` array sizes. Those require compile-time constants.
This is tracked separately from the GPU smoke gate; the smoke gate uses a
self-contained SiLU kernel plus cuBLAS sgemm so device/toolchain correctness
is measured without that unrelated port bug.

All performance figures in Strata/paper research remain authors' numbers; no
local throughput claim is made by this gate.
