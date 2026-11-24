/* bench_gpu_stages.c -- call the DLL's per-stage benchmark. */
#include <stdio.h>
#include <windows.h>
typedef int (*fn_t)(int);
int main(int argc, char **argv)
{
    HMODULE h = LoadLibraryA(argc > 1 ? argv[1] : "wubu_cuda.dll");
    if (!h) { printf("SKIP no dll\n"); return 0; }
    fn_t b = (fn_t)(void *)GetProcAddress(h, "wubu_cuda_bench_stages");
    fn_t w = (fn_t)(void *)GetProcAddress(h, "wubu_cuda_bench_wpb");
    if (!b || !w) { printf("no bench symbol\n"); return 1; }
    fn_t x = (fn_t)(void *)GetProcAddress(h, "wubu_cuda_bench_xfer");
    b(20);
    if (x) x(200);
    fn_t sh = (fn_t)(void *)GetProcAddress(h, "wubu_cuda_bench_shapes");
    fn_t op = (fn_t)(void *)GetProcAddress(h, "wubu_cuda_bench_oproj");
    if (op) printf("[BENCH] oproj rc=%d\n", op(30)); else printf("[BENCH] oproj missing\n");
    if (sh) sh(0);
    return w(0);
}
