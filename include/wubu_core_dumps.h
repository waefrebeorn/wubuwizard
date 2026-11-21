#ifndef WUBU_CORE_DUMPS_H
#define WUBU_CORE_DUMPS_H

/**
 * wubu_core_dumps.h — Core dump prevention for WSL/Linux
 *
 * Call wubu_disable_core_dumps() at the start of main() before loading
 * large models (16GB+) to prevent WSL's wsl-capture-crash from writing
 * massive crash dump files that fill the disk.
 *
 * Usage:
 *   #include "wubu_core_dumps.h"
 *   int main() {
 *       wubu_disable_core_dumps();
 *       // ... rest of main
 *   }
 */

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/prctl.h>
#endif

static inline void wubu_disable_core_dumps(void) {
#ifndef _WIN32
    struct rlimit rl = {0, 0};
    setrlimit(RLIMIT_CORE, &rl);
    prctl(PR_SET_DUMPABLE, 0);
#endif
}

#endif // WUBU_CORE_DUMPS_H