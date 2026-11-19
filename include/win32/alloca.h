/* include/win32/alloca.h -- MinGW keeps alloca in <malloc.h>. */
#ifndef WUBU_WIN32_ALLOCA_H
#define WUBU_WIN32_ALLOCA_H
#if defined(_WIN32)
#include <malloc.h>
#else
#include_next <alloca.h>
#endif
#endif
