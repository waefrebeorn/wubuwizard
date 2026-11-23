#!/usr/bin/env bash
# Invoke nvcc with the MSVC x64 environment. Makefile.win calls this script
# because GNU make on this host cannot spawn cmd.exe directly.
set -e
bat="$(cd "$(dirname "$0")" && pwd)/win_nvcc.bat"
batwin="$(cygpath -w "$bat")"
MSYS2_ARG_CONV_EXCL='*' cmd.exe /c "$batwin" "$@"
