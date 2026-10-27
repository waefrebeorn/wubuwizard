#!/bin/bash
# Build tower manifold test

cd /home/wubu/bytropix

# Compile with existing bytropix includes and math
gcc -std=c11 -O2     -Iinclude     -IENCODERS/hash-mind/c/include     -I/home/wubu/bytropix/power_tower     power_tower/tower_manifold.c     power_tower/test_tower.c     ENCODERS/hash-mind/c/wubu_math.c     -lm -o power_tower/test_tower

echo "Build complete: power_tower/test_tower"
