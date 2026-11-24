# Benchmark smoke summary

| variant | profile | pp | gen | avg t/s | sd | CV |
|---|---|---:|---:|---:|---:|---:|
| baseline | agent | 1024 | 0 | 40.6110 | 0.2050 | 0.00505 |
| baseline | agent | 0 | 32 | 15.4405 | 0.5128 | 0.03321 |
| baseline | chat | 128 | 0 | 127.1299 | 3.0842 | 0.02426 |
| baseline | chat | 0 | 32 | 15.1199 | 0.5854 | 0.03872 |
| baseline | coding | 512 | 0 | 44.2207 | 0.3069 | 0.00694 |
| baseline | coding | 0 | 64 | 15.7629 | 0.0800 | 0.00507 |
| baseline | saturation | 512 | 0 | 43.7625 | 0.3533 | 0.00807 |
| baseline | saturation | 1024 | 0 | 40.7336 | 0.0446 | 0.00110 |
| baseline | saturation | 2048 | 0 | 36.7312 | 0.0497 | 0.00135 |
| baseline | saturation | 0 | 64 | 15.4152 | 0.4034 | 0.02617 |
| baseline | saturation | 0 | 64 | 15.5243 | 0.1531 | 0.00986 |
| baseline | saturation | 0 | 64 | 15.3808 | 0.0240 | 0.00156 |
| loopopt | agent | 1024 | 0 | 327.0702 | 0.6230 | 0.00190 |
| loopopt | agent | 0 | 32 | 20.6503 | 0.7044 | 0.03411 |
| loopopt | chat | 128 | 0 | 195.3117 | 3.2597 | 0.01669 |
| loopopt | chat | 0 | 32 | 19.4188 | 0.8107 | 0.04175 |
| loopopt | coding | 512 | 0 | 349.8160 | 0.6491 | 0.00186 |
| loopopt | coding | 0 | 64 | 20.8893 | 0.4813 | 0.02304 |
| loopopt | saturation | 512 | 0 | 351.3036 | 0.8881 | 0.00253 |
| loopopt | saturation | 1024 | 0 | 326.2504 | 1.6453 | 0.00504 |
| loopopt | saturation | 2048 | 0 | 299.4931 | 0.5387 | 0.00180 |
| loopopt | saturation | 0 | 64 | 20.8107 | 0.4603 | 0.02212 |
| loopopt | saturation | 0 | 64 | 21.1256 | 0.3763 | 0.01781 |
| loopopt | saturation | 0 | 64 | 20.8348 | 0.5740 | 0.02755 |
