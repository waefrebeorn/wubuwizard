# IQ2_M Output Quality Analysis (Cell 180)

## Logit Statistics (Qwen3.6 35B A3B T50 IQ2_M, 248K vocab)

| Metric | Value |
|--------|-------|
| Shape | 248,320 |
| Mean | -1.58 |
| Std | 1.83 |
| Min | -12.47 |
| Max | 11.01 |
| Top-1 to Bottom-1 spread | 23.48 |
| Top-1 to Top-100 spread | 3.72 |
| Within 1.0 of max | 6 candidates |
| Within 2.0 of max | 20 candidates |
| Within 5.0 of max | 310 candidates |

## Temperature Calibration

| T | top-1 prob | top-5 | entropy |
|---|-----------|-------|---------|
| 0.3 | 29.6% | 0.296, 0.259, 0.214, 0.112, 0.063 | 1.74 |
| 0.5 | 20.5% | 0.205, 0.189, 0.168, 0.114, 0.081 | 2.61 |
| **0.7** | **12.7%** | **0.127, 0.120, 0.111, 0.084, 0.065** | **3.99** |
| 0.9 | 6.7% | 0.067, 0.064, 0.060, 0.049, 0.040 | 5.99 |
| 1.0 | 4.6% | 0.046, 0.044, 0.041, 0.034, 0.029 | 7.11 |
| 1.2 | 2.0% | 0.020, 0.019, 0.018, 0.016, 0.013 | 9.04 |

## Analysis

- **Logits are valid**: Range of 23.5 with mean -1.58 and std 1.83 is normal for a well-trained 248K vocab model.
- **Top-k competition**: Only 6 candidates within 1.0 of max, 20 within 2.0 — reasonable sharpness.
- **Softmax at T=1.0**: Top-1 probability 4.6% is on the low side for a 248K vocab. Ideal would be ~5-15%.
- **Temperature sweet spot**: T=0.7 gives top-1=12.7% with entropy 3.99 — good quality.
- **Cos-sim floor**: 0.974 vs llama.cpp is due to IQ2_M quantization noise in the output projection. The 0.026 residual manifests as slightly flattened logit distribution.

## Conclusion

**Not a bug — IQ2_M quantization floor.** The model produces usable output. Temperature calibration (T=0.7) compensates for the flatness. To exceed 0.99 cos-sim, need Q3_K+/F16 model (32GB+ RAM).

## References

- Cell 180 in battleship
- Previous cos-sim analysis: 0.974 = IQ2_M floor
- With F16 output projection: would restore full precision
