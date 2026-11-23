# Strata v0.2.19 — mechanism harvest (2026-11-22)

Source: https://github.com/Niko1221/Strata/releases/tag/v0.2.19
Announcement: https://x.com/coldniko/status/2124468831175549262

All performance numbers below are the authors' reported figures, not local
measurements. Cloned/downloaded Strata artifacts are study material only;
nothing is vendored into `wubuwizard`.

## CPU/portable mechanisms worth a gate

1. **SQ quantization contract.** Three packs (SQ-10/15/20) are characterized by
   top-1 agreement with original weights and test KL divergence: SQ-15 reports
   96.68% top-1 agreement and 0.00889 KL; SQ-20 reports 100.00% / 0.00000 and is
   called bit-exact lossless. Absorb as an agreement/KL gate over logits before
   accepting a quantized build. → EG-101.
2. **SF4+MX8 block microscaling.** SQ-20 is described as a bit-exact native
   `SF4+MX8` build. Absorb the block-scale + FP8-exponent representation and a
   bit-exactness gate independent of the GPU kernel. → EG-102.
3. **Zero-Logits-D2H serving.** The Inkling path advertises 0.00000 KLD while
   avoiding full-logits device-to-host transfer. Absorb as a gate that transfers
   only sampled/verified token ids and measures bytes avoided. → EG-103.
4. **W=4 Multi-Branch On-GPU Markov Beam Tree `DSpark`.** A Markov-branch
   speculative tree with zero-logits handoff. The tree verifier (EG-21/91) exists;
   the branch-selection policy does not. → EG-104.

## Already covered locally

- Direct `safetensors` / `.gguf` loading: `src/safetensors_reader.c` and
  `src/wubu_gguf_tokenizer.c` already own the header/parse path. No new gap.
- Quantized KV modes: EG-36a (E4M3 KV) and EG-43 (K4V4 draft KV) already cover
  the family named in the release's community acknowledgements.
- Mapped-memory NUMA doorbell, `--gpus 2` Expert Parallelism, `dense_fp4_tc`,
  `moe_fp4_tc`, `vhead_bitplane`, RDNA3 Wave32 WMMA and SYCL/XMX dispatch are
  device-specific or multi-GPU and are deferred, not gated on this 6 GB box.

## Reporting rule

The release's tok/s and prefill figures are device results for RTX 5090, RTX
PRO 5000/6000, M5 Max, R9700, Strix Halo, DGX Spark and Arc B70. None are
reproducible here; no wubuwizard gate may cite them as local evidence.
