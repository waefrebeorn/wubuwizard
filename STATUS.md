# Status — implementation and verification state

> 2026-08-09 refresh (Phase 0 truth sync — the INDEX is the source of
> truth; AN27–AN39 are the closed-loop colony organs). Every claim
> below was verified by running the command on this date (or the
> session it names). Weights policy: GGUF = SSD
> (`/home/wubu/models/`), safetensors = SD cold storage
> (`/home/wubu/sdcard/models/`, mount `D:` first).

## Verified this wave (2026-08-09 — the AGI colony, AN27–AN39)

| Subsystem | Evidence | Command |
|---|---|---|
| **Closed control loop** (AN27) | every batch -> Diagnosis -> hive fitness cells -> amoeba mutate -> validate (loss tol + Lean prover) -> archive/graveyard; wired into the REAL trainer via `--diag-every` | `make test_diagnosis` (20 batches -> 20 hive cells, bad mutation rejected -> graveyard, recovery path PASSES) |
| **Hive walk** (AN28) | `wubu_diag_save` + `tools/wubu_hive_walk` — lineage + graveyard queryable (the agent's introspection surface) | `wubu_hive_walk <checkpoint>.hive --accepted` (verified end-to-end on a live 30-batch run) |
| **Specialist-cell orchestrator** (AN29, Pillar 8) | goal decomposer spawns 4 lens cells (code/math/tool/critique), judge merges by argmax confidence, critique VETO | `make test_orch` (4 spawns, judge picks the strongest lens, veto keeps the goal, 9 hive cells) |
| **RLHF oracle** (AN30) | Bradley-Terry pairs FROM the hive mutations, per-cell survival credit assignment | `make test_pref` (cell 0.5 -> 0.677 on win, -> 1.000 after 13 wins, loser falls) |
| **Live Colonel** (AN31) | Brain->Body channel: 9P capability trust boundary at enqueue, priority pull + backoff, durable checkpoints | `make test_colonel` (out-of-subtree denied, critical first, 3 requests survive checkpoint) |
| **Self-critique + recovery** (AN32) | failed generations -> graveyard + shrink pressure + immediate mutation cycle | covered by `make test_diagnosis` (recovery verdict + graveyard +1) |
| **RSI mutation engine** (AN33, P20) | trace/span operator writes INTO the hive; gate/LADDER/bounded-delta proposals the amoeba calls | `make test_selfimprove` (spans in hive, bad mutation blocked, good passes + decomposes) |
| **Gradient-health telemetry** (AN34) | `grad_norm_sum`/`micro_steps` were DEAD — now accumulates the per-layer grad norm mean per microbatch | live run: `selfimprove: spans=10 proposed=5 accepted=5 fails=0` |
| **Lineage + extinction** (AN35) | parent chains, multi-gen survival as fitness, SOFT EXTINCTION, near-miss cousins | `make test_lineage` (winner 0.94 survival, stagnant 0.14 -> extinct) |
| **Tool-trajectory cells** (AN36) | typed hive cells (goal/steps/outcome/cost/hash); similar-goal queries; failed+partial-credit -> mutation seeds | `make test_trajcell` (4 cells, similar=2, failure is the seed) |
| **Dual-timescale diagnose** (AN37) | fast per-batch + slow scheduled colony-state META-CELL adjusting rate/floor | `make test_metadiag` (rising tail raises rate 0.50->0.57, improving lowers to 0.44) |
| **Capability-gated spawning** (AN38) | headroom/tool/verifier/redundancy checks; DECOMPOSE/FALLBACK/DENY; gaps are first-class cells | `make test_capgate` (all 4 verdicts, 2 gaps visible) |
| **Runtime contracts** (AN39) | Lean floor in runtime form (ball/exp/route/quant/finite); loss-improving but contract-violating mutations REJECTED; versioned meta-cells | `make test_contracts` (clean passes, violating rejected 2 violations, NaN caught) |
| **Phase 1: closed loop = DEFAULT** (AN40) | `--diag-every 1` default (every batch diagnoses + mutates); the gate = fitness + contracts + lineage; finite guard every step; ASan-clean (0 leaks) | `./wubu_train --init-random ... --steps 12` (7 accepted / 5 rejected, lineage 12, contracts 35 checks 0 violations); `gcc -fsanitize=address test_diagnosis` EXIT 0 |
| **Phase 3: autonomy harness** (AN41) | 7-task fixed suite, deterministic scorers, outcomes as traj cells, suite score = first-class fitness (failing suite raises the rate even when loss improves), loopguard-bounded | `make test_harness` (4 pass/2 fail, score 0.657, stops at step ceiling) |
| **Phase 4: skill curriculum** (AN42) | traj patterns -> versioned skill cells; drafts gate-accepted; orchestrator matches before spawning; unused skills pruned | `make test_skillcell` (3 drafts -> 3 accepted, match finds the 0.9 skill, 1 pruned) |
| **Phase 5: deny-by-default tool registry** (AN43) | unregistered tools DENIED; every action a traj cell; thrashing tools auto-barred | `make test_toolreg` (denied/registered/barred all correct) |
| **Phase 6: executable blueprint** (AN44) | WB01's lineage as machine-readable bounds; off-blueprint mutations refused; bounds move only via a versioned expansion meta-cell | `make test_blueprint` (in-range passes, MoE 64 refused, expansion versioned) |
| **Phase 2: priority store** (AN46) | per-cell BI + online Fisher/EWC + precision deltas + mutation ledger; the diagnose gate REFUSES mutations on protected cells (high Fisher + recent rejection); `.prio` sidecar next to the `.hive` archive; wired into wubu_diag_cycle + the CLI | `make test_priority_store` (Fisher 0.562, protected refused, sidecar round-trips); live run: 9 cells, 384 consults, sidecar 392 bytes |
| **Endurance + restart** (AN47) | `--resume` loads `.hive` + `.prio` (wubu_diag_load restores ledger + counters, cells replay into the hive); `--ckpt-hive N` mid-run sidecars; PROVEN kill/restart | 8-step run → resume → "resumed 3 fitness cells" + 4 more batches continue |
| **Release gate** (AN47) | `test_colony` — the "no ship without it" gate: harness floor + zero contract violations + prio sidecar + blueprint bounds | `make test_colony` (harness 0.643, contracts 35 checks 0 violations, prio 392B, blueprint intact — ALL PASS) |
| **GEMV dispatch** (AN48) | the precision ladder EXECUTES: >=16 bits → F32/BF16, <16 → INT8; reladder switches the backend (re-quantize, not paper) | `make test_gemv_dispatch` (dense 8→INT8, reladder 8→32 switches, INT8 rel err 0.0049) |
| **Live-file harness** (AN49) | REAL file work: byte-exact text chunk + the Kodak codec round-trip (shape corr 0.707) → traj cells | `make test_harness_file` (text pass, degenerate fail, codec pass) |
| **Skill → train** (AN50) | accepted skills → preference pairs, high-value traj → SFT segments; the stream drains into the next window | `make test_skill_train` (2 pairs + 2 SFT → drain 4) |
| **Colonel effect** (AN51) | the end-to-end chain PROVEN: specialist → Colonel request → toolreg gate → the file lands → traj + cap accounting | `make test_colonel_effect` (denied → registered → effect landed) |
| **Policy recorder** (AN52) | every metadiag policy change is a versioned meta-cell with the reason code (1=loss rising 2=suite failing 3=relax) | `make test_metadiag` (6 policy meta-cells recorded) |
| **Lineage merge** (AN53) | two `.hive` archives merge (each with its own priority evidence): union + better-fitness + the protected override + graveyards union | `make test_hive_merge` (3 cells, protected survives, 2 rejections preserved) |
| **Body handoff** (AN54) | `docs/BODY-PILLAR-HANDOFF.md` — the frozen Brain→Body message set + cap semantics; unblocks the WuBuOS executor | doc + `make test_colonel_effect` as the reference chain |
| **Chaos upgraded** (AN56) | suite-continuity assert + the A2 double-kill drill (kill during sidecar writes) | `tools/chaos_kill_resume.sh` — 3 kills → resumed suite 0.836, 847 events 0 violations; 4 rapid kills → 1378 events, no torn sidecars |
| **Skill persistence** (AN57) | the learned skills ride a `.skills` sidecar (K7/K8) — the nightly gate caught the resume re-learning from zero | resume restores 233 skills, the replay boundary divergence GONE; `make test_skillcell` |
| **Nightly gate** (AN58) | fixed budget + one forced resume + published artifacts | `tools/test_colony_nightly.sh` — 60s smoke PASSED (0 violations, suite 0.836, replay agrees) |
| **Resource cells** (AN59) | per-200-round RSS/CPU/throughput hive meta-cells + the metadiag soft-fitness trigger (reason 4 fires on resource stress even with fine loss) | `test_metadiag` (stress + fine loss → rate 0.60); live: rss 14.6MB soft 1.00 |
| **Skill decay** (AN60) | matched skills get the real outcome: fails decay fitness (EMA→0.1), a consistently-misleading skill auto-demotes toward prune | `make test_skillcell` (0.90→0.44 after 8 fails, recovers on good); live suite 0.836, 85% pass |
| **Qwen3.5 role split** (AN61) | the AN28 open question resolved: the fused attn_qkv 6144 = q_and_gate(4096)+k(512)+v(512)+z(1024) — the 1024 is the output-gate tail (attn_output_gate:true), not a missing head; GDN in_proj 8192 = qk+v+z; layer kinds from the config (6×[3 GDN+1 gated]) | `make test_qwen35` (all 0.8B numbers pinned); the forward + logit-parity remain next once the GGUF is on disk |

## Verified this wave (2026-08-09 — acceleration + build integrity)

| Subsystem | Evidence | Command |
|---|---|---|
| **GPU elementwise kernels** | sigm-bwd/rmsnorm-bwd/unrope/residual-add CUDA kernels, FD-verified against finite differences | `make test_backprop` ("all 26 parameter checks match the numeric gradient") |
| **GPU A/B** | GPU ON 3.21s/step vs OFF 61.7s/step (19×) — all 241 matmuls + 16 attentions hit cuBLAS | `tools/test_step_bench.c` |
| **Fold sincos8** | AVX2 8-wide folded sin/cos, 10.7× faster, 5.96e-8 accuracy (better than libm fp32) | `make test_fold_sincos8` |
| **Fast exp8** | AVX2 8-wide bit-trick exp, 14.6× faster than libm, rel err 3.9e-6 | `tools/test_fast_exp.c` |
| **Build integrity** | `wubu_dense_ffn.o` wired into all ~80 link targets (was failing test_nested_ssm -> load_model -> test_model chain) | `make all` = 0 errors |
| **Runtime-dims rename** | wubu35_dims -> wubu_runtime_dims (agnostic naming, user-mandated) | `make test_runtime_dims` |

## Verified (earlier waves — require SD card mounted for real-weight claims)

| Subsystem | Evidence | Command |
|---|---|---|
| SSM forward (real weights) | Agents-A1-4B live BF16 forward, finite logits | `make test_real_load` (weights on SD) |
| ds4-ssd slot-bank | KAT 256 experts/layer paged from source shards, all finite | `./test_kat_decode_bank <KAT dir> 16` |
| HF BPE tokenizer | 248,044-vocab Qwen tokenizer round-trip | `./gen_text "<prompt>"` |
| Model config adapter | real KAT/Qwen3.6/Agents dims from config.json | `make test_model_config` |
| LoRA merge | BTL-3 base+adapter, delta applied, finite | `make test_btl3_lora` |
| Repetition (DRY + repeat-penalty) | wired to F16 params | `make test_repetition` |
| Mixed export (5.12×) | seed F32 140.3 MB → mixed GGUF 27.4 MB | `make test_tensor_store` |
| mHC multi-head | all oracles maxdiff=0 | `make test_wubu_mhc_mh` |
| SFT run | loss 8.04 → 7.32 @ step 2000 (seq 2048) | `tools/wubu_train_cli.c` run log |
| Storage policy | GGUF on SSD / safetensors on SD, verified byte-exact | `du -sb` src vs dst |

## Build gates

- `make test_all` — the full gate (299 test targets; subset gates exist).
- The colony gates: `make test_diagnosis test_orch test_pref test_colonel test_selfimprove test_lineage test_trajcell test_metadiag test_capgate test_contracts` — ALL PASS (2026-08-09).
- Engine gates: `make test_optim test_fold_sincos8 test_runtime_dims` — ALL PASS (2026-08-09).
- `make all` = 0 errors (2026-08-09).
- Regenerate module tables: `python3 tools/repodoc/repodoc.py . --readme --modules`.
