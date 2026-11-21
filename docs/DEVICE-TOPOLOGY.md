# Device Topology — inference where it belongs

## Decision

Do **not** port the full engine to the distiller ARM devices. They are 8 GB RAM, 4-core,
no GPU, and already run always-on agent stacks (Claude Code, Codex, OpenClaw, :3000 web UI).
Porting 12 x86-intrinsic files to ARM buys a slow engine and loses the agents.

Instead: **inference stays on the 6 GB VRAM box; devices are agent endpoints and output.**

```
   distiller CM4 (.251, offline)      distiller CM5 (.70, online)
   e-ink 240x416 portrait slate       always-on agent host
        |                                  |
        +------- thin client --------------+
                     |
              HTTP /v1/chat/completions
                     |
        wubuwizard api_server on x86 + 6 GB GPU
                     |
        Gemma 4 / Qwen / Inkling GGUF
```

## Assets already present

- `tools/api_server.c` — OpenAI-compatible HTTP server: `/v1/chat/completions`,
  `/v1/completions`, `/v1/models`, `/health`; optional TLS and API key; POSIX + Win32
  sockets. This is the integration surface, not a port.
- `distiller-c11/src/neon/neon_dot_q4k_q8k.c` — ARM NEON quant dot kernel. Keep as a
  reference for a future thin ARM backend only; do not deploy it now.
- `distiller-device-ops` archived skill — device SSH, e-ink service, known CM4 bugs.

## Roles

| Node | Role | Why |
|------|------|-----|
| x86 + 6 GB GPU | Inference service + palace owner | Weights, KV, quant decode, CUDA |
| CM5 `distiller-bb8c` (192.168.1.70) | Always-on agent, task loop, client | Already has agent stacks, 7.2 GB free RAM, 43 GB disk |
| CM4 (192.168.1.251, offline) | Human-readable e-ink slate | 240x416 portrait panel; physical palace output |

## CM5 work loop (what "not idle" means)

1. Read the project palace from the shared repo (`state.md`, `slate.md`, `goal-mantra.md`).
2. Pick the lowest undone gated item.
3. POST the task to `/v1/chat/completions` on the inference box.
4. Write the verified result back into the palace with a DA pass.
5. Sleep, repeat. No new engine binary on the device.

## Gate for this topology

- `curl <inference-box>:8080/health` from the CM5 returns 200.
- A task round-trip produces a palace update with a verification tag.
- The CM4 e-ink path is only gated after the device is back online; do not assume it works.

## Known display defect (CM4)

The compiled `libdistiller_display_sdk_shared.so` defaults to 250x128 landscape and cannot
drive the CM4's 240x416 portrait panel. The C11 path in `distiller-c11` has the correct
dimensions and the correct partial-LUT sequence (`0xE5 0x6E` + `0x50 0xD7`), but the README
incorrectly calls the packing LSB-first (the code is MSB-first). Fix docs and the SDK config
before trusting the panel; verify with the isolation ladder in the archived device skill.

## Non-goals

- No full engine on ARM.
- No CUDA requirement on the devices.
- No new global config file; the palace stays per-project markdown.
