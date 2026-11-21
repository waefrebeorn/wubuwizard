# Device Topology — two devices, two SDKs, two screens, one network

## Correct the assumptions

These are **two different devices with different panels, SDKs and pinouts**. Treating them
as one platform is exactly what produced unusable e-ink code. Do not share a display module,
a resolution, an SDK call, or a GPIO map between them.

| | CM4 "Distiller One" v1 | CM5 "Distiller Alpha" v2 |
|---|---|---|
| Address | 192.168.1.251 (offline) | 192.168.1.70 (`distiller-bb8c`, online) |
| SoC | CM4 / Cortex-A72, no dotprod | CM5 Rev 1.0, ARMv8.2 with ASIMDDP |
| Panel | **240x416**, UC8253, portrait | **128x250** (W x H), landscape |
| Panel truth | `docs.pamir.ai/hardware_overview`: 240(H) x 416(V), 130 DPI | upstream `distiller-eink-cli`: "standard e-ink is **128x250**, not 250x128"; 1-bit, 4000 raw bytes |
| SDK on device | old `DistillerSDK` Python `EinkDSP` is the working path | `/opt/distiller-sdk`, `eink.conf` firmware `EPD128x250`, platform `cm5` |
| Upstream SDK | `Pamir-AI/DistillerSDK-distiller-one-` (archived) | `Pamir-AI/distiller-cm5-sdk` (installed at `/opt/distiller-cm5-sdk` upstream; **absent here**) |
| Pins | dc=6, rst=13, busy=9 | dc=7, rst=13, busy=9 |
| C11 driver | `distiller-c11` hardcodes 240x416, 30 bytes/row, MSB packing | same file is **wrong** for this panel; do not reuse |

Verified live on `.70`: only `/opt/distiller-sdk` exists; `libdistiller_display_sdk_shared.so`
is present; `eink.conf` says `EPD128x250`; `distiller_cm5_sdk` is **not installed**;
`/home/distiller/projects/eink-ui` does not exist; no e-ink/display service is running;
`caddy-distiller.service` is failed. The screen is doing nothing because nothing drives it.

## Assumptions that made the display unusable

1. **Resolution order.** 250x128 was assumed; upstream says 128x250 (width x height).
   Every composition and byte count follows from this.
2. **One SDK for both.** The CM4 works through the old Python `EinkDSP`; the CM5 needs
   `distiller_cm5_sdk`, which is not present on this unit.
3. **Compiled SDK default applied to CM4.** `libdistiller_display_sdk_shared.so` defaults to
   the CM5 panel and cannot drive the CM4's 240x416 portrait panel.
4. **C11 driver reused across generations.** `distiller-c11/src/hw/einkdrv` is correct for
   CM4 (dims, LUT `0xE5 0x6E` + `0x50 0xD7`, persistent session) but wrong for CM5.
5. **Packing documented backwards.** `distiller-c11` README says LSB-first; `pack.c` emits
   MSB-first (`1u << (7 - x%8)`), matching `einkdrv`. Fix the doc, not the code.
6. **No display service on CM5.** A screen with no writer is an idle screen.

## What the devices are for

Both are always-on ARM nodes on the LAN with agent stacks (Claude Code, Codex, OpenClaw,
:3000 UI). Idle nodes are **compute**, with a screen as one output capability:

```
     network agents (this box, CM5, others)
              |  a2a:// / HTTP task
              v
     WuBuFederation (AID, sponsor, capability)
              |
     task scheduler / work loop
        /                \
   CM5 node             CM4 node
   128x250 SDK          240x416 SDK
        \                /
   WuBuOS: WuBuHIVE + WuBuKernel
              |
   eventual WuBu P2P sharded inference network
```

## Contracts to plug into

- `WuBuFederation/JOINING.md` + `agent-protocol.json`: classes `IN`/`WI`, `AID`, named
  sponsor, `--endpoint a2a://host:port`, `--capability`. Nodes register capabilities like
  `compute.eink.cm5.128x250` and `compute.eink.cm4.240x416`. **The ledger is append-only —
  do not join on their behalf without an explicit instruction.**
- `WuBuOS/docs/ECOSYSTEM.md`: `WuBu` is the P2P distributed inference network; `WuBuHIVE`
  is the memory substrate; `WuBuKernel` supplies dispatch. That is the eventual runpod home.
- `wubuwizard/tools/api_server.c`: OpenAI-compatible `/v1/chat/completions` and `/health`.
  Heavy inference stays on the 6 GB GPU box; ARM nodes call it.
- `distiller-c11/src/neon/neon_dot_q4k_q8k.c`: ARM NEON quant dot — reference for a thin
  future ARM backend, not a deployed engine.

## Work loop per node ("not idle")

1. Register its own capability with the federation (sponsor + AID).
2. Pick the lowest undone gated item from the project palace.
3. Do local work (C11 render/dither/pack, preprocess, small jobs) or POST to the GPU API.
4. Render through **that node's SDK** to **that node's panel** — never the other's.
5. Write the verified result back into the palace with a DA pass; sleep, repeat.

## Gates

- `health`: node reachable, agent stack responsive.
- `capability`: registered AID claims and completes one task.
- `screen`: bold black/white pattern on the correct panel, verified by the isolation ladder
  in `distiller-device-ops`; trust the user's description of the panel, not a screenshot.
- CM4 is gated only after it returns to the network.

## Repairs before use

- **CM4:** use the old `EinkDSP` path or the C11 driver with correct dims/LUT; fix the
  `distiller-c11` README packing claim; wait for the device to come back online.
- **CM5:** install `distiller_cm5_sdk` from upstream or bind the compiled SDK to
  `EPD128x250`; start an actual display writer; repair `caddy-distiller.service`.

## Non-goals

- No full engine on ARM; no CUDA on the devices.
- No shared screen/SDK assumption between generations.
- No federation join without explicit instruction.
- No new global config; the palace stays per-project markdown.
