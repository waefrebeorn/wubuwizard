# `DIAGRAMS/` — SVG Architecture & Pipeline Visualizations

**SVG diagrams for the wubuwizard inference engine.**

## Current Diagrams

| Diagram | Description | Status |
|---------|-------------|--------|
| `inference-pipeline-v22.svg` | **Full inference pipeline**: per-layer data flow, quant types, 3:1 SSM/GQA interleaved, cos-sim badges, Q4_0 KV cache, honest status (VERIFIED/PARTIAL/BROKEN) | ✅ **Current (May 19 PM)** |
| `wubuwizard-lego-pieces.html` | **Lego-piece map of all 114 named EG gates**, generated from `Makefile.win`; grouped by family with benchmark and DA cards | ✅ Current (Nov 23, 2026) |
| `phase-roadmap.svg` | Project phase roadmap (may be stale) | 🟡 May 17 |
| `bug-status.svg` | Dequant verification status vs llama.cpp | 🟡 May 17 |
| `quant-layer-map.svg` | MoE down_exps tensor types by layer | 🟡 May 17 |
| `status-may19-2026.svg` | Status snapshot | May 19 |
| `paper-audit.svg` | Qwen3.6 params vs C implementation cross-ref | May 17 |
| `research-timeline.svg` | Research arc Aug 2025 → May 2026 | May 17 |
| Others | Training pipeline, tailslayer, gguf-rip, math, hamilton encoder | May 15-17 |

## Style Guide (for new SVGs)

Use `inference-pipeline-v22.svg` as template:
- **Background**: `#1a1a2e` (dark)
- **Headers**: `#e94560` (bright red), `#0f3460` (deep blue)
- **Code blocks**: `#16213e` with `#00ff88` text
- **Status colors**: ✅ green `#00ff88`, 🟡 yellow `#ffd700`, ❌ red `#ff4444`, 💤 gray `#666688`
- **Width**: 1200px, height variable
- **Font**: monospace (Consolas, Courier New)

## How to View
Open any SVG in a browser. All are standalone files (no external dependencies).
