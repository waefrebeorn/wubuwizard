# bytropix Wiki — Schema

## Purpose
Evolving schema for the bytropix LLM Wiki. Defines conventions for entity pages, concept pages, wikilinks, and ingest workflow.

## Directory Structure
```
.hermes/mind-palace/
├── raw/                    # Immutable research sources
│   ├── papers/             # PDFs, arxiv links
│   ├── articles/           # Blog posts, docs
│   └── images/             # Diagrams, screenshots
├── wiki/                   # Compiled, interlinked knowledge
│   ├── index.md            # Content catalog
│   ├── Overview.md         # Top-level entry
│   ├── entities/           # Model/architecture pages
│   ├── concepts/           # Technique/theory pages
│   ├── sources/            # Per-source summaries
│   └── synthesis/          # Cross-source analysis
├── log.md                  # Append-only chronological record
└── schema.md               # This file
```

## Conventions

### Page Naming
- Entities: `Model-Name.md`, `Architecture-Name.md` (Pascal case with hyphens)
- Concepts: `Concept-Name.md` (descriptive, hyphenated)
- Sources: `source-short-name.md` (lowercase, hyphenated)
- Synthesis: `topic-comparison.md`, `topic-survey.md`

### Wikilinks
- `[[Entity-Name]]` on first mention in each page
- Every entity/concept page must have ≥ 2 inbound links (or it's an orphan)
- Link text = page name (no aliases unless for readability)

### Frontmatter (optional, for Obsidian Dataview)
```yaml
---
type: entity | concept | source | synthesis
date: YYYY-MM-DD
sources: [source-name-1, source-name-2]
tags: [inference, gpu, ssm]
---
```

## Verification Tags (all wiki pages)
| Tag | Meaning |
|-----|---------|
| ✅ Verified | Cross-checked against source or runtime |
| 🟡 Partial | Partially verified, known gaps |
| ❓ Unchecked | Believed correct, not verified |
| 🔴 Contradicted | Contradicted by another source |
| ⚠️ Stale | Newer source may supersede |

## Current Wiki Contents

### Entities
| Page | Summary |
|------|---------|
| [[Qwen3.6-35B]] | 35B MoE model, Gated DeltaNet SSM |
| [[Moondream3]] | Vision model, 3D ViT + mmproj |
| [[Gated-DeltaNet]] | SSM with gating mechanism |

### Concepts
| Page | Summary |
|------|---------|
| [[SSM]] | State Space Models for sequence modeling |
| [[MoE]] | Mixture of Experts routing |
| [[GGUF]] | Weight quantization format |
| [[ViT]] | Vision Transformer architecture |
| [[mmproj]] | Multi-modal projection layer |
| [[Chunked-SSM]] | Chunked inference for SSM |
| [[MTP]] | Multi-Token Prediction spec decode |

### Sources
| Page | Type | Covers |
|------|------|--------|
| [[qwen36-tech-report]] | paper | Full Qwen3.6 architecture |
| [[deepseek-v3-report]] | paper | MoE + MLA architecture |
| [[mamba-paper]] | paper | SSM foundations |
