# Branch Consolidation Audit — 2026-08-09

**Result: NO CODE LOST.** All work lives in `origin/wubu-integration`
— the ONLY branch.

## What was retired

On 2026-08-09 the GitHub repository was consolidated from 7 branches
to **ONE** (`wubu-integration`), which is also the default branch.
Retired:

| Branch | Tip | Fate |
|---|---|---|
| `master` (old bytropix brand era) | `4699815` | 0 paths absent from trunk |
| `cpu-optimize-may26` | `816aea8` | only `.hermes` agent-scratch absent |
| `windows-port-support` | `0475569` | 3 paths absent, all empty LFS stubs |
| `interim-organization` (2025 reorg WIP) | `0b62bb9` | 97/140 files exact in trunk; rest superseded drafts or old-brand junk |
| `unified` (stale remote mirror) | `78b34f6` | 0 paths absent |
| `lfm25-adapter` (LFM/Distiller agent) | `4f26058` | 0 unique commits; 2 paths absent, both empty LFS stubs — the agent's work was ALREADY in the trunk |

The LFM/Distiller agent now works on the SAME trunk as everyone else.

## How the audit was done

For each retired branch tip, `git ls-tree -r` enumerated every path;
every path was checked against `origin/wubu-integration`'s tree:

1. **Path-level**: is the path present in the trunk at all?
2. **Content-level**: is the blob (content hash) present in the trunk?
3. **Same-name search**: if content differs, does the trunk hold a
   newer version of the same file (superseded, not lost)?

## The only non-trunk content (and why it's fine)

- `cpu-optimize-may26`: 45 files in `.hermes/mind-palace/vault/tmp-tools/`
  (agent diagnostic scratch, superseded by the 803 real `tools/` files)
  + agent state docs.
- `interim-organization`: root copies of research scripts whose newer
  versions live under `ENCODERS/`, `DIFFUSION/`, `DRAFT/` in the trunk;
  old MIT `.license` (superseded by License v3.0); result PNGs; an old
  `.pkl` checkpoint.
- `windows-port-support`: 3 empty (0-byte) LFS pointer stubs.

## Permanent recovery (the guarantee)

Every retired branch tip is preserved as an annotated tag, so nothing
can ever be garbage-collected away:

```
archive/retired-master                  -> 4699815
archive/retired-cpu-optimize-may26      -> 816aea8
archive/retired-windows-port-support    -> 0475569
archive/retired-interim-organization    -> 0b62bb9
archive/retired-unified-mirror          -> 78b34f6
archive/retired-lfm25-adapter           -> 4f26058
```

To recover any file from a retired branch:

```bash
git show archive/retired-interim-organization:WuBuMindV7.1.py
```

## Re-running the audit

```bash
for t in archive/retired-master archive/retired-cpu-optimize-may26 \
         archive/retired-windows-port-support archive/retired-interim-organization \
         archive/retired-unified-mirror; do
  echo "== $t =="
  git diff --name-status origin/wubu-integration $t | head
done
```
