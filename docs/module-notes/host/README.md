# host module notes

Goal state, agent sessions, telemetry, benchmark statistics and decision policy.

## Named gates (4)

- `eg106_bench_stats`
- `eg32_compact_session`
- `eg39_agent_spec`
- `eg39_agentstate`

## Reuse checklist

1. Read the owning header and source end to end before editing.
2. Confirm the gate links the owning source directly.
3. Extend the existing seam rather than creating a parallel module.
4. Add a REGISTER evidence paragraph with exact measurements.
5. Rebuild the full `test_eg*` suite and record the tally.
