# taskweft plans

HTN planning domains for the recommender roadmap, solved with the
[`taskweft`](https://github.com/V-Sekai-fire/multiplayer-fabric-taskweft) MCP planner
(`plan` tool). Kept so estimates are **comparable over time**.

## Layout

- `domains/*.jsonld` — the re-plannable HTN domains (source of truth). Actions carry
  an ISO-8601 `"duration"`; taskweft's STN schedules them and returns a total.
  `_estimate_basis` records where the durations came from.
- `snapshots/YYYYMMDD-<name>.json` — a dated capture of one `plan` run: the ordered
  plan, the temporal schedule (start/end per step), the `total`, and the `host_basis`
  (measured unit costs + assumptions) it rested on.

## Re-plan and compare

Re-solve a domain (via the taskweft MCP `plan` tool, or the CLI):

    taskweft plan priv/plans/domains/real-data-pretrain-path.jsonld

Write a new `snapshots/<today>-<name>.json`, then diff the `total` and per-step
durations against the previous snapshot. Estimates improve as real numbers land
(dataset size, GPU epoch time) — each re-estimate is a new dated snapshot, never an
edit of an old one, so the history is preserved.

## Current snapshots

| snapshot | domain | total | basis |
|---|---|---|---|
| `20260714-real-data-pretrain-path.json` | real-data pretrain path | **PT12H19M** | CPU-EXLA measured: tokenize 12171 items/s, FuXi ~3.7s/step; GPU train assumed ~100k sessions × 3 epochs |

The broader 18-step roadmap (baseline → multimodal embedder → upgrade) lives in
`domains/multimodal-recommender-roadmap.jsonld` (structural, no durations yet).
