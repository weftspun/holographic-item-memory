# Drop the HRR baseline; ship the pure FuXi-Linear generative recommender

**Status:** Accepted · 2026-07-14 · **Supersedes** [Switch the recommender to FuXi-Linear over Residual FSQ semantic IDs](20260713-fuxi-linear-residual-fsq.md)

## Context and problem statement

The prior decision ([20260713](20260713-fuxi-linear-residual-fsq.md), Accepted) added the
FuXi-Linear generative recommender over residual FSQ semantic IDs and **kept
`Recommender.Core.Memory` (the training-free HRR memory) as an apples-to-apples baseline** consuming
identical IDs.

With the port complete and running on EXLA, the HRR path is now redundant to how the system is used:
the product is the trained FuXi-Linear model, and the persistence/serving path only needs the
semantic-ID catalog, not HRR's phase algebra. Carrying HRR meant maintaining a second recommender,
a second ID consumer, and an HRR-shaped session store (`transitions(prev, next, n)` pairwise counts)
that the generative model does not use.

## Decision

1. **Remove the HRR recommender from the Elixir tree** — delete `Recommender.Core.Memory` and
   `Recommender.Core.HRR` (and their tests / golden fixture). FuXi-Linear (`Recommender.Adapters.Serve` over
   `Recommender.Core.{FuxiLinearInference*, Decode, Trie}`) is the sole recommender.

2. **Rehome the semantic-ID contract** (`tokens_per_item/0`, `codebook_size/0`, `valid_id?/1`) from
   `Recommender.Core.Memory` into `Recommender.Core.ResidualFSQ` — the codec that emits the IDs — as the single
   source of truth every consumer (CLI, store adapters, `Serve`) shares.

3. **Rewire the persistence read path.** `Recommender.Ports.ItemSource.load_memory/3` (which built a
   `Recommender.Core.Memory`) is replaced by `load_catalog/2`, returning `{token_id_list, item_ids}` — the
   catalog `Recommender.Adapters.Serve` consumes (index = model item index).

4. **Keep CockroachDB.** Persistence stays the embedded `Recommender.Adapters.CockroachStore` behind the
   `ItemSource`/`ItemSink` ports (a custom Ecto/DuckDB adapter was considered and rejected — see
   below). The `items` schema is migrated from 3 to **4 tokens** (`t0..t3`) to match the certified
   contract. The `transitions` table is retained as generic session logging.

5. **CLI `recommend` now serves through the model.** `rfr recommend <id…> --checkpoint <dir>` loads
   trained FuXi-Linear params (npy export) and runs `Serve`; without `--checkpoint` it errors (there
   is no longer a training-free recommender).

## Consequences

- **Positive:** one recommender, one ID contract, less surface. The Lean `RecommenderModel` codec proofs
  (`itemKey_injective`, stage round-trip) still certify the substrate.
- **Negative / trade-offs:** the training-free floor is gone — recommendation now requires a trained
  checkpoint, so `rfr recommend` is inert until a model is trained on a GPU host (gate-clean data:
  Amazon CC0 / OTTO CC-BY). The `docs/sequential-recall.md` HRR-vs-trained comparison becomes
  historical rather than a live baseline.
- The Lean HRR **phase-algebra** proofs (`bind_comm`, …) remain in `formal/RecommenderModel.lean` as formal
  artifacts; they are no longer exercised by Elixir code. Removing them is out of scope.

## Rejected alternatives (persistence)

Replacing CRDB with a swappable Ecto layer was explored and dropped in favor of keeping CRDB:
- **Ecto + SQLite** — real ecto_sql semantics, serverless, but a second store to maintain.
- **Custom `Ecto.Adapter` on Explorer** — no ecto_sql semantics (no SQL / migrations); large build.
- **Custom DuckDB `Ecto.Adapters.SQL.Connection`** — true SQL-over-Arrow with ecto_sql semantics, but
  a multi-week, library-sized effort with no existing `ecto_duckdb`.

Decision: **keep the embedded CockroachDB**, already wired behind the ports (so it remains swappable).
The CRDB host provisioning may later be extracted into a dedicated dependency (`weftspun/cockroach_local`).
