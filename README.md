# residual-fsq-recommender

Generative next-item recommender (**FuXi-Linear** linear-attention) over **residual FSQ semantic
IDs**, in Elixir on [Nx](https://github.com/elixir-nx/nx)/EXLA. The ID codec is
certified in Lean 4 via [`fire/plausible-witness-dag`](https://github.com/fire/plausible-witness-dag).

Each item is a 4-token semantic ID `[t0, t1, t2, t3]`, each token in `0..4095`, produced by a
coarse-to-fine residual FSQ quantizer (`levels = [8,8,8,8]`, 4 stages). A linear-attention decoder
predicts the next item's four tokens; a trie constrains decoding to real catalog IDs.

## Layout (hexagonal — `core` / `ports` / `adapters`)

- `Recommender.Core.FSQ` / `Recommender.Core.ResidualFSQ` — the residual FSQ codec: content
  embeddings → 4-token IDs, plus the ID contract (`valid_id?/1`, `tokens_per_item/0`,
  `codebook_size/0`) every consumer shares.
- `Recommender.Core.FuxiLinearInference*` — the FuXi-Linear model (Retention + linear temporal /
  positional channels + mFFN), O(n) in sequence length.
- `Recommender.Core.{Decode,Trie,Training,Eval}` — trie-constrained beam / multi-token-prediction
  decode, training losses (shifted cross-entropy + MTP), and Hit@k / MRR metrics.
- `Recommender.Adapters.Serve` — wires the forward pass + constrained decode + catalog into
  `recommend/3`.
- `Recommender.Adapters.CockroachStore` / `VersityBlobStore` — embedded CockroachDB (items and
  session transitions) and a Versity S3 blob store; the host lifecycles come from
  [`weftspun/cockroach-local`](https://github.com/weftspun/cockroach-local) and
  [`weftspun/versitygw-local`](https://github.com/weftspun/versitygw-local).
- `Recommender.Adapters.CLI` — the standalone `rfr` binary.

Coarse mass lands in `t0`, so similar items share ID prefixes; the decoder generates an ID prefix-first
and the trie keeps every beam on a valid catalog path.

## Semantic IDs

`Recommender.Core.ResidualFSQ` tokenizes content embeddings without a trained checkpoint (a seeded
projection makes it reproducible), or loads a trained projection from a checkpoint:

```elixir
ids = Recommender.Core.ResidualFSQ.encode_ids(embeddings)   # [[t0, t1, t2, t3], ...]
true = Recommender.Core.ResidualFSQ.valid_id?(hd(ids))
```

The packing `itemKey [t0,t1,t2,t3] = t0 + t1·4096 + t2·4096² + t3·4096³` is injective over valid IDs
(certified in Lean), so an ID maps to one catalog slot.

## Serving

`Serve` runs the trained model over a catalog (`token_id_list`, index = item index) and returns the
top-k next-item indices:

```elixir
# defn_params from a trained checkpoint (Recommender.Adapters.NpyCheckpointSource +
# Recommender.Core.FuxiLinearInferenceParams.build_defn_params/2)
serve = Recommender.Adapters.Serve.new(defn_params, token_id_list)
{:ok, item_indices} = Recommender.Adapters.Serve.recommend(serve, [0, 3], 5)
```

Recommendation needs a trained checkpoint; `Serve.from_init/2` builds a random-weight engine that
exercises the full pipeline for smoke tests. Training runs on a GPU host
(`Recommender.Adapters.AxonTrainer` + `Recommender.Core.Training`) over gate-clean corpora
(`Recommender.Core.Trajectories.LicenseGate`: CC0 / MIT / Apache-2.0 / CC-BY, English/Western).

## CLI (`rfr`)

```
rfr add <item_id> <t0> <t1> <t2> <t3>          store an item's semantic ID
rfr add --json                                 bulk add from stdin
rfr observe <id> <id> [...]                    record a session's transitions
rfr recommend <id> [...] --checkpoint <dir>    next-item recall (JSON)
rfr items [--limit N]                          list stored items (JSON)
rfr blob put|get|list                          content-addressed blob store
rfr db start                                   run the embedded CockroachDB
```

Persistence goes through the driven ports; the standalone binary bundles the `cockroach` and
`versitygw` single binaries per target.

## Formal model (`formal/`)

`RecommenderModel.lean` (built on `plausible-witness-dag`) certifies the ID codec by `omega` — symbolic,
so it holds at the real 4096-code / 4096⁴-key scale: `stage_bound`, `stage_roundtrip`,
`stage_injective`, and `itemKey_injective` (the stage index is a bijection onto `0..4095`, and distinct
IDs never collide).

```bash
cd formal
lake build
```

## Tests

```bash
mix deps.get
mix test          # runs on EXLA (config/config.exs)
```

## License

MIT — see [LICENSE.md](LICENSE.md).
