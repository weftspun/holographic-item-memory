# residual-fsq-recommender

Generative next-item recommender (FuXi-Linear linear-attention) over residual FSQ semantic
IDs, in Elixir on [Nx](https://github.com/elixir-nx/nx)/EXLA.

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
