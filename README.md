# residual-fsq-recommender

Generative next-item recommender (FuXi-Linear linear-attention) over **ResidualFSQ semantic IDs**.
Next-item recall in Elixir; the ID codec is certified in Lean 4 via
[`fire/plausible-witness-dag`](https://github.com/fire/plausible-witness-dag).

Companion to [`weftspun/multimodal-semantic-ids`](https://github.com/weftspun/multimodal-semantic-ids)
(successor line of `vsk-session-item-recommendation-01`): that repo's Python pipeline encodes each
asset's modalities (text / image / mesh / audio / body-phenotype) with FOSS encoders, concatenates
them into one fused vector, and quantizes it with a single ResidualFSQ
(`levels = [8,8,8,8]`, `num_quantizers = 3`) into a per-asset **semantic ID** — 3 tokens, each in
`0..4095`. This repo consumes those IDs as plain `{item_id, [t0, t1, t2]}` pairs and answers
"what comes next in this session?" without any neural model or training step.

Two modules, one runtime dependency (`Nx`):

- `Recommender.Core.HRR` — phase-vector algebra; SHA-256-deterministic atoms; f64 `Nx` tensors.
- `Recommender.Core.Memory` — the recommender: immutable struct, no processes, no storage backend.

## How it works

Holographic Reduced Representations (Plate 1995) over phase vectors: every concept is a vector of
angles in `[0, 2π)`, generated deterministically from SHA-256 — identical across machines and
languages (bit-for-bit parity with the Python reference is tested).

| Operation | Implementation | Meaning |
|-----------|----------------|---------|
| `bind(a, b)` | phase addition (mod 2π) | associate two concepts |
| `unbind(m, k)` | phase subtraction | retrieve: `unbind(bind(a,b), a) ≈ b` |
| `bundle(vs)` | circular mean | superpose; holds `O(√dim)` items |
| `similarity(a, b)` | `mean(cos(a−b))` | phase cosine, `[-1, 1]` |

The semantic ID **is** the item representation (`Recommender.Core.Memory.item_vector/3`):

```
item = bundle(bind(atom("sid:q0:t0"), ROLE_Q0),
              bind(atom("sid:q1:t1"), ROLE_Q1),
              bind(atom("sid:q2:t2"), ROLE_Q2))
```

Items sharing coarse (early-stage) ResidualFSQ tokens land near each other — the quantizer's
coarse-to-fine structure becomes vector similarity. Two recall signals combine in `Recommender.Core.Memory`:

- **Content (zero-shot):** the session's recent item vectors are bundled; candidates are ranked by
  similarity. A brand-new asset is recommendable the moment it has an ID: encode → atoms → vector.
  No retraining.
- **Transitions (online, optional):** observed `a → b` steps superpose into one hetero-associative
  bank `T = bundle(bind(vec a, vec b), …)`. Probing `unbind(T, vec last)` yields a noisy `vec next`;
  cleanup against the catalog ranks it. Capacity `O(√dim)`; `snr_estimate/1` warns past it.

## Usage

```elixir
# pairs from wherever you read asset_semantic_id.parquet: {item_id, [t0, t1, t2]}
mem =
  Recommender.Core.Memory.new(dim: 4096)
  |> Recommender.Core.Memory.add_items([
    {"sword-01", [17, 900, 3]},
    {"shield-03", [17, 901, 44]},
    {"helm-11", [2000, 31, 999]}
  ])
  |> Recommender.Core.Memory.observe(["sword-01", "shield-03", "helm-11"])   # optional online learning

{:ok, recs} = Recommender.Core.Memory.recommend(mem, ["sword-01", "shield-03"], top_k: 5)
# => [{"helm-11", 0.41}, ...]
```

## Formal model (`formal/`)

`RecommenderModel.lean` (Lean 4.30, built on `plausible-witness-dag`) certifies, by `omega` — symbolic, no
enumeration, so it holds at the real 4096-code / 4096³-key scale:

- `stage_bound` / `stage_roundtrip` / `stage_injective` / `itemKey_injective` — the upstream
  ResidualFSQ ID format this library consumes: the stage index codec is a bijection onto `0..4095`,
  and distinct semantic IDs never collide. (Deliberately no Elixir mirror — the theorems certify
  the contract itself.)
- `unbind_bind` — HRR retrieval is **exact** on the uint16 phase grid the atoms are generated on;
  the float implementation adds only representation noise.
- Transition recall is certified as a `plausible-witness-dag` witness: a budgeted cleanup scan
  recovers the stored successor; the shallow ladder rung budget-hits, the deeper rung resolves.

```bash
cd formal
lake build                    # type-checks the omega proofs
lake exe recommender-sample   # -> resolved level: L1 ; recalled successor: item 3
```

## Tests

```bash
mix deps.get
mix test
```

Includes golden-value parity tests against the Python reference HRR implementation
(`test/fixtures/hrr_golden.json`, regenerable with `scripts/gen_golden.py`).

## License

MIT — see [LICENSE.md](LICENSE.md).
