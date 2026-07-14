# Sequential recommendation with HRR primitives only

No ResidualFSQ, no neural network, no training. Everything below is built from the
four `Holo.Core.HRR` operations — `bind` (phase add), `unbind` (phase subtract),
`bundle` (circular mean), `similarity` (phase cosine) — plus one order primitive,
`permute` (a fixed cyclic shift `ρ`), which is the standard VSA way to make binding
non-commutative and thus position-aware (Gayler 2003; Kanerva 2009).

An item is just its atom: `vec(id) = encode_atom(id)` — SHA-256 deterministic, zero-shot,
recommendable the instant it has an id. Optionally bundle in text/entity atoms
(`Memory.item_vector/3` already does) as the *only* source of content generalization now
that there are no learned embeddings.

## 1. The task

Given a session `s = [x₁, …, xₜ]` (oldest → newest), rank catalog items by
`P(x_{t+1} = c | s)`. Two HRR signals, combined:

### Content (order-agnostic, zero-shot)
Bundle the recent window into one probe and rank by similarity — items that share
atoms (same id, shared text tokens, shared entities) score high. This is what
`Memory.recommend/3` already computes for the content term.

### Transition (order-aware, online)
A first-order successor memory. For an observed step `a → b` store the trace
`bind(vec a, vec b)`. Recall unbinds the last item: `unbind(T, vec xₜ) ≈ vec x_{t+1}`,
then a cleanup scan ranks catalog atoms against that noisy estimate. `unbind(bind a b) b = a`
is **exact on the SHA uint16 phase grid** — proved by `omega` as `HoloModel.unbind_bind`
in `formal/HoloModel.lean`; the f64 path adds only representation noise.

Order beyond the immediate predecessor comes from the trajectory encoding
(Plate 2003): a windowed context vector
`ctx = bundle( ρ⁰·vec xₜ, ρ¹·vec x_{t-1}, …, ρ^{k-1}·vec x_{t-k+1} )`
binds each item to its distance-from-now via permutation power, so "A then B" and
"B then A" produce distinct probes. The successor memory can then be keyed on `ctx`
instead of the bare last item to capture higher-order context — still pure HRR.

## 2. The capacity wall — why one bank is not enough

A single superposition of `M` traces has retrieval SNR ≈ `√(dim / M)`
(`HRR.snr_estimate/2`; Frady, Kleyko & Sommer 2018 give the full accuracy curve, this
is its leading term). Recall degrades once SNR drops below ~2, i.e.

```
M_max ≈ dim / 4          # dim=1024  ⟹  ~256 traces per bank
```

MovieLens-100K has 100 000 ratings → far more than 256 distinct transitions. Piling them
into one bank `T` turns it to noise. **This is the "limit the items in each bucket" wall.**

## 3. Bucketing: bounded banks, targeted search

Shard the one global bank into `B` **buckets**, each an independent superposition kept
under `M_max`, and route every read/write to exactly one bucket so no bank overflows and
no query scans the whole catalog.

Route by the **source** item (all of `a`'s successors must land together so a single
probe recovers them), using a SHA-derived hash for cross-language determinism:

```
bucket(a) = <first 4 bytes of sha256(id_a)>  mod  B
```

Per bucket `b` keep two things:

| field         | what                                              | HRR? |
|---------------|---------------------------------------------------|------|
| `bank[b]`     | `bundle( bind(vec a, vec bᵢ) )` over its traces   | yes  |
| `roster[b]`   | the set of distinct successor ids written to `b`  | no (a plain `MapSet`) |

- **write `a → b`:** `bank[bucket(a)] ⊕= bind(vec a, vec b)`; `roster[bucket(a)] ∪= {b}`.
- **read given last `xₜ`:** `probe = unbind(bank[bucket(xₜ)], vec xₜ)`; score **only**
  `roster[bucket(xₜ)]` by `similarity(probe, vec c)`. Cleanup is bounded by the roster,
  not `N` — this is the "still searchable" half.

Everything stays HRR: the bank is a bundle of binds, recall is an unbind + phase-cosine.
The hash and roster are just the index that keeps each bundle inside its capacity.

### Degenerate but clean special case — one bank per source (`B = N`)
`bank[a] = bind(vec a, bundle(successors of a))`, so `unbind(bank[a], vec a) =
bundle(successors of a)` — a holographic first-order Markov row. Per-source load =
distinct successors of `a`, almost always ≪ 256. Simple, always in-capacity; loses
cross-source generalization (recovered separately by the content term).

### Saturation handling
Monitor `snr_estimate(dim, load[b])`. When a hot source pushes its bucket past `M_max`,
either (a) raise `B` and rehash, or (b) roll that bucket into a fresh **generation** and
sum probes across generations at read time — bundle capacity is per-bank, so generations
never interfere. Whichever is chosen, `log` the spill so silent truncation never reads as
full coverage.

## 4. Sizing — bounds from the Lean model + VSA theory

Two facts pin the numbers:

1. **Single-shot retrieval is exact** on the phase grid — `HoloModel.unbind_bind`
   (`omega`, holds at the real 65536-per-component / 1024-dim scale). So *all* recall
   error is superposition noise, nothing else.
2. **Budgeted cleanup resolves iff the scan budget reaches the target.** The
   `plausible-witness-dag` ladder in `HoloModel` demonstrates this: `holoLevels` runs
   `walkSteps = 2` (budget below the successor's position → **budget-hit**) then
   `walkSteps = 4` = full roster → **resolves** item 3 at L1. Translated to buckets: a
   query's cleanup budget need only cover `|roster[b]|`, not `N`. Bucketing is what makes
   `walkSteps ≪ N` sufficient.

Combine with the SNR bound `M_b < dim/4`:

```
to hold L total transitions with every probe above threshold:
    B  ≥  L / (dim/4)  =  4L / dim
per-bucket cleanup budget:  |roster[b]|  (≈ L/B distinct successors)
```

For MovieLens-100K at `dim = 1024`: `L ≤ 100 000` ⟹ `B ≥ 391` buckets keeps every bank
under the 256-trace wall, and each query touches one bank + a few-hundred-item roster
instead of the full catalog. Raising `dim` to 4096 cuts the bucket count 4× (`M_max ≈ 1024`).

## 5. Scaling the search — resonator networks

When context is a *product* of factors (position ⊛ item ⊛ user, say) the key space is
combinatorial and even a roster scan is too much. A **resonator network** (Frady, Kent,
Olshausen & Sommer 2020) factors a bound product by searching *in superposition* — it
recovers the components without enumerating their cross-product, and is pure VSA
(binds + bundles + cleanup iterated). That is the drop-in when per-bucket rosters
themselves grow structured; it does not change any of the algebra above.

## References

See `CITATION.cff` for full entries. Core: Plate 1995 (HRR + capacity), Plate 2003
(trajectory association), Gayler 2003 (VSA), Kanerva 2009 (HDC), Frady/Kleyko/Sommer 2018
(superposition capacity bound), Frady/Kent/Olshausen/Sommer 2020 (resonator search),
Quadrana/Cremonesi/Jannach 2018 (sequence-aware recommendation task).
